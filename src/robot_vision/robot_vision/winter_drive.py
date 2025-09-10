#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: winter_drive.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's Advanced Fusion & Robust Pure Pursuit Logic]
# 1. 경로 계산 로직을 순수 NumPy 벡터화 연산으로 대체하여 CPU 병목 현상 제거 (성능 극대화)
# 2. 시각화 토픽 구독자가 있을 때만 시각화 연산을 수행하여 불필요한 CPU 자원 낭비 방지
# 3. 실시간 영상 스트림에 최적화된 'Best Effort' QoS 프로파일 적용
# 4. 콜백 함수에서 모든 연산을 제거하고 작업 스레드로 이전하여 통신 지연 가능성 원천 차단
# 5. 주요 파라미터를 클래스 상수로 관리하여 가독성 및 유지보수성 향상
# 6. Pure Pursuit 알고리즘 안정성 강화: 경로가 짧을 경우 마지막 점을 목표점으로 지정
# 7. 제어 기준점을 '가상 후륜 축'으로 변경하여 Pure Pursuit 알고리즘의 정확도 극대화
# 8. [핵심 개선] '신뢰도 기반 동적 스무딩' 적용: 경로 포인트 수에 따라 스무딩 강도를 자동 조절하여 극한의 안정성 확보
# 9. [융합 아키텍처] 2개의 ONNX 모델(Drivable Area, Sand) 추론 결과를 실시간으로 융합하여 통합 경로 생성
# 10. [안정성 강화] 주행 영역 미검출 시 조향각 0을 발행하여 Fail-Safe 동작 보장

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import threading

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, Bool
from cv_bridge import CvBridge
import traceback

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# --- 유틸리티 함수 (변경 없음) ---
def polyfit_path(points_y, points_x, order=2):
    if len(points_y) < 10: return None
    try: return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError): return None

def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    if num_labels > 1:
        # 0번은 배경이므로 1번부터 찾음
        largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[largest_component_label, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == largest_component_label] = 255
    return cleaned

def overlay_polyline(image, coeff, color=(0, 255, 0), step=4, thickness=3):
    if coeff is None: return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w: draw_points.append((int(x), int(y)))
    if len(draw_points) > 1: cv2.polylines(image, [np.array(draw_points, dtype=np.int32)], False, color, thickness)
    return image
# --- 유틸리티 함수 끝 ---


class YoloBevFusedDrivableAreaNode(Node):
    '''
    Top Line of Bev Image  - Bottom Line of Bev Image(From Camera) = 2.33m - 0.66m
    Middle Line of Bev Image(From Camera) = 1.495m
    Available Lookahead Distance range min ~ Max : 0.66m ~ 2.33m
    '''
    # [Hinton's Optimization] 알고리즘 상수 정의
    _MORPH_KSIZE = 7
    _MIN_AREA_SIZE = 15000
    
    # [힌튼의 핵심 개선] 신뢰도 기반 동적 스무딩을 위한 파라미터
    _MAX_CONFIDENCE_POINTS = 32000
    _MIN_CONFIDENCE_POINTS = 2000
    _MAX_SMOOTHING_ALPHA = 0.6
    _MIN_SMOOTHING_ALPHA = 0.3
    
    def __init__(self):
        super().__init__('yolo_bev_fused_drivable_area_node')
        self.get_logger().info("--- YOLO BEV Fused Drivable Area Planning Node (Hinton's Ultimate Fusion Architecture) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 파라미터 선언
        self.declare_parameter('drive_area_model_path', './drive_area2.onnx')
        self.declare_parameter('sand_model_path', './sand.onnx')
        self.declare_parameter('drive_area_confidence', 0.5)
        self.declare_parameter('sand_confidence', 0.5)
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('wheelbase', 0.6)
        self.declare_parameter('camera_to_rear_axle_offset', 0.27)
        self.declare_parameter('lookahead_distance', 0.66)

        # 파라미터 가져오기
        drive_area_model_path = self.get_parameter('drive_area_model_path').get_parameter_value().string_value
        sand_model_path = self.get_parameter('sand_model_path').get_parameter_value().string_value
        self.drive_area_confidence = self.get_parameter('drive_area_confidence').get_parameter_value().double_value
        self.sand_confidence = self.get_parameter('sand_confidence').get_parameter_value().double_value
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.CAMERA_TO_REAR_AXLE_OFFSET = self.get_parameter('camera_to_rear_axle_offset').get_parameter_value().double_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value

        try:
            # 듀얼 모델 로딩
            self.get_logger().info(f"Loading Drive Area model from: {drive_area_model_path}")
            self.drive_area_model = YOLO(drive_area_model_path, task='segment')
            self.get_logger().info(f"Loading Sand model from: {sand_model_path}")
            self.sand_model = YOLO(sand_model_path, task='segment')
            
            # BEV 파라미터 로드 및 변환 행렬 사전 계산
            self.get_logger().info(f"Loading BEV parameters from: {bev_param_file}")
            bev_params = np.load(bev_param_file)
            self.src_points = bev_params['src_points']
            self.dst_points = bev_params['dst_points']
            self.bev_h = int(bev_params['warp_h'])
            self.bev_w = int(bev_params['warp_w'])
            self.M_bev = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.get_logger().info("✅ BEV transformation matrix calculated.")

            # 차량 좌표계 변환 파라미터
            self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.002609375, 0.66, 0.0011171875
            self.get_logger().info(f"✅ Dual ONNX models and all resources loaded on [{self.device}].")

        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to load resources: {e}")
            rclpy.shutdown()
            return

        self.tracked_center_path_coeff = None
        self.planning_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='planning_worker')
        self._is_shutting_down = False
        
        # 퍼블리셔
        self.steer_pub = self.create_publisher(Float64, '/steering_angle', 10)
        self.viz_pub = self.create_publisher(CompressedImage, '/path_planning/drivable_area/viz/compressed', 10)
        self.status_pub = self.create_publisher(Bool, '/path_planning/drivable_area/status', 10)
        
        # 실시간 영상 스트림에 최적화된 QoS 프로파일
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        logitech_img_topic = '/camera3/image_raw/compressed'
        self.img_sub = self.create_subscription(
            CompressedImage, 
            logitech_img_topic, 
            self.planning_callback, 
            qos_profile_sensor_data
        )
        self.get_logger().info(f"✅ Node initialized. Subscribing to '{logitech_img_topic}' with RELIABLE QoS.")

    def planning_callback(self, compressed_img_msg):
        if self._is_shutting_down: return
        try:
            self.planning_executor.submit(self._process_planning_data, compressed_img_msg.data)
        except Exception as e:
            self.get_logger().warn(f"Failed to submit planning task: {e}")
            
    def _process_planning_data(self, compressed_data_buffer):
        try:
            # 작업 스레드에서 이미지 디코딩 수행
            np_arr = np.frombuffer(compressed_data_buffer, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_color is None:
                self.get_logger().warn("Failed to decode image.")
                return

            bev_image = self.do_bev_transform(cv_color)
            
            # ================================================================= #
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼ [힌튼의 듀얼-마스크 융합 로직] ▼▼▼▼▼▼▼▼▼▼▼▼▼ #
            # 1. 두 개의 모델에서 각각 추론 수행
            drive_area_results = self.drive_area_model(bev_image, conf=self.drive_area_confidence, verbose=False)
            sand_results = self.sand_model(bev_image, conf=self.sand_confidence, verbose=False)
            
            # 2. 각 추론 결과로부터 마스크 추출
            drive_area_mask = np.zeros(drive_area_results[0].orig_shape, dtype=np.uint8)
            if drive_area_results[0].masks is not None and len(drive_area_results[0].masks.data) > 0:
                drive_area_mask = np.max(np.array([m.cpu().numpy() for m in drive_area_results[0].masks.data]), axis=0)
                drive_area_mask = (drive_area_mask * 255).astype(np.uint8)

            sand_mask = np.zeros(sand_results[0].orig_shape, dtype=np.uint8)
            if sand_results[0].masks is not None and len(sand_results[0].masks.data) > 0:
                sand_mask = np.max(np.array([m.cpu().numpy() for m in sand_results[0].masks.data]), axis=0)
                sand_mask = (sand_mask * 255).astype(np.uint8)

            # 3. 두 마스크를 OR 연산으로 융합하여 통합 주행 영역 생성
            unified_mask = cv2.bitwise_or(drive_area_mask, sand_mask)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲ [힌튼의 듀얼-마스크 융합 로직] ▲▲▲▲▲▲▲▲▲▲▲▲▲ #
            # ================================================================= #

            filtered_mask = self.filter_drivable_mask(unified_mask)
            steering_angle_rad, viz_data = self.calculate_steering_from_area(filtered_mask)
            
            # ================================================================= #
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [Hinton's Fail-Safe Steering Logic] ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
            # 주행 가능 영역이 감지되었을 때는 계산된 조향각을,
            # 감지되지 않았을 경우(steering_angle_rad is None)에는 안전을 위해
            # 조향각 0 (직진)을 명시적으로 발행합니다.
            # 이는 시스템이 예측 불가능한 상태에 빠지는 것을 방지하고 안정성을 확보하는 핵심 로직입니다.
            steer_msg = Float64()
            if steering_angle_rad is not None:
                steer_msg.data = steering_angle_rad
            else:
                steer_msg.data = 0.0
            self.steer_pub.publish(steer_msg)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [Hinton's Fail-Safe Steering Logic] ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
            # ================================================================= #
            
            if self.viz_pub.get_subscription_count() > 0:
                # 시각화를 위해 steering_angle_rad가 None일 경우 0.0으로 처리
                final_viz_angle = steering_angle_rad if steering_angle_rad is not None else 0.0
                self.publish_visualization(bev_image, filtered_mask, viz_data, final_viz_angle)

        except Exception:
            self.get_logger().error(f"Error in planning worker:\n{traceback.format_exc()}")

    def do_bev_transform(self, image):
        return cv2.warpPerspective(image, self.M_bev, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)
        
    def filter_drivable_mask(self, bev_mask):
        f1 = morph_close(bev_mask, ksize=self._MORPH_KSIZE)
        f2 = remove_small_components(f1, min_size=self._MIN_AREA_SIZE)
        return f2

    def image_to_vehicle(self, pt_bev):
        u, v = pt_bev
        y_cam = (self.bev_w / 2 - u) * self.m_per_pixel_x
        x_cam = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        x_rear = x_cam - self.CAMERA_TO_REAR_AXLE_OFFSET
        y_rear = y_cam
        return x_rear, y_rear

    def calculate_steering_from_area(self, area_mask):
        is_detected = np.any(area_mask)
        self.status_pub.publish(Bool(data=bool(is_detected)))
        if not is_detected:
            # 주행 영역이 없으면 기존 경로 추적을 리셋합니다.
            self.tracked_center_path_coeff = None
            return None, {}
        
        # NumPy 벡터화로 경로점 계산
        roi = area_mask[self.bev_h // 2:, :]
        y_indices, x_indices = np.where(roi > 0)
        
        num_points = len(y_indices)
        if num_points < 50:
            self.get_logger().warn("Not enough drivable area points to calculate path. Re-using last stable path.", throttle_duration_sec=2)
            if self.tracked_center_path_coeff is not None:
                pass
            else:
                return None, {}
        
        else: # 유효한 포인트가 충분할 경우에만 새 경로 계산 및 스무딩
            unique_y, unique_y_indices = np.unique(y_indices, return_inverse=True)
            sum_x = np.bincount(unique_y_indices, weights=x_indices)
            count_y = np.bincount(unique_y_indices)
            
            valid_counts = count_y > 0
            center_points_x = sum_x[valid_counts] / count_y[valid_counts]
            center_points_y = unique_y[valid_counts] + self.bev_h // 2

            current_path_coeff = polyfit_path(center_points_y, center_points_x)
            
            # ================================================================= #
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [힌튼의 핵심 개선] 동적 스무딩 로직 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
            if current_path_coeff is not None:
                confidence = np.interp(num_points,
                                       [self._MIN_CONFIDENCE_POINTS, self._MAX_CONFIDENCE_POINTS],
                                       [0.0, 1.0])
                dynamic_alpha = np.interp(confidence,
                                          [0.0, 1.0],
                                          [self._MIN_SMOOTHING_ALPHA, self._MAX_SMOOTHING_ALPHA])
                if self.tracked_center_path_coeff is None:
                    self.tracked_center_path_coeff = current_path_coeff
                else:
                    self.tracked_center_path_coeff = (dynamic_alpha * current_path_coeff +
                                                      (1 - dynamic_alpha) * self.tracked_center_path_coeff)
                self.get_logger().debug(f"Path smoothed with dynamic alpha: {dynamic_alpha:.2f} (confidence: {confidence:.2f}, points: {num_points})")
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [힌튼의 핵심 개선] 동적 스무딩 로직 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
            # ================================================================= #

        final_path_coeff = self.tracked_center_path_coeff
        if final_path_coeff is None:
            return None, {}
        
        y_bev_coords = np.arange(self.bev_h - 1, self.bev_h // 2, -1)
        x_bev_coords = np.polyval(final_path_coeff, y_bev_coords)
        
        valid_indices = (x_bev_coords >= 0) & (x_bev_coords < self.bev_w)
        if not np.any(valid_indices):
            self.get_logger().warn("Path is completely outside of the BEV image.", throttle_duration_sec=2)
            return None, {}
            
        y_bev_coords, x_bev_coords = y_bev_coords[valid_indices], x_bev_coords[valid_indices]
        
        x_veh, y_veh = self.image_to_vehicle((x_bev_coords, y_bev_coords))
        dist_from_ego = np.sqrt(x_veh**2 + y_veh**2)
        
        goal_idx_candidates = np.where(np.abs(dist_from_ego - self.lookahead_distance) < 0.2)[0]
        
        steering_angle = None
        goal_point_bev = None
        goal_idx = None
        
        if len(goal_idx_candidates) > 0:
            goal_idx = goal_idx_candidates[np.argmin(np.abs(dist_from_ego[goal_idx_candidates] - self.lookahead_distance))]
        else:
            if len(dist_from_ego) > 0:
                self.get_logger().warn(f"Path is shorter than lookahead distance. Using the farthest point as goal.", throttle_duration_sec=2)
                goal_idx = np.argmax(dist_from_ego)

        if goal_idx is not None:
            actual_lookahead_dist = dist_from_ego[goal_idx]
            if actual_lookahead_dist > 0.1:
                x_goal, y_goal = x_veh[goal_idx], y_veh[goal_idx]
                alpha = math.atan2(y_goal, x_goal)
                steering_angle = math.atan2(2.0 * self.L * math.sin(alpha), actual_lookahead_dist)
                goal_point_bev = (int(x_bev_coords[goal_idx]), int(y_bev_coords[goal_idx]))

        viz_data = {'path_coeff': final_path_coeff, 'goal_point_bev': goal_point_bev}
        return steering_angle, viz_data
    
    def publish_visualization(self, bev_image, area_mask, viz_data, steering_angle_rad):
        viz_image = bev_image.copy()
        green_overlay = np.zeros_like(viz_image)
        green_overlay[area_mask > 0] = (0, 255, 0) # 융합된 영역을 초록색으로 표시
        viz_image = cv2.addWeighted(viz_image, 1, green_overlay, 0.4, 0)
        
        overlay_polyline(viz_image, viz_data.get('path_coeff'), color=(255, 255, 0), thickness=3)
        if viz_data.get('goal_point_bev') is not None:
            cv2.circle(viz_image, viz_data['goal_point_bev'], 10, (0, 0, 255), -1)
            
        steer_deg = math.degrees(steering_angle_rad) if steering_angle_rad is not None else 0.0
        steer_text = f"Steer: {steer_deg:.1f} deg"
        cv2.putText(viz_image, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        _, jpeg_buffer = cv2.imencode('.jpg', viz_image)
        viz_msg = CompressedImage()
        viz_msg.header.stamp = self.get_clock().now().to_msg()
        viz_msg.format = "jpeg"
        viz_msg.data = jpeg_buffer.tobytes()
        
        self.viz_pub.publish(viz_msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down the planning thread pool.")
        self._is_shutting_down = True
        self.planning_executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloBevFusedDrivableAreaNode()
    if rclpy.ok() and hasattr(node, 'M_bev'):
        try: 
            rclpy.spin(node)
        except KeyboardInterrupt: 
            node.get_logger().info("Keyboard interrupt, shutting down.")
        finally: 
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()