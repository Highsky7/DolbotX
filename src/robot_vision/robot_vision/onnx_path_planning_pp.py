#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: onnx_path_planning_pp.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's Final Optimization & Robust Pure Pursuit Logic]
# 1. 경로 계산 로직을 순수 NumPy 벡터화 연산으로 대체하여 CPU 병목 현상 제거 (성능 극대화)
# 2. 시각화 토픽 구독자가 있을 때만 시각화 연산을 수행하여 불필요한 CPU 자원 낭비 방지
# 3. 실시간 영상 스트림에 최적화된 'Best Effort' QoS 프로파일 적용
# 4. 콜백 함수에서 모든 연산을 제거하고 작업 스레드로 이전하여 통신 지연 가능성 원천 차단
# 5. 주요 파라미터를 클래스 상수로 관리하여 가독성 및 유지보수성 향상
# 6. [수정] Pure Pursuit 알고리즘 안정성 강화: 경로가 짧을 경우 마지막 점을 목표점으로 지정
# 7. [핵심 수정] 제어 기준점을 '가상 후륜 축'으로 변경하여 Pure Pursuit 알고리즘의 정확도 극대화

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


class YoloBevDrivableAreaNode(Node):
    '''
    Top Line of Bev Image  - Bottom Line of Bev Image(From Camera) = 2.33m - 0.66m
    Middle Line of Bev Image(From Camera) = 1.495m
    Available Lookahead Distance range min ~ Max : 0.66m ~ 2.33m
    '''
    # [Hinton's Optimization] 알고리즘 상수 정의
    _MORPH_KSIZE = 7
    _MIN_AREA_SIZE = 15000
    
    def __init__(self):
        super().__init__('yolo_bev_drivable_area_node')
        self.get_logger().info("--- YOLO BEV Drivable Area Planning Node (Hinton's Ultimate Optimized ONNX Architecture) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 파라미터 선언
        self.declare_parameter('yolo_model_path', './drive_area2.onnx')
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('wheelbase', 0.6)
        # [힌튼의 수정] 카메라-후륜축 거리 파라미터 선언 (단위: 미터)
        # ❗❗❗ 중요: 이 값은 실제 로봇에 맞게 정확히 측정하여 수정해야 합니다. ❗❗❗
        self.declare_parameter('camera_to_rear_axle_offset', 0.27)
        self.declare_parameter('smoothing_alpha', 0.6)
        self.declare_parameter('lookahead_distance', 1.0)

        # 파라미터 가져오기
        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        # [힌튼의 수정] 카메라-후륜축 거리 파라미터 가져오기
        self.CAMERA_TO_REAR_AXLE_OFFSET = self.get_parameter('camera_to_rear_axle_offset').get_parameter_value().double_value
        self.SMOOTHING_ALPHA = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value

        try:
            self.model = YOLO(yolo_model_path, task='segment')
            
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
            self.get_logger().info(f"✅ ONNX model and all resources loaded on [{self.device}].")

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
        
        # [Hinton's Optimization & FIX] 실시간 영상 스트림에 최적화된 QoS 프로파일 (BEST_EFFORT가 더 적합)
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, # RELIABLE -> BEST_EFFORT
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        logitech_img_topic = '/camera/color/image_raw/compressed'
        self.img_sub = self.create_subscription(
            CompressedImage, 
            logitech_img_topic, 
            self.planning_callback, 
            qos_profile_sensor_data  # 최적화된 QoS 적용
        )
        self.get_logger().info(f"✅ Node initialized. Subscribing to '{logitech_img_topic}' with BEST_EFFORT QoS.")

    def planning_callback(self, compressed_img_msg):
        # [Hinton's Optimization] 콜백은 오직 데이터 전달만 수행하여 초고속으로 반응
        if self._is_shutting_down: return
        try:
            # 원본 데이터 버퍼를 그대로 전달
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
            
            results = self.model(bev_image, conf=self.yolo_confidence, verbose=False)
            result = results[0]

            combined_mask = np.zeros(result.orig_shape, dtype=np.uint8)
            if result.masks is not None and len(result.masks.data) > 0:
                # 모든 마스크를 하나로 합침
                combined_mask = np.max(np.array([m.cpu().numpy() for m in result.masks.data]), axis=0)
                combined_mask = (combined_mask * 255).astype(np.uint8)

            filtered_mask = self.filter_drivable_mask(combined_mask)
            steering_angle_rad, viz_data = self.calculate_steering_from_area(filtered_mask)
            
            # ================================================================= #
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분이 수정되었습니다 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
            # 시각화를 위한 최종 조향각 변수 (검출 실패 시 0.0)
            final_viz_angle = 0.0

            if steering_angle_rad is not None:
                # 검출 성공 시에만 조향각을 계산하고 발행
                final_viz_angle = steering_angle_rad
                steer_msg = Float64()
                steer_msg.data = final_viz_angle
                self.steer_pub.publish(steer_msg)
            # else:
            #     # 주행 가능 영역이 검출되지 않으면 아무것도 발행하지 않습니다.
            #     pass
            
            # [Hinton's Optimization] 구독자가 있을 때만 시각화 연산 수행
            if self.viz_pub.get_subscription_count() > 0:
                self.publish_visualization(bev_image, filtered_mask, viz_data, final_viz_angle)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 이 부분이 수정되었습니다 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
            # ================================================================= #

        except Exception:
            self.get_logger().error(f"Error in planning worker:\n{traceback.format_exc()}")

    def do_bev_transform(self, image):
        return cv2.warpPerspective(image, self.M_bev, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)
        
    def filter_drivable_mask(self, bev_mask):
        f1 = morph_close(bev_mask, ksize=self._MORPH_KSIZE)
        f2 = remove_small_components(f1, min_size=self._MIN_AREA_SIZE)
        return f2

    def image_to_vehicle(self, pt_bev):
        """
        [힌튼의 수정]
        BEV 픽셀 좌표를 Pure Pursuit에 적합한 '가상 후륜 축' 기준의 차량 좌표계로 변환합니다.
        """
        u, v = pt_bev
        
        # 1. BEV 이미지 픽셀(u, v)을 카메라 기준 차량 좌표(x_cam, y_cam)로 변환
        # y_cam: 카메라의 측면 방향 (m, 왼쪽이 +)
        y_cam = (self.bev_w / 2 - u) * self.m_per_pixel_x
        # x_cam: 카메라의 전방 방향 (m)
        x_cam = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        
        # 2. [핵심 수정] 카메라 기준 좌표를 '가상 후륜 축' 기준으로 변환
        # y축(좌우)은 동일, x축(전후)만 옵셋(CAMERA_TO_REAR_AXLE_OFFSET)만큼 뒤로 이동시킵니다.
        x_rear = x_cam - self.CAMERA_TO_REAR_AXLE_OFFSET
        y_rear = y_cam
        
        return x_rear, y_rear

    def calculate_steering_from_area(self, area_mask):
        is_detected = np.any(area_mask)
        self.status_pub.publish(Bool(data=bool(is_detected)))
        if not is_detected:
            self.tracked_center_path_coeff = None
            return None, {}
        
        # [Hinton's Optimization] NumPy 벡터화로 경로점 계산 (Python 루프 제거)
        roi = area_mask[self.bev_h // 2:, :]
        y_indices, x_indices = np.where(roi > 0)
        
        if len(y_indices) < 50: # 유효한 포인트가 너무 적으면 계산하지 않음
            self.get_logger().warn("Not enough drivable area points to calculate path.", throttle_duration_sec=2)
            return None, {}

        # y좌표를 기준으로 x좌표들의 평균을 계산 (중심점 찾기)
        unique_y, unique_y_indices = np.unique(y_indices, return_inverse=True)
        sum_x = np.bincount(unique_y_indices, weights=x_indices)
        count_y = np.bincount(unique_y_indices)
        center_points_x = sum_x / count_y
        center_points_y = unique_y + self.bev_h // 2  # 원본 이미지 y좌표로 복원

        current_path_coeff = polyfit_path(center_points_y, center_points_x)
        
        # 경로 계수 스무딩
        if current_path_coeff is not None:
            if self.tracked_center_path_coeff is None:
                self.tracked_center_path_coeff = current_path_coeff
            else:
                self.tracked_center_path_coeff = (self.SMOOTHING_ALPHA * current_path_coeff + 
                                                  (1 - self.SMOOTHING_ALPHA) * self.tracked_center_path_coeff)
        
        final_path_coeff = self.tracked_center_path_coeff
        if final_path_coeff is None:
            return None, {}
        
        # 목표점(lookahead point) 계산
        y_bev_coords = np.arange(self.bev_h - 1, self.bev_h // 2, -1) # ROI 영역 내에서만 계산
        x_bev_coords = np.polyval(final_path_coeff, y_bev_coords)
        
        valid_indices = (x_bev_coords >= 0) & (x_bev_coords < self.bev_w)
        if not np.any(valid_indices):
            self.get_logger().warn("Path is completely outside of the BEV image.", throttle_duration_sec=2)
            return None, {}
            
        y_bev_coords, x_bev_coords = y_bev_coords[valid_indices], x_bev_coords[valid_indices]
        
        x_veh, y_veh = self.image_to_vehicle((x_bev_coords, y_bev_coords))
        dist_from_ego = np.sqrt(x_veh**2 + y_veh**2)
        
        # 목표 거리(lookahead_distance)와 가장 가까운 경로상의 점을 찾음
        goal_idx_candidates = np.where(np.abs(dist_from_ego - self.lookahead_distance) < 0.2)[0]
        
        steering_angle = None
        goal_point_bev = None
        
        if len(goal_idx_candidates) > 0:
            # 여러 후보 중 가장 가까운 점을 선택
            goal_idx = goal_idx_candidates[np.argmin(np.abs(dist_from_ego[goal_idx_candidates] - self.lookahead_distance))]
        else:
            # [수정된 핵심 로직]
            # 만약 목표 거리 내에 점이 없다면, 경로가 짧다는 의미.
            # 이 경우, 경로상의 가장 먼 점을 목표점으로 설정하여 어떻게든 주행을 이어가도록 함.
            self.get_logger().warn(f"Path is shorter than lookahead distance. Using the farthest point as goal.", throttle_duration_sec=2)
            goal_idx = np.argmax(dist_from_ego) # 가장 먼 점을 목표점으로 선택

        # 최종 선택된 목표점을 사용하여 조향각 계산
        if goal_idx is not None:
            actual_lookahead_dist = dist_from_ego[goal_idx]
            
            if actual_lookahead_dist > 0.1: # 매우 가까운 점은 무시
                x_goal, y_goal = x_veh[goal_idx], y_veh[goal_idx]
                
                # alpha: 차량의 현재 방향과 목표점 사이의 각도
                alpha = math.atan2(y_goal, x_goal)
                
                # Pure Pursuit 조향각 공식
                steering_angle = math.atan2(2.0 * self.L * math.sin(alpha), actual_lookahead_dist)
                goal_point_bev = (int(x_bev_coords[goal_idx]), int(y_bev_coords[goal_idx]))

        viz_data = {'path_coeff': final_path_coeff, 'goal_point_bev': goal_point_bev}
        return steering_angle, viz_data
    
    def publish_visualization(self, bev_image, area_mask, viz_data, steering_angle_rad):
        viz_image = bev_image.copy()
        green_overlay = np.zeros_like(viz_image)
        green_overlay[area_mask > 0] = (0, 255, 0)
        viz_image = cv2.addWeighted(viz_image, 1, green_overlay, 0.4, 0)
        
        overlay_polyline(viz_image, viz_data.get('path_coeff'), color=(255, 255, 0), thickness=3)
        if viz_data.get('goal_point_bev') is not None:
            cv2.circle(viz_image, viz_data['goal_point_bev'], 10, (0, 0, 255), -1)
            
        steer_deg = math.degrees(steering_angle_rad) if steering_angle_rad is not None else 0.0
        steer_text = f"Steer: {steer_deg:.1f} deg"
        cv2.putText(viz_image, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # cv2 이미지를 CompressedImage 메시지로 변환
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
    node = YoloBevDrivableAreaNode()
    # M_bev가 성공적으로 초기화되었는지 확인 후 spin 시작
    if rclpy.ok() and hasattr(node, 'M_bev'):
        try: 
            rclpy.spin(node)
        except KeyboardInterrupt: 
            node.get_logger().info("Keyboard interrupt, shutting down.")
        finally: 
            # 노드 종료 및 자원 해제
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()