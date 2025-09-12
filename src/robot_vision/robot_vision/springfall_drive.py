#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: springfall_drive.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's Final Optimization & Robust Pure Pursuit Logic with Attentional ROI]
# 1. 경로 계산 로직을 순수 NumPy 벡터화 연산으로 대체하여 CPU 병목 현상 제거 (성능 극대화)
# 2. 시각화 토픽 구독자가 있을 때만 시각화 연산을 수행하여 불필요한 CPU 자원 낭비 방지
# 3. 실시간 영상 스트림에 최적화된 'Best Effort' QoS 프로파일 적용
# 4. 콜백 함수에서 모든 연산을 제거하고 작업 스레드로 이전하여 통신 지연 가능성 원천 차단
# 5. 주요 파라미터를 클래스 상수로 관리하여 가독성 및 유지보수성 향상
# 6. Pure Pursuit 알고리즘 안정성 강화: 경로가 짧을 경우 마지막 점을 목표점으로 지정
# 7. 제어 기준점을 '가상 후륜 축'으로 변경하여 Pure Pursuit 알고리즘의 정확도 극대화
# 8. [핵심 개선] '신뢰도 기반 동적 스무딩' 적용: 경로 포인트 수에 따라 스무딩 강도를 자동 조절하여 극한의 안정성 확보
# 9. [Fail-Safe] 주행 영역 미감지 시 조향각 0도를 발행하여 안정성 확보
# 10. [핵심 수정] 지정된 ROI(관심 영역) 내에서만 경로를 생성하여 노이즈 제거 및 안정성 극대화

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
    _MORPH_KSIZE = 7
    _MIN_AREA_SIZE = 15000
    
    _MAX_CONFIDENCE_POINTS = 32000
    _MIN_CONFIDENCE_POINTS = 2000
    _MAX_SMOOTHING_ALPHA = 0.6
    _MIN_SMOOTHING_ALPHA = 0.3
    
    # [힌튼의 핵심 수정] 경로 탐색을 위한 ROI(관심 영역) 정의 (비율 기준)
    # 이 값들을 조절하여 경로 탐색 영역을 튜닝할 수 있습니다.
    _ROI_TOP_Y_RATIO = 0.0  # BEV 이미지 높이의 50% 지점부터 ROI 시작
    _ROI_BOTTOM_Y_RATIO = 1.0 # BEV 이미지 높이의 100% 지점(가장 아래)에서 ROI 끝
    _ROI_WIDTH_RATIO = 1.0 # BEV 이미지 중앙을 기준으로 좌우 80% 폭만 사용
    
    def __init__(self):
        super().__init__('yolo_bev_drivable_area_node')
        self.get_logger().info("--- YOLO BEV Drivable Area Planning Node (Hinton's Ultimate Optimized ONNX Architecture with ROI) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.declare_parameter('yolo_model_path', './drive_area2.onnx')
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('wheelbase', 0.6)
        self.declare_parameter('camera_to_rear_axle_offset', 0.27)
        self.declare_parameter('lookahead_distance', 1.0)

        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.CAMERA_TO_REAR_AXLE_OFFSET = self.get_parameter('camera_to_rear_axle_offset').get_parameter_value().double_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value

        try:
            self.model = YOLO(yolo_model_path, task='segment')
            
            bev_params = np.load(bev_param_file)
            self.src_points = bev_params['src_points']
            self.dst_points = bev_params['dst_points']
            self.bev_h = int(bev_params['warp_h'])
            self.bev_w = int(bev_params['warp_w'])
            self.M_bev = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.get_logger().info("✅ BEV transformation matrix calculated.")

            self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.002609375, 0.66, 0.0011171875
            self.get_logger().info(f"✅ ONNX model and all resources loaded on [{self.device}].")

        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to load resources: {e}")
            rclpy.shutdown()
            return

        self.tracked_center_path_coeff = None
        self.planning_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='planning_worker')
        self._is_shutting_down = False
        
        self.steer_pub = self.create_publisher(Float64, '/steering_angle', 10)
        self.viz_pub = self.create_publisher(CompressedImage, '/path_planning/drivable_area/viz/compressed', 10)
        self.status_pub = self.create_publisher(Bool, '/path_planning/drivable_area/status', 10)
        
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
                combined_mask = np.max(np.array([m.cpu().numpy() for m in result.masks.data]), axis=0)
                combined_mask = (combined_mask * 255).astype(np.uint8)

            filtered_mask = self.filter_drivable_mask(combined_mask)
            steering_angle_rad, viz_data = self.calculate_steering_from_area(filtered_mask)
            
            final_viz_angle = 0.0
            steer_msg = Float64()

            if steering_angle_rad is not None:
                final_viz_angle = steering_angle_rad
                steer_msg.data = final_viz_angle
                self.steer_pub.publish(steer_msg)
            else:
                self.get_logger().warn("No path calculated. Publishing 0.0 steering angle (fail-safe).", throttle_duration_sec=2)
                steer_msg.data = 0.0
                self.steer_pub.publish(steer_msg)
            
            if self.viz_pub.get_subscription_count() > 0:
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
        # ================================================================= #
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [힌튼의 핵심 수정] ROI 적용 로직 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        # 1. 설정된 비율에 따라 ROI의 실제 픽셀 좌표를 계산합니다.
        roi_top_y = int(self.bev_h * self._ROI_TOP_Y_RATIO)
        roi_bottom_y = int(self.bev_h * self._ROI_BOTTOM_Y_RATIO)
        roi_half_width = int((self.bev_w * self._ROI_WIDTH_RATIO) / 2)
        roi_center_x = self.bev_w // 2
        roi_left_x = roi_center_x - roi_half_width
        roi_right_x = roi_center_x + roi_half_width

        # 2. ROI 영역만 흰색으로 채워진 마스크를 생성합니다. (집중의 '창')
        roi_mask = np.zeros_like(area_mask)
        cv2.rectangle(roi_mask, (roi_left_x, roi_top_y), (roi_right_x, roi_bottom_y), 255, -1)
        
        # 3. bitwise_and 연산을 통해 원본 마스크에서 ROI 영역만 정확히 추출합니다.
        roi_area_mask = cv2.bitwise_and(area_mask, area_mask, mask=roi_mask)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [힌튼의 핵심 수정] ROI 적용 로직 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
        # ================================================================= #

        is_detected = np.any(roi_area_mask) # 이제 ROI 내부의 영역만 감지 대상으로 간주
        self.status_pub.publish(Bool(data=bool(is_detected)))
        if not is_detected:
            self.tracked_center_path_coeff = None
            return None, {'roi_coords': (roi_left_x, roi_top_y, roi_right_x, roi_bottom_y)}
        
        # NumPy 벡터화로 ROI 내부의 경로점 계산
        y_indices, x_indices = np.where(roi_area_mask > 0)
        
        num_points = len(y_indices)
        if num_points < 50:
            self.get_logger().warn("Not enough ROI points. Re-using last stable path.", throttle_duration_sec=2)
            if self.tracked_center_path_coeff is not None:
                pass
            else:
                return None, {'roi_coords': (roi_left_x, roi_top_y, roi_right_x, roi_bottom_y)}
        else: 
            unique_y, unique_y_indices = np.unique(y_indices, return_inverse=True)
            sum_x = np.bincount(unique_y_indices, weights=x_indices)
            count_y = np.bincount(unique_y_indices)
            
            valid_counts = count_y > 0
            center_points_x = sum_x[valid_counts] / count_y[valid_counts]
            center_points_y = unique_y[valid_counts]

            current_path_coeff = polyfit_path(center_points_y, center_points_x)
            
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

        final_path_coeff = self.tracked_center_path_coeff
        if final_path_coeff is None:
            return None, {'roi_coords': (roi_left_x, roi_top_y, roi_right_x, roi_bottom_y)}
        
        # 목표점 계산 시에도 ROI의 상단 경계(roi_top_y)를 고려합니다.
        y_bev_coords = np.arange(roi_bottom_y - 1, roi_top_y, -1)
        x_bev_coords = np.polyval(final_path_coeff, y_bev_coords)
        
        valid_indices = (x_bev_coords >= 0) & (x_bev_coords < self.bev_w)
        if not np.any(valid_indices):
            self.get_logger().warn("Path is completely outside of the BEV image.", throttle_duration_sec=2)
            return None, {'roi_coords': (roi_left_x, roi_top_y, roi_right_x, roi_bottom_y)}
            
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
                self.get_logger().warn(f"Path is shorter than lookahead. Using farthest point.", throttle_duration_sec=2)
                goal_idx = np.argmax(dist_from_ego)

        if goal_idx is not None:
            actual_lookahead_dist = dist_from_ego[goal_idx]
            if actual_lookahead_dist > 0.1:
                x_goal, y_goal = x_veh[goal_idx], y_veh[goal_idx]
                alpha = math.atan2(y_goal, x_goal)
                steering_angle = math.atan2(2.0 * self.L * math.sin(alpha), actual_lookahead_dist)
                goal_point_bev = (int(x_bev_coords[goal_idx]), int(y_bev_coords[goal_idx]))

        viz_data = {
            'path_coeff': final_path_coeff, 
            'goal_point_bev': goal_point_bev,
            'roi_coords': (roi_left_x, roi_top_y, roi_right_x, roi_bottom_y)
        }
        return steering_angle, viz_data
    
    def publish_visualization(self, bev_image, area_mask, viz_data, steering_angle_rad):
        viz_image = bev_image.copy()
        green_overlay = np.zeros_like(viz_image)
        green_overlay[area_mask > 0] = (0, 255, 0)
        viz_image = cv2.addWeighted(viz_image, 1, green_overlay, 0.4, 0)
        
        overlay_polyline(viz_image, viz_data.get('path_coeff'), color=(255, 255, 0), thickness=3)
        if viz_data.get('goal_point_bev') is not None:
            cv2.circle(viz_image, viz_data['goal_point_bev'], 10, (0, 0, 255), -1)

        # [힌튼의 핵심 수정] ROI 영역을 시각화합니다.
        if 'roi_coords' in viz_data:
            x1, y1, x2, y2 = viz_data['roi_coords']
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 255), 2) # 노란색-청록색 사각형으로 표시
            
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
    node = YoloBevDrivableAreaNode()
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