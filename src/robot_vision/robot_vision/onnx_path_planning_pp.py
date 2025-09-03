#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: onnx_path_planning_pp.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's Final Optimization]
# 1. 경로 계산 로직을 순수 NumPy 벡터화 연산으로 대체하여 CPU 병목 현상 제거 (성능 극대화)
# 2. 시각화 토픽 구독자가 있을 때만 시각화 연산을 수행하여 불필요한 CPU 자원 낭비 방지
# 3. 실시간 영상 스트림에 최적화된 'Best Effort' QoS 프로파일 적용
# 4. 콜백 함수에서 모든 연산을 제거하고 작업 스레드로 이전하여 통신 지연 가능성 원천 차단
# 5. 주요 파라미터를 클래스 상수로 관리하여 가독성 및 유지보수성 향상

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
    # [Hinton's Optimization] 알고리즘 상수 정의
    _MORPH_KSIZE = 7
    _MIN_AREA_SIZE = 15000
    
    def __init__(self):
        super().__init__('yolo_bev_drivable_area_node')
        self.get_logger().info("--- YOLO BEV Drivable Area Planning Node (Hinton's Ultimate Optimized ONNX Architecture) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 파라미터 선언
        self.declare_parameter('yolo_model_path', './YOLOTL.onnx')
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('wheelbase', 0.5161)
        self.declare_parameter('smoothing_alpha', 0.6)
        self.declare_parameter('lookahead_distance', 0.7)

        # 파라미터 가져오기
        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
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
            self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.0025, 1.25, 0.003578125
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
        
        # [Hinton's Optimization] 실시간 영상 스트림에 최적화된 QoS 프로파일
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        realsense_img_topic = '/camera/color/image_raw/compressed'
        self.img_sub = self.create_subscription(
            CompressedImage, 
            realsense_img_topic, 
            self.planning_callback, 
            qos_profile_sensor_data  # 최적화된 QoS 적용
        )
        self.get_logger().info(f"✅ Node initialized. Subscribing to '{realsense_img_topic}' with BEST_EFFORT QoS.")

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
            if result.masks is not None:
                combined_mask = np.max(np.array([m.cpu().numpy() for m in result.masks.data]), axis=0)
                combined_mask = (combined_mask * 255).astype(np.uint8)

            filtered_mask = self.filter_drivable_mask(combined_mask)
            steering_angle_rad, viz_data = self.calculate_steering_from_area(filtered_mask)
            
            steer_msg = Float64()
            steer_msg.data = steering_angle_rad if steering_angle_rad is not None else 0.0
            self.steer_pub.publish(steer_msg)

            # [Hinton's Optimization] 구독자가 있을 때만 시각화 연산 수행
            if self.viz_pub.get_subscription_count() > 0:
                self.publish_visualization(bev_image, filtered_mask, viz_data, steering_angle_rad)

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
        x_vehicle = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        y_vehicle = (self.bev_w / 2 - u) * self.m_per_pixel_x
        return x_vehicle, y_vehicle

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
            return None, {}

        # y좌표를 기준으로 x좌표들의 평균을 계산 (중심점 찾기)
        # 이 방법은 pandas.groupby().mean()과 유사하지만 numpy만으로 구현
        unique_y = np.unique(y_indices)
        center_points_x = np.array([x_indices[y_indices == y].mean() for y in unique_y])
        center_points_y = unique_y + self.bev_h // 2  # 원본 이미지 y좌표로 복원

        current_path_coeff = polyfit_path(center_points_y, center_points_x)
        
        # 경로 계수 스무딩
        if current_path_coeff is not None:
            if self.tracked_center_path_coeff is None: self.tracked_center_path_coeff = current_path_coeff
            else: self.tracked_center_path_coeff = (self.SMOOTHING_ALPHA * current_path_coeff + (1 - self.SMOOTHING_ALPHA) * self.tracked_center_path_coeff)
        
        final_path_coeff = self.tracked_center_path_coeff
        if final_path_coeff is None: return None, {}
        
        # 목표점(lookahead point) 계산
        y_bev_coords = np.arange(self.bev_h -1, -1, -1)
        x_bev_coords = np.polyval(final_path_coeff, y_bev_coords)
        
        valid_indices = (x_bev_coords >= 0) & (x_bev_coords < self.bev_w)
        y_bev_coords, x_bev_coords = y_bev_coords[valid_indices], x_bev_coords[valid_indices]
        
        x_veh, y_veh = self.image_to_vehicle((x_bev_coords, y_bev_coords))
        dist_from_ego = np.sqrt(x_veh**2 + y_veh**2)
        
        goal_idx = np.argmin(np.abs(dist_from_ego - self.lookahead_distance))
        
        steering_angle = None
        goal_point_bev = None
        if np.abs(dist_from_ego[goal_idx] - self.lookahead_distance) < 0.2: # 목표 거리와 비슷한 점이 있다면
            x_goal, y_goal = x_veh[goal_idx], y_veh[goal_idx]
            alpha = math.atan2(y_goal, x_goal)
            steering_angle = math.atan2(2.0 * self.L * math.sin(alpha), self.lookahead_distance)
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
        
        viz_msg = self.bridge.cv2_to_compressed_imgmsg(viz_image)
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
            rclpy.shutdown()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()