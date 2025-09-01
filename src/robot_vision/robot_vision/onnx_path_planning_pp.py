#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: yolotl_path_planning_pp_optimized.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# 1. [Hinton's Fix] 비동기 처리를 위해 ThreadPoolExecutor를 도입하여 ROS 콜백과 YOLO 추론 연산을 분리.
# 2. [Hinton's Fix] 콜백 함수를 경량화하여 DDS 통신 지연 및 메시지 드랍을 원천 차단.
# 3. [Hinton's Fix] 안전한 노드 종료 로직을 추가하여 자원 누수 방지.
# 4. [Hinton's ONNX Fix] PyTorch(.pt) 가중치를 ONNX(.onnx)로 변경하여 추론 가속.

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

def filter_drivable_mask(bev_mask):
    f1 = morph_close(bev_mask, ksize=7)
    f2 = remove_small_components(f1, min_size=15000)
    return f2

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
    def __init__(self):
        super().__init__('yolo_bev_drivable_area_node')
        self.get_logger().info("--- YOLO BEV Drivable Area Planning Node (Hinton's Optimized ONNX Architecture) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"PyTorch detected device: {self.device}. ONNX Runtime will use the best available provider (e.g., CUDA, CPU).")
        
        # # QoS 프로파일 정의(주석처리)
        # self.qos_profile_sensor_data = QoSProfile(
        #     reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        # self.qos_profile_actuator_command = QoSProfile(
        #     reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        
        # 파라미터 선언
        # [Hinton's ONNX Fix] .pt 파일 대신 .onnx 파일을 기본값으로 설정
        self.declare_parameter('yolo_model_path', './YOLOTL.onnx')
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('wheelbase', 0.5161)
        self.declare_parameter('smoothing_alpha', 0.6)
        self.declare_parameter('lookahead_distance', 0.7)

        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.SMOOTHING_ALPHA = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value

        try:
            # [Hinton's ONNX Fix] .to(self.device)를 제거하여 ONNX 모델을 로드
            self.model = YOLO(yolo_model_path, task='segment')
            self.bev_params = np.load(bev_param_file)
            self.bev_h = int(self.bev_params['warp_h'])
            self.bev_w = int(self.bev_params['warp_w'])
            self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.0025, 1.25, 0.003578125
            self.get_logger().info(f"✅ ONNX model and resources loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load resources: {e}")
            self.destroy_node(); return

        self.tracked_center_path_coeff = None
        
        # [힌튼 수정] 비동기 처리를 위한 스레드 풀 및 종료 플래그 추가
        self.planning_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='planning_worker')
        self._is_shutting_down = False
        
        # 퍼블리셔/섭스크라이버(qos 프로파일 주석처리)
        # self.steer_pub = self.create_publisher(Float64, '/steering_angle', qos_profile=self.qos_profile_actuator_command)
        self.steer_pub = self.create_publisher(Float64, '/steering_angle', 10)
        # self.viz_pub = self.create_publisher(CompressedImage, '/path_planning/drivable_area/viz/compressed', qos_profile=self.qos_profile_sensor_data)
        self.viz_pub = self.create_publisher(CompressedImage, '/path_planning/drivable_area/viz/compressed', 10)
        # self.status_pub = self.create_publisher(Bool, '/path_planning/drivable_area/status', qos_profile=self.qos_profile_sensor_data)
        self.status_pub = self.create_publisher(Bool, '/path_planning/drivable_area/status', 10)
        realsense_img_topic = '/camera/color/image_raw/compressed'
        self.img_sub = self.create_subscription(CompressedImage, realsense_img_topic, self.planning_callback, qos_profile=self.qos_profile_sensor_data)
        self.get_logger().info(f"✅ Node initialized. Subscribing to {realsense_img_topic}")

    # [힌튼 수정] 콜백은 작업을 제출하고 즉시 리턴
    def planning_callback(self, compressed_img_msg):
        if self._is_shutting_down:
            return
        try:
            # 스레드 풀에 데이터 처리 작업 제출
            self.planning_executor.submit(self._process_planning_data, compressed_img_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to submit planning task: {e}")
            
    # [힌튼 추가] 실제 모든 연산을 수행하는 워커 함수
    def _process_planning_data(self, compressed_img_msg):
        try:
            np_arr = np.frombuffer(compressed_img_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            bev_image = self.do_bev_transform(cv_color)
            
            # YOLO 추론 (Heavy Task)
            results = self.model(bev_image, conf=self.yolo_confidence, verbose=False)
            result = results[0]

            combined_mask = np.zeros(result.orig_shape, dtype=np.uint8)
            if result.masks is not None:
                for mask_tensor in result.masks.data:
                    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                    combined_mask = np.maximum(combined_mask, mask_np)
            
            filtered_mask = filter_drivable_mask(combined_mask)
            
            steering_angle_rad, viz_data = self.calculate_steering_from_area(filtered_mask)
            
            # 결과 퍼블리시 (ROS 퍼블리셔는 스레드에 안전함)
            steer_msg = Float64()
            steer_msg.data = steering_angle_rad if steering_angle_rad is not None else 0.0
            self.steer_pub.publish(steer_msg)

            self.publish_visualization(bev_image, filtered_mask, viz_data, steering_angle_rad)

        except Exception as e:
            error_msg = f"Error in planning worker: {e}\n{traceback.format_exc()}"
            self.get_logger().error(error_msg)

    # --- 나머지 멤버 함수들은 변경 없음 ---
    def do_bev_transform(self, image):
        M = cv2.getPerspectiveTransform(self.bev_params['src_points'], self.bev_params['dst_points'])
        return cv2.warpPerspective(image, M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)

    def image_to_vehicle(self, pt_bev):
        u, v = pt_bev; x_vehicle = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        y_vehicle = (self.bev_w / 2 - u) * self.m_per_pixel_x; return x_vehicle, y_vehicle

    def calculate_steering_from_area(self, area_mask):
        is_detected = bool(np.any(area_mask)); self.status_pub.publish(Bool(data=is_detected))
        if not is_detected: self.tracked_center_path_coeff = None; return None, {}
        center_points_x, center_points_y = [], []
        for y in range(self.bev_h - 1, self.bev_h // 2, -5):
            drivable_pixels_x = np.where(area_mask[y, :] > 0)[0]
            if len(drivable_pixels_x) > 0: center_points_x.append(np.mean(drivable_pixels_x)); center_points_y.append(y)
        current_path_coeff = polyfit_path(center_points_y, center_points_x)
        if current_path_coeff is not None:
            if self.tracked_center_path_coeff is None: self.tracked_center_path_coeff = current_path_coeff
            else: self.tracked_center_path_coeff = (self.SMOOTHING_ALPHA * current_path_coeff + (1 - self.SMOOTHING_ALPHA) * self.tracked_center_path_coeff)
        final_path_coeff = self.tracked_center_path_coeff
        if final_path_coeff is None: return None, {}
        goal_point_vehicle, goal_point_bev = None, None
        for y_bev in range(self.bev_h - 1, -1, -1):
            x_bev = np.polyval(final_path_coeff, y_bev)
            if not (0 <= x_bev < self.bev_w): continue
            x_veh, y_veh = self.image_to_vehicle((x_bev, y_bev)); dist = math.sqrt(x_veh**2 + y_veh**2)
            if dist >= self.lookahead_distance: goal_point_vehicle = (x_veh, y_veh); goal_point_bev = (int(x_bev), int(y_bev)); break
        steering_angle = None
        if goal_point_vehicle is not None: x_goal, y_goal = goal_point_vehicle; alpha = math.atan2(y_goal, x_goal); steering_angle = math.atan2(2.0 * self.L * math.sin(alpha), self.lookahead_distance)
        viz_data = {'path_coeff': final_path_coeff, 'goal_point_bev': goal_point_bev}; return steering_angle, viz_data
    
    def publish_visualization(self, bev_image, area_mask, viz_data, steering_angle_rad):
        viz_image = bev_image.copy(); green_overlay = np.zeros_like(viz_image); green_overlay[area_mask > 0] = (0, 255, 0)
        viz_image = cv2.addWeighted(viz_image, 1, green_overlay, 0.4, 0)
        overlay_polyline(viz_image, viz_data.get('path_coeff'), color=(255, 255, 0), thickness=3)
        if viz_data.get('goal_point_bev') is not None: cv2.circle(viz_image, viz_data['goal_point_bev'], 10, (0, 0, 255), -1)
        steer_deg = math.degrees(steering_angle_rad) if steering_angle_rad is not None else 0.0
        steer_text = f"Steer: {steer_deg:.1f} deg"; cv2.putText(viz_image, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        viz_msg = self.bridge.cv2_to_compressed_imgmsg(viz_image); self.viz_pub.publish(viz_msg)

    # [힌튼 추가] 안전한 종료를 위한 메서드
    def destroy_node(self):
        self.get_logger().info("Shutting down the planning thread pool.")
        self._is_shutting_down = True
        self.planning_executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloBevDrivableAreaNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()