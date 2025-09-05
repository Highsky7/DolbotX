#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: onnx_path_planning_physical_model.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's Ultimate Architecture with Physical Model]
# 1. [핵심 아키텍처 변경] 3D 물리 모델 도입: 2D 픽셀 공간의 한계를 극복하기 위해,
#    카메라의 높이, 각도 등 물리적 설치 정보를 이용해 이미지 좌표를 실제 차량 좌표계(미터 단위)로 변환합니다.
# 2. [핵심 함수 추가] 'pixel_to_vehicle_coords': 역투영 변환(IPM)을 수행하여 픽셀을 물리적 좌표로 매핑합니다.
#    - 이 과정에서 카메라 뒤 0.375m에 위치한 실제 회전 중심을 정확히 반영합니다.
# 3. [핵심 알고리즘 변경] Pure Pursuit 공식 적용: 변환된 물리 좌표를 사용하여 기구학적으로 의미 있는
#    'Pure Pursuit' 조향각 공식을 통해 조향각을 계산, 정확도를 극대화합니다.
# 4. [파라미터 재설계] 'camera_height_m', 'camera_pitch_deg', 'wheelbase_m', 'lookahead_distance_m' 등
#    물리적 의미를 갖는 파라미터들을 사용하며, 이는 반드시 실제 환경에 맞게 교정(Calibration)되어야 합니다.

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import traceback

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from scipy.special import binom

# --- 유틸리티 클래스 (BezierPathPlanner는 이전과 동일) ---
class BezierPathPlanner:
    # (이전 답변과 동일한 내용이므로 생략)
    def __init__(self, lane_polygon, car_position, centroid, initial_lookahead):
        self.lane_polygon = lane_polygon
        self.car_position = car_position
        self.centroid = centroid
        self.lookahead_distance = initial_lookahead
    def _find_nearest_value(self, arr, value):
        idx = np.argmin(np.abs(arr - value))
        return arr[idx]
    def generate_control_points(self):
        trans_polygon = self.lane_polygon.copy()
        dest_x = self._find_nearest_value(trans_polygon[:, 0], self.centroid[0])
        target_y = trans_polygon[trans_polygon[:, 0] == dest_x][0][1]
        dist = self.centroid[1] - target_y
        if dist > 0: trans_polygon[:, 1] += int(dist)
        else: trans_polygon[:, 1] -= int(dist)
        if 0 <= np.abs(dist) <= 100: self.lookahead_distance += (100 - np.abs(dist)) * 0.5
        sort_index = np.argsort(trans_polygon[:, 1])
        y_max = trans_polygon[sort_index[0]]
        if dist < 100:
            mid_control1 = (self.car_position[0] - (self.car_position[0] - self.centroid[0]) / 3, self.car_position[1] - ((self.car_position[1] - self.centroid[1]) * 5 / 10))
            mid_control2 = (self.car_position[0] - (self.car_position[0] - self.centroid[0]) * 2 / 3, self.car_position[1] - ((self.car_position[1] - self.centroid[1]) * 8 / 10))
            mid_control3 = (self.centroid[0] - (self.centroid[0] - y_max[0]) / 3, self.centroid[1] - ((self.centroid[1] - y_max[1]) * 5 / 10))
            mid_control4 = (self.centroid[0] - (self.centroid[0] - y_max[0]) * 2 / 3, self.centroid[1] - ((self.centroid[1] - y_max[1]) * 8 / 10))
            control_points = (self.car_position, mid_control1, mid_control2, self.centroid, mid_control3, mid_control4, y_max)
        else:
            mid_control1 = (self.car_position[0] - (self.car_position[0] - self.centroid[0]) / 3, self.car_position[1] - ((self.car_position[1] - self.centroid[1]) * 5 / 10))
            mid_control2 = (self.car_position[0] - (self.car_position[0] - self.centroid[0]) * 2 / 3, self.car_position[1] - ((self.car_position[1] - self.centroid[1]) * 8 / 10))
            control_points = (self.car_position, mid_control1, mid_control2, self.centroid)
        return control_points, self.lookahead_distance
    def compute_bezier_curve(self, bezier_points, num_points=100):
        n = len(bezier_points) - 1
        t_values = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
        for i in range(n + 1):
            bernstein_poly = binom(n, i) * (t_values ** i) * ((1 - t_values) ** (n - i))
            curve += np.outer(bernstein_poly, bezier_points[i])
        return curve
    def find_lookahead_point(self, curve, current_pos, lookahead_distance):
        distances = np.linalg.norm(curve - current_pos, axis=1)
        idx = np.argmin(np.abs(distances - lookahead_distance))
        return curve[idx]

class YoloPhysicalModelNode(Node):
    def __init__(self):
        super().__init__('yolo_physical_model_node')
        self.get_logger().info("--- YOLO Physical Model Node (Hinton's Ultimate Architecture) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # YOLO 및 경로 생성 파라미터
        self.declare_parameter('yolo_model_path', './drive_area.onnx')
        self.declare_parameter('yolo_confidence', 0.5)

        # [Hinton's NEW] 물리 모델 파라미터 (❗❗❗반드시 측정 및 교정 필요❗❗❗)
        # 1. 로봇 자체 파라미터
        self.declare_parameter('wheelbase_m', 0.5) # 로봇 축거 (m)
        self.declare_parameter('camera_to_pivot_offset_m', 0.375) # 카메라-회전중심 거리 (m)
        # 2. 카메라 설치 파라미터 (Extrinsics)
        self.declare_parameter('camera_height_m', 0.39) # 지면으로부터 카메라 높이 (m)
        self.declare_parameter('camera_pitch_deg', 15.0) # 카메라가 아래를 보는 각도 (degree)
        # 3. 카메라 내부 파라미터 (Intrinsics) - Logitech C922 기준 근사치
        self.declare_parameter('camera_focal_length_px', 615.0) # 카메라 초점거리 (pixels)
        
        # [Hinton's NEW] 제어 파라미터
        self.declare_parameter('lookahead_distance_m', 0.6) # 전방 주시 거리 (m)
        self.declare_parameter('angle_scaling_factor', 1.0)
        self.declare_parameter('steering_angle_topic', '/steering_angle')

        # 파라미터 로드
        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
        
        # 물리 파라미터
        self.WHEELBASE = self.get_parameter('wheelbase_m').get_parameter_value().double_value
        self.CAMERA_TO_PIVOT_OFFSET = self.get_parameter('camera_to_pivot_offset_m').get_parameter_value().double_value
        self.CAMERA_HEIGHT = self.get_parameter('camera_height_m').get_parameter_value().double_value
        self.CAMERA_PITCH_RAD = math.radians(self.get_parameter('camera_pitch_deg').get_parameter_value().double_value)
        self.FOCAL_LENGTH = self.get_parameter('camera_focal_length_px').get_parameter_value().double_value
        
        # 제어 파라미터
        self.LOOKAHEAD_DISTANCE = self.get_parameter('lookahead_distance_m').get_parameter_value().double_value
        self.ANGLE_SCALING_FACTOR = self.get_parameter('angle_scaling_factor').get_parameter_value().double_value
        steering_angle_topic = self.get_parameter('steering_angle_topic').get_parameter_value().string_value

        try: self.model = YOLO(yolo_model_path, task='segment')
        except Exception as e: self.get_logger().error(f"FATAL: Failed to load resources: {e}"); rclpy.shutdown(); return

        self.planning_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='planning_worker')
        self._is_shutting_down = False
        
        self.steer_pub = self.create_publisher(Float64, steering_angle_topic, 10)
        self.viz_pub = self.create_publisher(CompressedImage, '/path_planning/drivable_area/viz/compressed', 10)
        self.status_pub = self.create_publisher(Bool, '/path_planning/drivable_area/status', 10)
        
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.img_sub = self.create_subscription(CompressedImage, '/camera3/image_raw/compressed', self.planning_callback, qos)
        self.get_logger().info("✅ Node initialized with Physical Model.")

    def planning_callback(self, msg):
        if self._is_shutting_down: return
        self.planning_executor.submit(self._process_planning_data, msg.data)
            
    def _process_planning_data(self, compressed_data_buffer):
        try:
            np_arr = np.frombuffer(compressed_data_buffer, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv_color=1)
            if cv_color is None: return

            results = self.model(cv_color, conf=self.yolo_confidence, verbose=False)
            steering_angle_rad, viz_data, combined_mask = self.calculate_steering_with_physical_model(results[0])
            
            steer_msg = Float64()
            steer_msg.data = self.ANGLE_SCALING_FACTOR * steering_angle_rad if steering_angle_rad is not None else 0.0
            self.steer_pub.publish(steer_msg)

            if self.viz_pub.get_subscription_count() > 0:
                self.publish_visualization(cv_color, combined_mask, viz_data, steer_msg.data)
        except Exception:
            self.get_logger().error(f"Error in planning worker:\n{traceback.format_exc()}")
    
    def pixel_to_vehicle_coords(self, u, v, img_h, img_w):
        """[Hinton's NEW CORE] 픽셀(u,v)을 실제 회전 중심 기준 차량 좌표(m)로 변환"""
        # 1. 이미지 중심을 원점으로 하는 정규화된 픽셀 좌표
        u_norm = u - img_w / 2
        v_norm = img_h / 2 - v # Y축 방향이 반대이므로

        # 2. 카메라 각도를 고려하여 지면의 점 계산
        # Zc(광축 방향 거리)는 v_norm과 초점거리, 카메라 각도로부터 유도됨
        alpha = self.CAMERA_PITCH_RAD + math.atan(v_norm / self.FOCAL_LENGTH)
        # 지면까지의 거리(카메라 기준 x_cam)
        x_cam = self.CAMERA_HEIGHT * math.tan(alpha)
        # 측면 거리(카메라 기준 y_cam)
        y_cam = x_cam * u_norm / self.FOCAL_LENGTH
        
        # 3. [핵심] 카메라 기준 좌표를 실제 회전 중심(Pivot) 기준으로 변환
        # x축은 옵셋만큼 뒤로 이동. y축은 동일.
        x_pivot = x_cam - self.CAMERA_TO_PIVOT_OFFSET
        y_pivot = y_cam
        
        return x_pivot, y_pivot

    def calculate_steering_with_physical_model(self, yolo_result):
        lane_polygon, combined_mask = None, np.zeros(yolo_result.orig_shape, dtype=np.uint8)
        img_h, img_w = yolo_result.orig_shape

        # ... (이전과 동일한 YOLO 결과 파싱 로직) ...
        if yolo_result.masks is not None:
            if len(yolo_result.masks.data) > 0:
                raster_mask = np.max(np.array([m.cpu().numpy() for m in yolo_result.masks.data]), axis=0)
                combined_mask = (raster_mask * 255).astype(np.uint8)
            if yolo_result.boxes is not None:
                classes = yolo_result.boxes.cls.cpu().numpy()
                masks_xy = yolo_result.masks.xy
                for cls_id, mask_xy in zip(classes, masks_xy):
                    if int(cls_id) == 0: lane_polygon = mask_xy; break
        
        is_detected = lane_polygon is not None
        self.status_pub.publish(Bool(data=bool(is_detected)))
        if not is_detected: return None, {}, combined_mask

        # 1. 이미지 상의 경로 생성 (이전과 동일)
        M = cv2.moments(lane_polygon)
        if M["m00"] == 0: return None, {}, combined_mask
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # 가상 후륜축(카메라 바로 아래 지점)을 경로 생성의 시작점으로 사용
        cam_bottom_pixel = np.array([img_w / 2, img_h]) 
        
        # 픽셀 공간에서 목표점 찾기 (물리적 거리와 가장 유사한 픽셀 거리를 찾기 위함)
        # lookahead_pixel_approx = self.LOOKAHEAD_DISTANCE / (self.CAMERA_HEIGHT / self.FOCAL_LENGTH) # 매우 거친 근사치
        lookahead_pixel_approx = 100 # 고정값으로 시작하는 것이 안정적일 수 있음

        planner = BezierPathPlanner(lane_polygon, cam_bottom_pixel, centroid, lookahead_pixel_approx)
        control_points, _ = planner.generate_control_points()
        bezier_path = planner.compute_bezier_curve(control_points)
        
        # 2. [핵심] 물리적 거리 기반의 목표점 재탐색
        x_veh, y_veh = self.pixel_to_vehicle_coords(bezier_path[:,0], bezier_path[:,1], img_h, img_w)
        dist_from_pivot = np.sqrt(x_veh**2 + y_veh**2)
        
        # 목표 거리(LOOKAHEAD_DISTANCE)와 가장 가까운 경로상의 점을 최종 목표점으로 선택
        goal_idx = np.argmin(np.abs(dist_from_pivot - self.LOOKAHEAD_DISTANCE))
        
        x_goal, y_goal = x_veh[goal_idx], y_veh[goal_idx]
        lookahead_point_pixel = bezier_path[goal_idx]

        # 3. [핵심] Pure Pursuit 조향각 공식 적용
        # alpha는 차량의 현재 방향(x축)과 목표점 사이의 각도
        alpha = math.atan2(y_goal, x_goal)
        # 실제 목표점까지의 거리
        actual_lookahead = dist_from_pivot[goal_idx]
        
        # Pure Pursuit 공식: δ = atan(2 * L * sin(α) / ld)
        steering_angle_rad = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha), actual_lookahead)
        
        viz_data = {'bezier_path': bezier_path, 'lookahead_pixel': lookahead_point_pixel, 'goal_coords_m': (x_goal, y_goal)}
        return steering_angle_rad, viz_data, combined_mask

    def publish_visualization(self, raw_image, area_mask, viz_data, final_steering_angle_rad):
        viz_image = raw_image.copy()
        if np.any(area_mask):
            green_overlay = np.zeros_like(viz_image); green_overlay[area_mask > 0] = (0, 255, 0)
            viz_image = cv2.addWeighted(viz_image, 1, green_overlay, 0.4, 0)
        
        if 'bezier_path' in viz_data: cv2.polylines(viz_image, [viz_data['bezier_path'].astype(np.int32)], False, (0, 255, 255), 3)
        if 'lookahead_pixel' in viz_data:
            pt = viz_data['lookahead_pixel']
            cv2.circle(viz_image, (int(pt[0]), int(pt[1])), 12, (0, 0, 255), -1)
            
        steer_deg = math.degrees(final_steering_angle_rad)
        steer_text = f"Steer Angle: {steer_deg:.1f} deg"
        cv2.putText(viz_image, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if 'goal_coords_m' in viz_data:
            x_m, y_m = viz_data['goal_coords_m']
            coord_text = f"Target (m): x={x_m:.2f}, y={y_m:.2f}"
            cv2.putText(viz_image, coord_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        _, jpeg_buffer = cv2.imencode('.jpg', viz_image);
        viz_msg = CompressedImage(data=jpeg_buffer.tobytes(), format="jpeg")
        viz_msg.header.stamp = self.get_clock().now().to_msg(); self.viz_pub.publish(viz_msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down..."); self._is_shutting_down = True
        self.planning_executor.shutdown(wait=True); super().destroy_node()

def main(args=None):
    rclpy.init(args=args); node = YoloPhysicalModelNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info("Keyboard interrupt, shutting down.")
    finally: node.destroy_node(); rclpy.shutdown() if rclpy.ok() else None; cv2.destroyAllWindows()

if __name__ == '__main__':
    main()