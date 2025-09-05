#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: onnx_path_planning_bezier_angle.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's Final Architecture for Accurate Steering Angle Publication]
# 1. [핵심 목표 재설정] 제어 출력을 '/steering_angle' (Float64) 발행으로 복귀시켰습니다.
#    스키드 조향 로직을 제거하고, 순수 조향각 계산에 집중합니다.
# 2. [핵심 파라미터 추가] 조향각 보정 계수 'angle_scaling_factor'를 도입했습니다.
#    - 이 파라미터는 픽셀 공간에서 계산된 기하학적 각도를 실제 로봇의 조향 메커니즘에 맞게
#      스케일링하여, 실질적인 조향 명령으로 변환하는 핵심적인 역할을 합니다.
# 3. 파라미터 튜닝의 편의성을 위해 조향각 토픽 이름도 파라미터로 관리하도록 변경했습니다.
# 4. 시각화 정보를 조향각(degree)으로 다시 변경하여 직관적인 디버깅을 지원합니다.
# 5. 베지어 곡선 기반의 정교한 경로 생성 로직은 그대로 유지됩니다.

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# [Hinton's MOD] Float64 메시지를 다시 사용합니다.
from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import traceback

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from scipy.special import binom

# --- 유틸리티 클래스 및 함수 (이전과 동일) ---
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
        if 0 <= np.abs(dist) <= 100:
            self.lookahead_distance += (100 - np.abs(dist)) * 0.5
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

class YoloBezierAngleNode(Node):
    def __init__(self):
        super().__init__('yolo_bezier_angle_node')
        self.get_logger().info("--- YOLO Bezier Angle Node (Hinton's Final Angle Architecture) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 경로 생성 파라미터
        self.declare_parameter('yolo_model_path', './drive_area.onnx')
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('initial_lookahead_distance', 500.0)
        self.declare_parameter('car_position_pixel_u', 320.0)
        self.declare_parameter('car_position_pixel_v', 720.0)

        # [Hinton's NEW] 조향각 제어 파라미터
        self.declare_parameter('angle_scaling_factor', 1.0) # 조향각 보정 계수
        self.declare_parameter('steering_angle_topic', '/steering_angle')

        # 파라미터 가져오기
        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
        self.INITIAL_LOOKAHEAD_DISTANCE = self.get_parameter('initial_lookahead_distance').get_parameter_value().double_value
        car_pos_u = self.get_parameter('car_position_pixel_u').get_parameter_value().double_value
        car_pos_v = self.get_parameter('car_position_pixel_v').get_parameter_value().double_value
        self.CAR_POSITION_PIXEL = np.array([car_pos_u, car_pos_v])
        
        self.ANGLE_SCALING_FACTOR = self.get_parameter('angle_scaling_factor').get_parameter_value().double_value
        steering_angle_topic = self.get_parameter('steering_angle_topic').get_parameter_value().string_value
        
        try:
            self.model = YOLO(yolo_model_path, task='segment')
            self.get_logger().info(f"✅ ONNX model loaded on [{self.device}].")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to load resources: {e}")
            rclpy.shutdown(); return

        self.planning_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='planning_worker')
        self._is_shutting_down = False
        
        # [Hinton's MOD] Float64 메시지를 발행하는 퍼블리셔로 복귀
        self.steer_pub = self.create_publisher(Float64, steering_angle_topic, 10)
        self.viz_pub = self.create_publisher(CompressedImage, '/path_planning/drivable_area/viz/compressed', 10)
        self.status_pub = self.create_publisher(Bool, '/path_planning/drivable_area/status', 10)
        
        qos_profile_sensor_data = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        logitech_img_topic = '/camera3/image_raw/compressed'
        self.img_sub = self.create_subscription(CompressedImage, logitech_img_topic, self.planning_callback, qos_profile_sensor_data)
        self.get_logger().info(f"✅ Node initialized. Publishing Float64 to '{steering_angle_topic}'.")

    def planning_callback(self, compressed_img_msg):
        if self._is_shutting_down: return
        self.planning_executor.submit(self._process_planning_data, compressed_img_msg.data)
            
    def _process_planning_data(self, compressed_data_buffer):
        try:
            np_arr = np.frombuffer(compressed_data_buffer, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_color is None: return

            results = self.model(cv_color, conf=self.yolo_confidence, verbose=False)
            
            steering_angle_rad, viz_data, combined_mask = self.calculate_steering_with_bezier(results[0])
            
            # ================================================================= #
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분이 수정되었습니다 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
            # 시각화를 위한 최종 조향각 변수 (검출 실패 시 0.0)
            final_viz_angle = 0.0
            
            if steering_angle_rad is not None:
                # 검출에 성공했을 때만 조향각을 계산하고 발행합니다.
                final_viz_angle = self.ANGLE_SCALING_FACTOR * steering_angle_rad
                steer_msg = Float64()
                steer_msg.data = final_viz_angle
                self.steer_pub.publish(steer_msg)
            # else:
            #     # 주행 가능 영역이 검출되지 않으면 아무것도 발행하지 않습니다.
            #     pass

            if self.viz_pub.get_subscription_count() > 0:
                self.publish_visualization(cv_color, combined_mask, viz_data, final_viz_angle)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 이 부분이 수정되었습니다 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
            # ================================================================= #

        except Exception:
            self.get_logger().error(f"Error in planning worker:\n{traceback.format_exc()}")
    
    def calculate_steering_with_bezier(self, yolo_result):
        # (이전 답변의 calculate_path_error_with_bezier 와 로직 동일)
        lane_polygon, combined_mask = None, np.zeros(yolo_result.orig_shape, dtype=np.uint8)
        if yolo_result.masks is not None:
            if len(yolo_result.masks.data) > 0:
                raster_mask = np.max(np.array([m.cpu().numpy() for m in yolo_result.masks.data]), axis=0)
                combined_mask = (raster_mask * 255).astype(np.uint8)
            if yolo_result.boxes is not None:
                classes = yolo_result.boxes.cls.cpu().numpy()
                masks_xy = yolo_result.masks.xy
                for cls_id, mask_xy in zip(classes, masks_xy):
                    if int(cls_id) == 0:
                        lane_polygon = mask_xy; break
        
        is_detected = lane_polygon is not None
        self.status_pub.publish(Bool(data=bool(is_detected)))
        if not is_detected: return None, {}, combined_mask

        M = cv2.moments(lane_polygon)
        if M["m00"] == 0: return None, {}, combined_mask
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        planner = BezierPathPlanner(lane_polygon, self.CAR_POSITION_PIXEL, centroid, self.INITIAL_LOOKAHEAD_DISTANCE)
        control_points, lookahead_distance = planner.generate_control_points()
        bezier_path = planner.compute_bezier_curve(control_points)
        lookahead_point = planner.find_lookahead_point(bezier_path, self.CAR_POSITION_PIXEL, lookahead_distance)
        
        # 픽셀 공간에서의 기하학적 각도 계산
        pixel_space_angle_rad = self.get_pixel_space_angle(lookahead_point)
        
        viz_data = {'lane_polygon': lane_polygon, 'control_points': control_points, 'bezier_path': bezier_path, 'lookahead_point': lookahead_point}
        return pixel_space_angle_rad, viz_data, combined_mask

    def get_pixel_space_angle(self, lookahead_point):
        x1, y1 = self.CAR_POSITION_PIXEL
        x2, y2 = lookahead_point
        delta_y, delta_x = y1 - y2, x1 - x2
        angle_rad = math.atan2(delta_x, delta_y)
        return (angle_rad + math.pi) % (2 * math.pi) - math.pi

    def publish_visualization(self, raw_image, area_mask, viz_data, final_steering_angle_rad):
        # (이전 답변과 거의 동일, 텍스트만 수정)
        viz_image = raw_image.copy()
        if np.any(area_mask):
            green_overlay = np.zeros_like(viz_image)
            green_overlay[area_mask > 0] = (0, 255, 0)
            viz_image = cv2.addWeighted(viz_image, 1, green_overlay, 0.4, 0)
        
        if 'lane_polygon' in viz_data: cv2.polylines(viz_image, [viz_data['lane_polygon'].astype(np.int32)], True, (255, 0, 0), 2)
        if 'bezier_path' in viz_data: cv2.polylines(viz_image, [viz_data['bezier_path'].astype(np.int32)], False, (0, 255, 255), 3)
        if 'control_points' in viz_data:
            for pt in viz_data['control_points']: cv2.circle(viz_image, (int(pt[0]), int(pt[1])), 8, (255, 100, 100), -1)
        if 'lookahead_point' in viz_data:
            pt = viz_data['lookahead_point']
            cv2.circle(viz_image, (int(pt[0]), int(pt[1])), 12, (0, 0, 255), -1)
            
        # [Hinton's MOD] 최종 발행되는 조향각을 Degree로 변환하여 표시
        steer_deg = math.degrees(final_steering_angle_rad)
        steer_text = f"Steer Angle: {steer_deg:.1f} deg"
        cv2.putText(viz_image, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        _, jpeg_buffer = cv2.imencode('.jpg', viz_image)
        viz_msg = CompressedImage(data=jpeg_buffer.tobytes(), format="jpeg")
        viz_msg.header.stamp = self.get_clock().now().to_msg()
        self.viz_pub.publish(viz_msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down the planning thread pool.")
        self._is_shutting_down = True
        self.planning_executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloBezierAngleNode()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally: 
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()