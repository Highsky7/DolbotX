#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: hsv_vision_node.py (Corrected and Modified)

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import ast
import message_filters

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

class HsvVisionNode(Node):
    def __init__(self):
        super().__init__('hsv_traffic_node')
        self.get_logger().info("--- HSV Vision Node (Decoupled) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using compute device: {self.device}")
        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value
        self.get_logger().info(f"Processing images at {self.proc_width}x{self.proc_height}")
        try:
            # 기존 모델 로드
            self.declare_parameter('supply_model_path', './tracking2.pt')
            self.declare_parameter('marker_model_path', './vision_enemy.pt')
            self.declare_parameter('traffic_model_path', './traffic_light.pt')
            supply_model_path = self.get_parameter('supply_model_path').get_parameter_value().string_value
            self.supply_model = YOLO(supply_model_path).to(self.device)
            marker_model_path = self.get_parameter('marker_model_path').get_parameter_value().string_value
            self.marker_model = YOLO(marker_model_path).to(self.device)
            self.marker_class_names = ['A', 'E', 'Enemy', 'Heart', 'K', 'M', 'O', 'R', 'ROKA',  'Y']


            traffic_model_path = self.get_parameter('traffic_model_path').get_parameter_value().string_value
            self.new_detection_model = YOLO(traffic_model_path).to(self.device)
            self.new_model_class_names = ['red', 'green']

        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO models: {e}")
            self.destroy_node()
            return
        self.declare_parameter('lower_hsv_bound', '[0, 0, 50]')
        self.declare_parameter('upper_hsv_bound', '[100, 40, 220]')
        lower_hsv_param = self.get_parameter('lower_hsv_bound').get_parameter_value().string_value
        upper_hsv_param = self.get_parameter('upper_hsv_bound').get_parameter_value().string_value
        self.lower_hsv_bound = np.array(ast.literal_eval(lower_hsv_param))
        self.upper_hsv_bound = np.array(ast.literal_eval(upper_hsv_param))
        self.kernel = np.ones((5, 5), np.uint8)
        self.scaled_camera_intrinsics = None
        self.distance_pub = self.create_publisher(Point, '/supply_distance', 1)
        self.realsense_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/realsense/viz/compressed', 1)
        self.usb_cam_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam/viz/compressed', 1)
        self.mask_pub = self.create_publisher(Image, '/path_planning/hsv/mask', 1)
        self.depth_for_path_pub = self.create_publisher(Image, '/path_planning/hsv/depth', 1)
        self.info_for_path_pub = self.create_publisher(CameraInfo, '/path_planning/hsv/info', 1)
        realsense_img_topic = '/camera/color/image_raw/compressed'
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        info_topic = "/camera/color/camera_info"
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, realsense_img_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub, info_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.realsense_callback)
        usb_cam_topic = '/camera1/image_compressed'
        self.usb_cam_sub = self.create_subscription(CompressedImage, usb_cam_topic, self.usb_cam_callback, 1)
        self.get_logger().info("✅ HSV Vision Node initialized successfully.")

    def realsense_callback(self, compressed_img_msg, depth_msg, info_msg):
        try:
            np_arr = np.frombuffer(compressed_img_msg.data, np.uint8)
            cv_color_orig = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_orig = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            cv_color = cv2.resize(cv_color_orig, (self.proc_width, self.proc_height), interpolation=cv2.INTER_AREA)
            cv_depth = cv2.resize(cv_depth_orig, (self.proc_width, self.proc_height), interpolation=cv2.INTER_NEAREST)
            if self.scaled_camera_intrinsics is None:
                self.scale_camera_info(info_msg)
            self.run_supply_tracking(cv_color, cv_depth)
            drivable_mask = self.create_drivable_mask(cv_color)
            
            mask_msg = self.bridge.cv2_to_imgmsg(drivable_mask, "mono8")
            mask_msg.header = depth_msg.header
            self.mask_pub.publish(mask_msg)
            
            depth_img_msg = self.bridge.cv2_to_imgmsg(cv_depth, "16UC1")
            depth_img_msg.header = depth_msg.header
            self.depth_for_path_pub.publish(depth_img_msg)
            
            self.info_for_path_pub.publish(info_msg)

            if np.any(drivable_mask):
                cv_color[drivable_mask > 0] = cv2.addWeighted(cv_color[drivable_mask > 0], 0.5, np.full_like(cv_color[drivable_mask > 0], (0, 255, 0)), 0.5, 0)
            self.publish_compressed_viz(self.realsense_viz_pub, cv_color)
        except Exception as e:
            self.get_logger().error(f"Error in Realsense callback: {e}", exc_info=True)

    def scale_camera_info(self, info_msg):
        scale_x = self.proc_width / info_msg.width
        scale_y = self.proc_height / info_msg.height
        self.scaled_camera_intrinsics = {
            'fx': info_msg.k[0] * scale_x, 'fy': info_msg.k[4] * scale_y,
            'ppx': info_msg.k[2] * scale_x, 'ppy': info_msg.k[5] * scale_y
        }
        self.get_logger().info(f"Cam intrinsics scaled for vision node: {self.scaled_camera_intrinsics}")

    def create_drivable_mask(self, cv_color):
        hsv = cv2.cvtColor(cv_color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv_bound, self.upper_hsv_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        return mask

    def run_supply_tracking(self, color_image, depth_image):
        if self.scaled_camera_intrinsics is None: return
        results = self.supply_model(color_image, verbose=False)
        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                    depth_in_meters = depth_image[cy, cx] / 1000.0
                    if depth_in_meters > 0:
                        fx, fy, ppx, ppy = self.scaled_camera_intrinsics['fx'], self.scaled_camera_intrinsics['fy'], self.scaled_camera_intrinsics['ppx'], self.scaled_camera_intrinsics['ppy']
                        x = (cx - ppx) * depth_in_meters / fx
                        y = (cy - ppy) * depth_in_meters / fy
                        point_msg = Point(x=depth_in_meters, y=-x, z=-y)
                        self.distance_pub.publish(point_msg)
                        label = f"Supply Box: x={point_msg.x:.2f}m, y={point_msg.y:.2f}m, z={point_msg.z:.2f}m"
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def usb_cam_callback(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 1. 기존 marker_model(vision_enemy.pt) 추론 및 결과 그리기
            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            
            # 2. 새로운 new_detection_model 추론 및 결과 그리기
            results_new = self.new_detection_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_new_detections(annotated_image, results_new)

            # 3. 최종 결과 이미지 발행
            self.publish_compressed_viz(self.usb_cam_viz_pub, annotated_image)
        except Exception as e:
            self.get_logger().error(f"Error in USB Cam callback: {e}")

    def publish_compressed_viz(self, publisher, cv_image):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
        publisher.publish(msg)

    def draw_marker_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.marker_class_names[cls_id] if cls_id < len(self.marker_class_names) else "Unknown"
                # 초록색(0, 255, 0)으로 바운딩 박스 표시
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    # <<< [추가] 새로운 모델의 탐지 결과를 시각화하는 함수 >>>
    def draw_new_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.new_model_class_names[cls_id] if cls_id < len(self.new_model_class_names) else "Unknown"
                # 파란색(255, 0, 0)으로 바운딩 박스를 그려 기존 탐지와 구분합니다.
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image

def main(args=None):
    rclpy.init(args=args)
    node = HsvVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
