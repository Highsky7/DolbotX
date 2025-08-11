#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: hsv_traffic_modified.py

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

class HsvVisionNode(Node):
    def __init__(self):
        super().__init__('hsv_traffic_node')
        self.get_logger().info("--- HSV Vision Node (Image Raw) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using compute device: {self.device}")

        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value
        self.get_logger().info(f"Processing images at {self.proc_width}x{self.proc_height}")

        try:
            # 사용할 모델(보급품, 마커, 신호등)만 로드
            self.declare_parameter('supply_model_path', './tracking.pt')
            self.declare_parameter('marker_model_path', './vision_enemy2.pt')
            self.declare_parameter('traffic_model_path', './traffic_light.pt')
            
            supply_model_path = self.get_parameter('supply_model_path').get_parameter_value().string_value
            self.supply_model = YOLO(supply_model_path).to(self.device)
            
            marker_model_path = self.get_parameter('marker_model_path').get_parameter_value().string_value
            self.marker_model = YOLO(marker_model_path).to(self.device)
            self.marker_class_names = ['A', 'E', 'Enemy', 'Heart', 'K', 'M', 'O', 'R', 'ROKA',  'Y']

            traffic_model_path = self.get_parameter('traffic_model_path').get_parameter_value().string_value
            self.traffic_detection_model = YOLO(traffic_model_path).to(self.device)
            self.traffic_model_class_names = ['red', 'green']

        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO models: {e}")
            self.destroy_node()
            return
        
        self.scaled_camera_intrinsics = None
        
        # Publisher 선언 (Image 타입으로 변경)
        self.distance_pub = self.create_publisher(Point, '/supply_distance', 1)
        self.realsense_viz_pub = self.create_publisher(Image, '/unified_vision/realsense/viz', 1)
        self.usb_cam_viz_pub = self.create_publisher(Image, '/unified_vision/usb_cam/viz', 1)
        
        # Subscriber 선언 (Image 타입으로 변경)
        realsense_img_topic = '/camera/color/image_raw' # raw 이미지 토픽
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        info_topic = "/camera/color/camera_info"
        realsense_img_sub = message_filters.Subscriber(self, Image, realsense_img_topic) # CompressedImage -> Image
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub, info_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.realsense_callback)
        
        usb_cam_topic = '/camera1/image_raw' # raw 이미지 토픽
        self.usb_cam_sub = self.create_subscription(Image, usb_cam_topic, self.usb_cam_callback, 1) # CompressedImage -> Image
        
        self.get_logger().info("✅ HSV Vision Node initialized successfully.")

    def realsense_callback(self, img_msg, depth_msg, info_msg):
        try:
            # CompressedImage 디코딩 로직 제거, CvBridge로 바로 변환
            cv_color_orig = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            cv_depth_orig = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            cv_color = cv2.resize(cv_color_orig, (self.proc_width, self.proc_height), interpolation=cv2.INTER_AREA)
            cv_depth = cv2.resize(cv_depth_orig, (self.proc_width, self.proc_height), interpolation=cv2.INTER_NEAREST)
            
            if self.scaled_camera_intrinsics is None:
                self.scale_camera_info(info_msg)
            
            self.run_supply_tracking(cv_color, cv_depth)
            
            # 시각화 이미지 발행
            self.publish_image_viz(self.realsense_viz_pub, cv_color)
            
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

    def usb_cam_callback(self, msg):
        try:
            # CompressedImage 디코딩 로직 제거, CvBridge로 바로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            
            results_traffic = self.traffic_detection_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_traffic_detections(annotated_image, results_traffic)

            # 최종 결과 이미지 발행
            self.publish_image_viz(self.usb_cam_viz_pub, annotated_image)
        except Exception as e:
            self.get_logger().error(f"Error in USB Cam callback: {e}")

    # CompressedImage 발행 함수를 Image 발행 함수로 변경
    def publish_image_viz(self, publisher, cv_image):
        # cv2_to_imgmsg를 사용하여 OpenCV 이미지를 Image 메시지로 변환
        msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        publisher.publish(msg)

    def draw_marker_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.marker_class_names[cls_id] if cls_id < len(self.marker_class_names) else "Unknown"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    def draw_traffic_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.traffic_model_class_names[cls_id] if cls_id < len(self.traffic_model_class_names) else "Unknown"
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