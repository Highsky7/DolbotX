#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: marker_uniform_node.py
# DESCRIPTION:
# 이 노드는 USB 카메라를 사용하여 비전 마커와 군복을 인식하는 역할을 전담합니다.
# 아군('ROKA')과 적군('Enemy')을 식별하여 '/led_control' 토픽을 통해
# 외부 장치(예: LED)가 현재 상황을 표시할 수 있도록 신호를 발행합니다.

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge

class MarkerUniformNode(Node):
    def __init__(self):
        super().__init__('marker_uniform_node')
        self.get_logger().info("--- Vision Marker & Uniform Detection Node ---")

        self.usb_cam_locks = {'cam1': threading.Lock(), 'cam2': threading.Lock()}
        
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        self.use_half = self.device == 'cuda'

        try:
            self.declare_parameter('marker_model_path', './vision_enemy3.onnx')
            marker_model_path = self.get_parameter('marker_model_path').get_parameter_value().string_value
            self.marker_model = YOLO(marker_model_path, task='detect')
            self.marker_class_names = ['A', 'E', 'Enemy', 'Heart', 'K', 'M', 'O', 'R', 'ROKA', 'Y']
            self.get_logger().info("✅ Marker/Uniform ONNX model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            self.destroy_node(); return

        # 퍼블리셔 설정
        self.led_pub = self.create_publisher(String, '/led_control', 10)
        self.usb_cam1_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam1_marker/viz/compressed', 10)
        self.usb_cam2_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam2_marker/viz/compressed', 10)

        self.yolo_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='marker_worker')
        self._is_shutting_down = False

        # 구독자 설정
        usb_cam1_topic = 'camera1/image_raw/compressed'
        self.usb_cam1_sub = self.create_subscription(
            CompressedImage, usb_cam1_topic, lambda msg: self.usb_cam_callback(msg, 'cam1'), 10)
        
        usb_cam2_topic = 'camera2/image_raw/compressed'
        self.usb_cam2_sub = self.create_subscription(
            CompressedImage, usb_cam2_topic, lambda msg: self.usb_cam_callback(msg, 'cam2'), 10)
        
        self.get_logger().info("✅ Marker & Uniform Node initialized successfully.")

    def usb_cam_callback(self, compressed_msg, camera_id):
        if self._is_shutting_down: return
        lock = self.usb_cam_locks[camera_id]
        if lock.acquire(blocking=False):
            try:
                self.yolo_executor.submit(self._process_usb_cam_data, compressed_msg, camera_id)
            finally:
                pass # The lock is released in the worker thread
        else:
            self.get_logger().warn(f"Dropping a frame from {camera_id}, processing is busy.", throttle_duration_sec=1)

    def _process_usb_cam_data(self, compressed_msg, camera_id):
        lock = self.usb_cam_locks[camera_id]
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().warn(f"Failed to decompress USB cam image from {camera_id}.")
                return

            # 마커 탐지
            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False, half=self.use_half)
            roka_found, enemy_found = False, False
            for r in results_marker:
                for box in r.boxes.cpu().numpy():
                    label = self.marker_class_names[int(box.cls[0])]
                    if label == 'ROKA': roka_found = True
                    elif label == 'Enemy': enemy_found = True
            
            # LED 제어 메시지 발행 (ROKA 우선)
            led_data = "roka" if roka_found else "enemy" if enemy_found else "none"
            self.led_pub.publish(String(data=led_data))
            
            # 시각화 이미지 생성 및 발행
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            viz_publisher = self.usb_cam1_viz_pub if camera_id == 'cam1' else self.usb_cam2_viz_pub
            self.publish_compressed_viz(viz_publisher, annotated_image)

        except Exception as e:
            self.get_logger().error(f"Error in Marker USB Cam worker ({camera_id}): {e}\n{traceback.format_exc()}")
        finally:
            lock.release()

    def publish_compressed_viz(self, publisher, cv_image):
        msg = CompressedImage(format="jpeg")
        msg.header.stamp = self.get_clock().now().to_msg()
        success, encoded_image = cv2.imencode('.jpg', cv_image)
        if success:
            msg.data = encoded_image.tobytes()
            publisher.publish(msg)

    def draw_marker_detections(self, image, results):
        for r in results:
            for box in r.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.marker_class_names[cls_id] if cls_id < len(self.marker_class_names) else "Unknown"
                color = (0, 255, 0) if label == 'ROKA' else (255, 0, 0) if label == 'Enemy' else (200, 200, 200)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return image

    def destroy_node(self):
        self.get_logger().info("Shutting down the thread pool.")
        self._is_shutting_down = True
        self.yolo_executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MarkerUniformNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()