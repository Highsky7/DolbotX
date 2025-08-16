#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class FireAndDoorDetector(Node):
    def __init__(self):
        super().__init__('fire_and_door_detector')
        self.get_logger().info('🔥🚪 Start Fire and Door Handle Detector.')
        
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using compute device: {self.device}")

        # QoS 프로파일 정의 (yolo_traffic_qos.py 참고)
        # 센서 데이터(이미지 등)에 적합한 Best Effort 프로파일
        self.qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 처리할 이미지 크기 파라미터 선언
        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value
        
        try:
            # 화재 감지 모델 로드
            self.declare_parameter('fire_detector_model_path', './fire.pt')
            fire_detector_model_path = self.get_parameter('fire_detector_model_path').get_parameter_value().string_value
            self.fire_detector_model = YOLO(fire_detector_model_path).to(self.device)
            self.fire_detector_class_names = ['Fire']
            
            # 문고리 감지 모델 로드 (추가된 부분)
            self.declare_parameter('door_handle_model_path', './door_handle.pt')
            door_handle_model_path = self.get_parameter('door_handle_model_path').get_parameter_value().string_value
            self.door_handle_model = YOLO(door_handle_model_path).to(self.device)
            self.door_handle_class_names = ['door_handle']
        
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO models: {e}")
            self.destroy_node()
            return
        
        # Publisher 선언 (QoS 프로파일 적용)
        self.detector_viz_pub = self.create_publisher(
            CompressedImage, 
            'fire_and_door_detector/compressed', 
            qos_profile=self.qos_profile_sensor_data
        )
        
        # Subscriber 선언 (QoS 프로파일 적용)
        usb_cam_topic = 'camera1/image_compressed'
        self.usb_cam_sub = self.create_subscription(
            CompressedImage, 
            usb_cam_topic, 
            self.usb_cam_callback, 
            qos_profile=self.qos_profile_sensor_data
        )
        
        self.get_logger().info("✅ Fire and Door Handle Detector Node initialized successfully.")
    
    def usb_cam_callback(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 1. 화재 감지 모델(fire.pt) 추론 및 결과 그리기
            results_fire = self.fire_detector_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_fire_detections(cv_image, results_fire)
            
            # 2. 문고리 감지 모델(door_handle.pt) 추론 및 결과 그리기
            results_door_handle = self.door_handle_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_door_handle_detections(annotated_image, results_door_handle)

            # 3. 최종 결과 이미지 발행
            self.publish_compressed_viz(self.detector_viz_pub, annotated_image)
        except Exception as e:
            self.get_logger().error(f"Error in USB Cam callback: {e}")
            
    def publish_compressed_viz(self, publisher, cv_image):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
        publisher.publish(msg)
        
    def draw_fire_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                cls_id = int(box.cls[0])
                if cls_id < len(self.fire_detector_class_names):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = self.fire_detector_class_names[cls_id]
                    # 화재는 빨간색으로 표시
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return image

    def draw_door_handle_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                cls_id = int(box.cls[0])
                if cls_id < len(self.door_handle_class_names):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = self.door_handle_class_names[cls_id]
                    # 문고리는 파란색으로 표시
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image

def main(args=None):
    rclpy.init(args=args)
    node = FireAndDoorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()