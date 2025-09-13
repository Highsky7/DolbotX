#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: vision_marker_detector.py
# DESCRIPTION:
# 이 노드는 USB 카메라를 사용하여 다양한 비전 마커를 인식하는 역할을 전담합니다.
# 'vision_enemy3.onnx' 모델을 사용하여 추론을 수행하며,
# 탐지 결과를 시각화하여 '/unified_vision/usb_camN_marker/viz/compressed' 토픽으로 발행합니다.
#
# MODIFIED BY: Geoffrey Hinton (for enhanced visualization)
# - 각 마커 클래스에 고유한 색상을 할당하여 즉각적인 식별이 가능하도록 개선
# - 텍스트 레이블에 배경을 추가하여 모든 영상 조건에서 최고의 가독성 확보
# - 텍스트 색상을 검은색으로, 두께를 강화하여 시인성 극대화

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

class VisionMarkerDetectorNode(Node):
    def __init__(self):
        super().__init__('vision_marker_detector_node')
        self.get_logger().info("--- Vision Marker Detection Node (Enhanced by Hinton) ---")

        self.usb_cam_locks = {'cam1': threading.Lock(), 'cam2': threading.Lock()}
        
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        self.use_half = self.device == 'cuda'

        try:
            # 비전 마커 탐지 모델 로드
            self.declare_parameter('marker_model_path', './vision_marker2.onnx')
            marker_model_path = self.get_parameter('marker_model_path').get_parameter_value().string_value
            self.marker_model = YOLO(marker_model_path, task='detect')
            self.marker_class_names = ['A', 'E', 'Heart', 'K', 'M', 'O', 'R', 'Y']
            
            # --- 시각화 개선: 각 클래스별 고유 색상 정의 (BGR 형식) ---
            self.marker_colors = {
                'A': (255, 0, 0),      # 파란색 (Blue)
                'E': (0, 255, 0),      # 초록색 (Green)
                'Heart': (0, 0, 255),    # 빨간색 (Red)
                'K': (255, 255, 0),    # 청록색 (Cyan)
                'M': (255, 0, 255),    # 자홍색 (Magenta)
                'O': (0, 165, 255),    # 주황색 (Orange)
                'R': (128, 0, 128),    # 보라색 (Purple)
                'Y': (0, 255, 255)     # 노란색 (Yellow)
            }
            self.get_logger().info("✅ Vision Marker ONNX model loaded successfully.")
            self.get_logger().info("✅ Enhanced visualization color palette is active.")

        except Exception as e:
            self.get_logger().error(f"Failed to load Vision Marker YOLO model: {e}")
            self.destroy_node(); return

        # 퍼블리셔 설정 (시각화 전용)
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
        
        self.get_logger().info("✅ Vision Marker Node initialized successfully.")

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
            
            # 시각화 이미지 생성 및 발행
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            viz_publisher = self.usb_cam1_viz_pub if camera_id == 'cam1' else self.usb_cam2_viz_pub
            self.publish_compressed_viz(viz_publisher, annotated_image)

        except Exception as e:
            self.get_logger().error(f"Error in Vision Marker USB Cam worker ({camera_id}): {e}\n{traceback.format_exc()}")
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
        """
        전문가의 손길로 개선된 시각화 함수:
        1. 각 마커 클래스에 고유 색상을 적용합니다.
        2. 텍스트 가독성을 위해 레이블에 채워진 배경 사각형을 추가합니다.
        3. 최고의 시인성을 위해 텍스트 색상을 검은색으로, 두께를 강화합니다.
        """
        for r in results:
            for box in r.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                
                if cls_id >= len(self.marker_class_names):
                    continue

                label = self.marker_class_names[cls_id]
                # 클래스에 할당된 고유 색상 가져오기 (없을 경우 회색)
                color = self.marker_colors.get(label, (128, 128, 128))
                
                # 1. 바운딩 박스 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 2. 텍스트 레이블 및 배경 준비
                text = f"{label}: {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                
                # --- [Hinton's Enhancement] 글자 두께 강화 ---
                font_thickness = 2
                
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                
                # 텍스트 배경을 위한 사각형 좌표 계산
                # 바운딩 박스 위쪽에 공간이 충분하면 위에, 아니면 아래에 표시
                if y1 - text_h - baseline > 0:
                    text_bg_y1 = y1 - text_h - baseline - 2
                    text_bg_y2 = y1
                    text_y = y1 - baseline // 2 - 2
                else:
                    text_bg_y1 = y2
                    text_bg_y2 = y2 + text_h + baseline + 2
                    text_y = y2 + text_h
                
                cv2.rectangle(image, (x1, text_bg_y1), (x1 + text_w, text_bg_y2), color, cv2.FILLED)
                
                # --- [Hinton's Enhancement] 텍스트 색상을 검은색(0,0,0)으로 변경하여 시인성 극대화 ---
                cv2.putText(image, text, (x1, text_y), font, font_scale, (0, 0, 0), font_thickness)

        return image

    def destroy_node(self):
        self.get_logger().info("Shutting down the thread pool.")
        self._is_shutting_down = True
        self.yolo_executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionMarkerDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()