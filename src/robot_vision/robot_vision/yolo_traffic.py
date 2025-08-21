#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: yolo_traffic.py
# DESCRIPTION: camera_info 일회성 구독 및 pyrealsense2 SDK를 이용한 거리 계산 로직 적용 버전

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import message_filters
import traceback

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from cv_bridge import CvBridge

# pyrealsense2 SDK 추가
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_traffic_node')
        self.get_logger().info("--- YOLO Vision Node (SDK & Single CamInfo Sub) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using compute device: {self.device}")
        
        # 처리할 이미지 크기 파라미터 선언
        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value

        try:
            # 사용할 모델(보급품, 마커, 신호등)만 로드
            self.declare_parameter('supply_model_path', './tracking.pt')
            self.declare_parameter('marker_model_path', './vision_enemy2.pt')
            self.declare_parameter('traffic_model_path', './traffic_light.pt')
            
            supply_model_path = self.get_parameter('supply_model_path').get_parameter_value().string_value
            self.supply_model = YOLO(supply_model_path).to(self.device)
            
            marker_model_path = self.get_parameter('marker_model_path').get_parameter_value().string_value
            self.marker_model = YOLO(marker_model_path).to(self.device)
            self.marker_class_names = ['A', 'E', 'Enemy', 'Heart', 'K', 'M', 'O', 'R', 'ROKA', 'Y']

            traffic_model_path = self.get_parameter('traffic_model_path').get_parameter_value().string_value
            self.traffic_detection_model = YOLO(traffic_model_path).to(self.device)
            self.traffic_model_class_names = ['red', 'green']

        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO models: {e}")
            self.destroy_node()
            return

        # rs2.intrinsics 객체를 저장할 변수
        self.intrinsics = None
        
        # Publisher 선언
        self.distance_pub = self.create_publisher(Point, '/supply_distance', 1)
        self.status_pub = self.create_publisher(Bool, '/supply_status', 1)

        self.realsense_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/realsense/viz/compressed', 1)
        self.usb_cam_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam/viz/compressed', 1)

        # Subscriber 선언 (USB Cam은 즉시 시작)
        usb_cam_topic = 'camera1/image_compressed'
        self.usb_cam_sub = self.create_subscription(CompressedImage, usb_cam_topic, self.usb_cam_callback, 1)

        # CameraInfo를 먼저 받기 위한 구독자 생성
        info_topic = "/camera/color/camera_info"
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, 1)
        self.get_logger().info(f"Waiting for CameraInfo on topic: {info_topic}")

    def camera_info_callback(self, info_msg):
        """
        CameraInfo를 한 번만 받아 self.intrinsics를 설정하고,
        이미지/깊이 메시지 동기화를 시작한 뒤 자신을 파괴하는 콜백.
        """
        if self.intrinsics is not None:
            return
            
        self.get_logger().info("✅ CameraInfo received.")
        
        # CameraInfo 메시지를 pyrealsense2.intrinsics 객체로 변환
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = info_msg.width
        self.intrinsics.height = info_msg.height
        self.intrinsics.ppx = info_msg.k[2]
        self.intrinsics.ppy = info_msg.k[5]
        self.intrinsics.fx = info_msg.k[0]
        self.intrinsics.fy = info_msg.k[4]
        if info_msg.distortion_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif info_msg.distortion_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in info_msg.d]
        
        # Intrinsics를 성공적으로 받으면 이미지 동기화 시작
        self.initialize_image_sync()
        
        # 이 구독자는 더 이상 필요 없으므로 파괴
        self.destroy_subscription(self.camera_info_sub)
        self.get_logger().info("CameraInfo subscription destroyed. Starting image synchronization.")

    def initialize_image_sync(self):
        """
        RealSense의 이미지와 깊이 토픽에 대한 message_filters 동기화를 설정.
        """
        realsense_img_topic = '/camera/color/image_raw/compressed'
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, realsense_img_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub], queue_size=5, slop=0.5)
        self.ts.registerCallback(self.realsense_callback)
        self.get_logger().info("✅ YOLO Vision Node initialized successfully.")

    def realsense_callback(self, compressed_image_msg, depth_msg):
        try:
            if self.intrinsics is None:
                self.get_logger().warn("Waiting for camera intrinsics...")
                return

            # 컬러 이미지는 압축 해제 후 그대로 사용 (나중에 추론 및 시각화에 활용)
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 깊이 이미지는 원본 해상도(16UC1, mm 단위)로 변환
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

            # 보급품 추적 알고리즘 실행
            # 시각화를 위해 라벨이 그려질 원본 컬러 이미지를 함께 전달
            supply_detected = self.run_supply_tracking(cv_color, cv_depth)
            
            # [추가] 감지 상태를 /supply_status 토픽으로 발행
            status_msg = Bool()
            status_msg.data = supply_detected
            self.status_pub.publish(status_msg)
            
            # 시각화 이미지는 지정된 크기로 리사이즈하여 발행
            viz_color = cv2.resize(cv_color, (self.proc_width, self.proc_height))
            self.publish_compressed_viz(self.realsense_viz_pub, viz_color)
            
        except Exception as e:
            self.get_logger().error(f"Error in Realsense callback: {e}\n{traceback.format_exc()}")

    def run_supply_tracking(self, color_image, depth_image):
        if self.intrinsics is None: return
        
        # [추가] 프레임 내에서 supply가 감지되었는지 추적하기 위한 플래그
        supply_detected_in_frame = False

        # YOLO 모델에는 리사이즈된 이미지를 입력
        resized_color_image = cv2.resize(color_image, (self.proc_width, self.proc_height))
        results = self.supply_model(resized_color_image, verbose=False)

        for box in results[0].boxes:
            if int(box.cls) == 0:
                
                # supply 감지 시 플래그 True 설정
                supply_detected_in_frame = True
                # 감지된 bounding box 좌표는 리사이즈된 이미지 기준
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 깊이 값 참조를 위해 원본 이미지 기준으로 좌표 스케일링
                orig_x1 = int(x1 * self.intrinsics.width / self.proc_width)
                orig_y1 = int(y1 * self.intrinsics.height / self.proc_height)
                orig_x2 = int(x2 * self.intrinsics.width / self.proc_width)
                orig_y2 = int(y2 * self.intrinsics.height / self.proc_height)

                cx = (orig_x1 + orig_x2) // 2
                cy = (orig_y1 + orig_y2) // 2

                if 0 <= cy < self.intrinsics.height and 0 <= cx < self.intrinsics.width:
                    # 원본 해상도 깊이 이미지에서 거리(mm) 값 추출
                    depth_in_mm = depth_image[cy, cx]
                    
                    if depth_in_mm > 0:
                        # SDK 함수를 사용하여 3D 좌표 계산
                        result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], depth_in_mm)
                        
                        # ROS 좌표계에 맞게 변환하여 Point 메시지 생성 (단위: m)
                        # result[2]: Z (앞), result[0]: X (오른쪽), result[1]: Y (아래)
                        # ROS: x (앞), y (왼쪽), z (위)
                        point_msg = Point(
                            x=float(result[2] / 1000.0),
                            y=float(-result[0] / 1000.0),
                            z=float(-result[1] / 1000.0)
                        )
                        self.distance_pub.publish(point_msg)
                        
                        label = f"Supply Box: x={point_msg.x:.2f}m, y={point_msg.y:.2f}m, z={point_msg.z:.2f}m"
                        
                        # 시각화를 위해 원본 이미지에 bounding box와 라벨 그리기
                        cv2.rectangle(color_image, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 255), 2)
                        cv2.putText(color_image, label, (cx-200, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # return supply_detection_status boolean
        return supply_detected_in_frame
    
    def usb_cam_callback(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 1. marker_model(vision_enemy.pt) 추론 및 결과 그리기
            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            
            # 2. traffic_detection_model 추론 및 결과 그리기
            results_traffic = self.traffic_detection_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_traffic_detections(annotated_image, results_traffic)

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
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                cv2.putText(image, f"{label}: {conf:.2f}", (box_center_x, box_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    def draw_traffic_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                cls_id = int(box.cls[0])
                if cls_id < len(self.traffic_model_class_names):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = self.traffic_model_class_names[cls_id]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    box_center_x = (x1 + x2) // 2
                    box_center_y = (y1 + y2) // 2
                    cv2.putText(image, f"{label}: {conf:.2f}", (box_center_x, box_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()