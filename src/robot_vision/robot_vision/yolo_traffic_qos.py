#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: yolo_traffic_qos.py (최종 해결본 v3 - RealSense SDK 적용 및 supply_status 토픽 추가)
# 수정 사항:
# 1. 압축 깊이 토픽 대신 원본(Raw) 깊이 토픽('/camera/aligned_depth_to_color/image_raw')을 구독
# 2. pyrealsense2 라이브러리를 사용하여 CameraInfo를 intrinsics 객체로 변환
# 3. 수동 거리 계산 대신 rs2.rs2_deproject_pixel_to_point 함수를 사용하여 3D 좌표 획득
# 4. supply 객체 감지 시 /supply_status 토픽에 Bool 메시지(True)를 발행하는 기능 추가

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import message_filters
import traceback

# [수정] 원본 Image 메시지 타입과 pyrealsense2 라이브러리 import
from sensor_msgs.msg import Image as msg_Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Bool  # [추가] Bool 메시지 타입 import
from cv_bridge import CvBridge

# [수정] pyrealsense2 SDK 추가
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_traffic_node')
        self.get_logger().info("--- YOLO Vision Node (RealSense SDK Version) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using compute device: {self.device}")

        self.qos_profile_sensor_data = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.qos_profile_reliable_default = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.qos_profile_camera_info = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE)
        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value
        
        try:
            # ... (모델 로딩은 기존과 동일) ...
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

        # [수정] scaled_camera_intrinsics 대신 rs2.intrinsics 객체를 저장할 변수
        self.intrinsics = None
        self.camera_info_sub = None

        self.distance_pub = self.create_publisher(Point, '/supply_distance', self.qos_profile_reliable_default)
        # [추가] supply 감지 상태를 발행할 퍼블리셔 생성
        self.status_pub = self.create_publisher(Bool, '/supply_status', self.qos_profile_reliable_default)
        self.realsense_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/realsense/viz/compressed', self.qos_profile_sensor_data)
        self.usb_cam_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam/viz/compressed', self.qos_profile_sensor_data)

        info_topic = "/camera/color/camera_info"
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, self.qos_profile_camera_info)
        self.get_logger().info(f"Waiting for CameraInfo on topic: {info_topic}")

        usb_cam_topic = 'camera1/image_compressed'
        self.usb_cam_sub = self.create_subscription(CompressedImage, usb_cam_topic, self.usb_cam_callback, qos_profile=self.qos_profile_sensor_data)
        
    def camera_info_callback(self, info_msg):
        if self.intrinsics is not None: return
        self.get_logger().info("✅ CameraInfo received.")
        
        # [수정] CameraInfo 메시지를 pyrealsense2.intrinsics 객체로 변환
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
        
        self.initialize_image_sync()
        self.destroy_subscription(self.camera_info_sub)
        self.get_logger().info("CameraInfo subscription destroyed. Starting image synchronization.")

    def initialize_image_sync(self):
        realsense_img_topic = '/camera/color/image_raw/compressed'
        # [수정] 압축되지 않은 원본 깊이 토픽을 구독
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, realsense_img_topic, qos_profile=self.qos_profile_sensor_data)
        # [수정] 깊이 토픽의 타입과 QoS 프로파일을 센서 데이터에 맞게 변경
        depth_sub = message_filters.Subscriber(self, msg_Image, depth_topic, qos_profile=self.qos_profile_sensor_data)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.realsense_callback)
        self.get_logger().info("✅ YOLO Vision Node initialized successfully.")

    def realsense_callback(self, compressed_image_msg, depth_msg):
        try:
            if self.intrinsics is None:
                self.get_logger().warn("Waiting for camera intrinsics...")
                return

            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # YOLO 처리를 위해 리사이즈하지 않은 원본 사용
            
            # [수정] 압축 해제 과정이 필요 없으므로 바로 cv2 이미지로 변환
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
            
            # [수정] run_supply_tracking이 감지 여부를 반환하도록 변경
            supply_detected = self.run_supply_tracking(cv_color, cv_depth)
            
            # [추가] 감지 상태를 /supply_status 토픽으로 발행
            status_msg = Bool()
            status_msg.data = supply_detected
            self.status_pub.publish(status_msg)
            
            # 시각화 이미지는 리사이즈하여 발행
            viz_color = cv2.resize(cv_color, (self.proc_width, self.proc_height))
            self.publish_compressed_viz(self.realsense_viz_pub, viz_color)
            
        except Exception as e:
            self.get_logger().error(f"Error in Realsense callback: {e}\n{traceback.format_exc()}")

    def run_supply_tracking(self, color_image, depth_image):
        if self.intrinsics is None: return False

        # [추가] 프레임 내에서 supply가 감지되었는지 추적하기 위한 플래그
        supply_detected_in_frame = False

        # [수정] YOLO 모델에는 리사이즈된 이미지를 입력
        resized_color_image = cv2.resize(color_image, (self.proc_width, self.proc_height))
        results = self.supply_model(resized_color_image, verbose=False)

        for box in results[0].boxes:
            if int(box.cls) == 0:
                # [추가] supply가 하나라도 감지되면 플래그를 True로 설정
                supply_detected_in_frame = True

                # 감지된 bounding box 좌표는 리사이즈된 이미지 기준이므로, 원본 이미지 기준으로 다시 스케일링
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                orig_x1 = int(x1 * self.intrinsics.width / self.proc_width)
                orig_y1 = int(y1 * self.intrinsics.height / self.proc_height)
                orig_x2 = int(x2 * self.intrinsics.width / self.proc_width)
                
                cx = (orig_x1 + orig_x2) // 2
                cy = (orig_y1 + int(y2 * self.intrinsics.height / self.proc_height)) // 2

                if 0 <= cy < self.intrinsics.height and 0 <= cx < self.intrinsics.width:
                    depth_in_mm = depth_image[cy, cx]
                    if depth_in_mm > 0:
                        # [수정] SDK 함수를 사용하여 3D 좌표 계산
                        result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], depth_in_mm)
                        
                        point_msg = Point(
                            x=float(result[2] / 1000.0),
                            y=float(-result[0] / 1000.0),
                            z=float(-result[1] / 1000.0)
                        )
                        self.distance_pub.publish(point_msg)
                        
                        label = f"Supply Box: x={point_msg.x:.2f}m, y={point_msg.y:.2f}m, z={point_msg.z:.2f}m"
                        cv2.rectangle(color_image, (orig_x1, orig_y1), (orig_x2, int(y2 * self.intrinsics.height / self.proc_height)), (0, 255, 255), 2)
                        cv2.putText(color_image, label, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # [수정] 해당 프레임의 감지 상태를 반환
        return supply_detected_in_frame

    def usb_cam_callback(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            results_traffic = self.traffic_detection_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_traffic_detections(annotated_image, results_traffic)
            self.publish_compressed_viz(self.usb_cam_viz_pub, annotated_image)
        except Exception as e:
            self.get_logger().error(f"Error in USB Cam callback: {e}\n{traceback.format_exc()}")

    def publish_compressed_viz(self, publisher, cv_image):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
        publisher.publish(msg)

    def draw_marker_detections(self, image, results):
        # 이 함수는 예시이며, 실제 구현에 맞게 수정해야 합니다.
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls)
                label = self.marker_class_names[cls_id]
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image

    def draw_traffic_detections(self, image, results):
        # 이 함수는 예시이며, 실제 구현에 맞게 수정해야 합니다.
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls)
                label = self.traffic_model_class_names[cls_id]
                color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
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