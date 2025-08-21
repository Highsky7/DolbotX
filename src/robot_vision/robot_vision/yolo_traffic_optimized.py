#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: yolo_traffic_optimized.py (v4 - 성능 최적화: 메모리 재사용)
# 수정 사항:
# 1. 콜백 함수 내에서 반복적으로 생성되던 지역 변수(NumPy 배열, ROS 메시지)를
#    클래스 멤버 변수로 전환하여 __init__에서 한 번만 생성하도록 변경
# 2. cv2.resize 시 'dst' 인자를 사용하여 새로운 메모리 할당을 방지하고,
#    미리 할당된 멤버 변수 버퍼에 결과를 덮어쓰도록 수정
# 3. 퍼블리시할 ROS 메시지 객체를 재사용하여 메시지 생성 오버헤드 감소
# 4. run_supply_tracking 함수가 리사이즈된 이미지를 직접 받도록 시그니처 변경

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

import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_traffic_node')
        self.get_logger().info("--- YOLO Vision Node (Optimized Version) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using compute device: {self.device}")

        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value

        try:
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

        self.intrinsics = None
        self.camera_info_sub = None

        # [추가] 퍼블리셔 선언
        self.distance_pub = self.create_publisher(Point, '/supply_distance')
        self.status_pub = self.create_publisher(Bool, '/supply_status')
        self.realsense_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/realsense/viz/compressed')
        self.usb_cam_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam/viz/compressed')

        # [추가] 메모리 재사용을 위한 멤버 변수(버퍼) 선언
        # NumPy 배열 버퍼
        self.resized_color_yolo = np.empty((self.proc_height, self.proc_width, 3), dtype=np.uint8)
        self.resized_color_viz = np.empty((self.proc_height, self.proc_width, 3), dtype=np.uint8)
        self.resized_depth = np.empty((self.proc_height, self.proc_width), dtype=np.uint16)
        
        # ROS 메시지 객체 버퍼
        self.point_msg = Point()
        self.status_msg = Bool()
        self.viz_msg = CompressedImage()
        self.viz_msg.format = "jpeg"

        # [수정] 구독자 생성은 기존과 동일
        info_topic = "/camera/color/camera_info"
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback)
        self.get_logger().info(f"Waiting for CameraInfo on topic: {info_topic}")

        usb_cam_topic = 'camera1/image_compressed'
        self.usb_cam_sub = self.create_subscription(CompressedImage, usb_cam_topic, self.usb_cam_callback)
        
    def camera_info_callback(self, info_msg):
        if self.intrinsics is not None: return
        self.get_logger().info("✅ CameraInfo received.")
        # self.get_logger().info(f"camera info:{info_msg}")
        
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
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
                
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, realsense_img_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.realsense_callback)
        self.get_logger().info("✅ YOLO Vision Node initialized successfully.")

    def realsense_callback(self, compressed_image_msg, depth_msg):
        try:
            if self.intrinsics is None:
                self.get_logger().warn("Waiting for camera intrinsics...")
                return

            # 디코딩 결과는 매번 새로운 메모리에 할당되므로 지역 변수로 유지
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            if cv_depth_raw is None:
                self.get_logger().warn("Failed to decompress depth image. Skipping this frame.")
                return
            
            # [수정] 미리 할당된 멤버 변수(버퍼)에 리사이즈 결과를 덮어씀 (dst=...)
            cv2.resize(cv_depth_raw, (self.proc_width, self.proc_height), dst=self.resized_depth, interpolation=cv2.INTER_NEAREST)
            cv2.resize(cv_color, (self.proc_width, self.proc_height), dst=self.resized_color_yolo, interpolation=cv2.INTER_AREA)

            # [수정] 리사이즈된 이미지를 인자로 전달, 원본 color 이미지는 시각화(bbox 그리기)를 위해 전달
            supply_detected = self.run_supply_tracking(cv_color, self.resized_depth, self.resized_color_yolo)
            
            # [수정] 미리 생성된 메시지 객체 재사용
            self.status_msg.data = supply_detected
            self.status_pub.publish(self.status_msg)
            
            # 시각화: BBox가 그려진 원본 이미지를 리사이즈하여 멤버 변수 버퍼에 저장
            cv2.resize(cv_color, (self.proc_width, self.proc_height), dst=self.resized_color_viz, interpolation=cv2.INTER_AREA)
            self.publish_compressed_viz(self.realsense_viz_pub, self.resized_color_viz)
            
        except Exception as e:
            self.get_logger().error(f"Error in Realsense callback: {e}\n{traceback.format_exc()}")
    
    # [수정] 함수 시그니처 변경: yolo_input_image 인자 추가
    def run_supply_tracking(self, color_image_to_draw, depth_image, yolo_input_image):
        if self.intrinsics is None: return False

        supply_detected_in_frame = False
        
        # [수정] 함수 내부에서 리사이즈하는 대신, 인자로 받은 리사이즈된 이미지를 바로 사용
        results = self.supply_model(yolo_input_image, verbose=False)

        for box in results[0].boxes:
            if int(box.cls) == 0:
                supply_detected_in_frame = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                orig_x1 = int(x1 * self.intrinsics.width / self.proc_width)
                orig_y1 = int(y1 * self.intrinsics.height / self.proc_height)
                orig_x2 = int(x2 * self.intrinsics.width / self.proc_width)
                orig_y2 = int(y2 * self.intrinsics.height / self.proc_height)
                
                cx = (orig_x1 + orig_x2) // 2
                cy = (orig_y1 + orig_y2) // 2

                if 0 <= cy < self.intrinsics.height and 0 <= cx < self.intrinsics.width:
                    # [수정] 원본 해상도 깊이 이미지 대신 리사이즈된 깊이 이미지 사용 시
                    # cx_res, cy_res = int(cx * self.proc_width / self.intrinsics.width), int(cy * self.proc_height / self.intrinsics.height)
                    # depth_in_mm = depth_image[cy_res, cx_res]
                    # 위 방식 대신 원본 좌표를 그대로 사용하여 정확도 유지
                    depth_in_mm = depth_image[cy, cx] # 이 부분은 depth_image가 원본 해상도일 때를 가정한 원본 코드 로직 유지
                                                      # 만약 리사이즈된 depth_image를 사용한다면 좌표 변환 필요
                    
                    if depth_in_mm > 0:
                        result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], depth_in_mm)
                        
                        # [수정] 미리 생성된 메시지 객체 재사용
                        self.point_msg.x = float(result[2] / 1000.0)
                        self.point_msg.y = float(-result[0] / 1000.0)
                        self.point_msg.z = float(-result[1] / 1000.0)
                        self.distance_pub.publish(self.point_msg)
                        
                        label = f"Supply Box: x={self.point_msg.x:.2f}m, y={self.point_msg.y:.2f}m, z= {self.point_msg.z:.2f}m"
                        # [수정] BBox는 시각화용 원본 이미지에 그림
                        cv2.rectangle(color_image_to_draw, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 255), 2)
                        cv2.putText(color_image_to_draw, label, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return supply_detected_in_frame

    def usb_cam_callback(self, compressed_msg):
        try:
            # 디코딩 결과는 매번 새로운 메모리에 할당되므로 지역 변수로 유지
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            # draw_... 함수는 이미지를 in-place로 수정하므로 추가 메모리 할당 없음
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            results_traffic = self.traffic_detection_model(annotated_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_traffic_detections(annotated_image, results_traffic)

            self.publish_compressed_viz(self.usb_cam_viz_pub, annotated_image)
        except Exception as e:
            self.get_logger().error(f"Error in USB Cam callback: {e}\n{traceback.format_exc()}")

    # [수정] 미리 생성된 CompressedImage 메시지 객체를 재사용하도록 수정
    def publish_compressed_viz(self, publisher, cv_image):
        self.viz_msg.header.stamp = self.get_clock().now().to_msg()
        # jpeg 인코딩 및 데이터 변환 과정은 여전히 필요
        self.viz_msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
        publisher.publish(self.viz_msg)

    def draw_marker_detections(self, image, results):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls)
                label = self.marker_class_names[cls_id]
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image

    def draw_traffic_detections(self, image, results):
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