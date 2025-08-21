#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: yolo_vision_node_final_fix.py
# AUTHOR: Geoffrey Hinton (Optimized & Fixed Version)
# DESCRIPTION:
# 1. [FIX] 'self.executor' 변수명을 'self.yolo_executor'로 변경하여 rclpy.Node의 예약어와 충돌하는 문제 해결.
# 2. [FIX] 종료 시그널 확인 로직을 보다 안정적인 플래그 방식으로 수정.

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import message_filters
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from cv_bridge import CvBridge

import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node_optimized')
        self.get_logger().info("--- YOLO Vision Node (Hinton's Optimized Architecture) ---")
        
        self.lock = threading.Lock()
        
        self.bridge = CvBridge()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using compute device: {self.device}")

        # QoS Profiles
        self.qos_profile_sensor_data = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.qos_profile_reliable_default = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.qos_profile_camera_info = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE)
        
        # Parameters
        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value

        # Model Loading
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

        # Publishers
        self.distance_pub = self.create_publisher(Point, '/supply_distance', self.qos_profile_reliable_default)
        self.status_pub = self.create_publisher(Bool, '/supply_status', self.qos_profile_reliable_default)
        self.realsense_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/realsense/viz/compressed', self.qos_profile_sensor_data)
        self.usb_cam_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam/viz/compressed', self.qos_profile_sensor_data)

        # Pre-allocated buffers
        self.resized_color_yolo = np.empty((self.proc_height, self.proc_width, 3), dtype=np.uint8)
        self.resized_depth = np.empty((self.proc_height, self.proc_width), dtype=np.uint16)
        
        # Reusable ROS message objects
        self.point_msg = Point()
        self.status_msg = Bool()

        # [수정] 변수명을 'self.executor'에서 'self.yolo_executor'로 변경
        self.yolo_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='yolo_worker')
        # [추가] 노드 종료 시 안전하게 스레드 풀에 작업을 제출하지 않도록 플래그 추가
        self._is_shutting_down = False

        # Subscriptions
        info_topic = "/camera/color/camera_info"
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, self.qos_profile_camera_info)
        self.get_logger().info(f"Waiting for CameraInfo on topic: {info_topic}")

        usb_cam_topic = 'camera1/image_compressed'
        self.usb_cam_sub = self.create_subscription(CompressedImage, usb_cam_topic, self.usb_cam_callback, qos_profile=self.qos_profile_sensor_data)
        
    def camera_info_callback(self, info_msg):
        if self.intrinsics is not None: return
        self.get_logger().info("✅ CameraInfo received.")
        
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = info_msg.width
        self.intrinsics.height = info_msg.height
        self.intrinsics.ppx = info_msg.k[2]
        self.intrinsics.ppy = info_msg.k[5]
        self.intrinsics.fx = info_msg.k[0]
        self.intrinsics.fy = info_msg.k[4]
        self.intrinsics.model = rs2.distortion.brown_conrady if info_msg.distortion_model == 'plumb_bob' else rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in info_msg.d]
        
        self.initialize_image_sync()
        if self.camera_info_sub:
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None
        self.get_logger().info("CameraInfo subscription destroyed. Starting image synchronization.")

    def initialize_image_sync(self):
        realsense_img_topic = '/camera/color/image_raw/compressed'
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
                
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, realsense_img_topic, qos_profile=self.qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=self.qos_profile_sensor_data)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub], queue_size=5, slop=0.2)
        self.ts.registerCallback(self.realsense_callback)
        self.get_logger().info("✅ YOLO Vision Node initialized successfully.")

    def realsense_callback(self, compressed_image_msg, depth_msg):
        # [수정] 종료 플래그를 확인하여 안전하게 처리
        if self.intrinsics is None or self._is_shutting_down:
            return
        
        try:
            # [수정] 이름이 변경된 스레드 풀 실행기를 사용
            self.yolo_executor.submit(self._process_realsense_data, compressed_image_msg, depth_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to submit realsense task: {e}")

    def usb_cam_callback(self, compressed_msg):
        # [수정] 종료 플래그를 확인하여 안전하게 처리
        if self._is_shutting_down:
            return
            
        try:
            # [수정] 이름이 변경된 스레드 풀 실행기를 사용
            self.yolo_executor.submit(self._process_usb_cam_data, compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to submit usb_cam task: {e}")

    def _process_realsense_data(self, compressed_image_msg, depth_msg):
        try:
            # ... (내부 로직은 이전과 동일)
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            if cv_color is None or cv_depth_raw is None:
                return
            
            cv2.resize(cv_depth_raw, (self.proc_width, self.proc_height), dst=self.resized_depth, interpolation=cv2.INTER_NEAREST)
            cv2.resize(cv_color, (self.proc_width, self.proc_height), dst=self.resized_color_yolo, interpolation=cv2.INTER_AREA)

            color_image_to_draw = cv_color.copy()
            supply_detected = self.run_supply_tracking(color_image_to_draw, self.resized_depth, self.resized_color_yolo)
            
            self.status_msg.data = supply_detected
            self.status_pub.publish(self.status_msg)
            
            self.publish_compressed_viz(self.realsense_viz_pub, color_image_to_draw)
        except Exception as e:
            self.get_logger().error(f"Error in Realsense worker: {e}\n{traceback.format_exc()}")
            
    def _process_usb_cam_data(self, compressed_msg):
        try:
            # ... (내부 로직은 이전과 동일)
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                return

            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            results_traffic = self.traffic_detection_model(annotated_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_traffic_detections(annotated_image, results_traffic)

            self.publish_compressed_viz(self.usb_cam_viz_pub, annotated_image)
        except Exception as e:
            self.get_logger().error(f"Error in USB Cam worker: {e}\n{traceback.format_exc()}")

    def run_supply_tracking(self, color_image_to_draw, resized_depth_image, yolo_input_image):
        # ... (내부 로직은 이전과 동일)
        if self.intrinsics is None: return False
        supply_detected_in_frame = False
        results = self.supply_model(yolo_input_image, verbose=False)
        for box in results[0].boxes:
            if int(box.cls) == 0:
                supply_detected_in_frame = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_res = (x1 + x2) // 2
                cy_res = (y1 + y2) // 2
                if 0 <= cy_res < self.proc_height and 0 <= cx_res < self.proc_width:
                    depth_in_mm = resized_depth_image[cy_res, cx_res]
                    if depth_in_mm > 0:
                        orig_cx = int(cx_res * self.intrinsics.width / self.proc_width)
                        orig_cy = int(cy_res * self.intrinsics.height / self.proc_height)
                        result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [orig_cx, orig_cy], depth_in_mm)
                        self.point_msg.x = float(result[2] / 1000.0)
                        self.point_msg.y = float(-result[0] / 1000.0)
                        self.point_msg.z = float(-result[1] / 1000.0)
                        self.distance_pub.publish(self.point_msg)
                        orig_x1 = int(x1 * self.intrinsics.width / self.proc_width)
                        orig_y1 = int(y1 * self.intrinsics.height / self.proc_height)
                        orig_x2 = int(x2 * self.intrinsics.width / self.proc_width)
                        orig_y2 = int(y2 * self.intrinsics.height / self.proc_height)
                        label = f"Supply: x={self.point_msg.x:.2f}m, y={self.point_msg.y:.2f}m, z={self.point_msg.z:.2f}m"
                        cv2.rectangle(color_image_to_draw, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 255), 2)
                        cv2.putText(color_image_to_draw, label, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return supply_detected_in_frame

    def publish_compressed_viz(self, publisher, cv_image):
        # ... (내부 로직은 이전과 동일)
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        success, encoded_image = cv2.imencode('.jpg', cv_image)
        if success:
            msg.data = encoded_image.tobytes()
            publisher.publish(msg)

    def draw_marker_detections(self, image, results):
        # ... (내부 로직은 이전과 동일)
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.marker_class_names[cls_id] if cls_id < len(self.marker_class_names) else "Unknown"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    def draw_traffic_detections(self, image, results):
        # ... (내부 로직은 이전과 동일)
        for result in results:
            for box in result.boxes.cpu().numpy():
                cls_id = int(box.cls[0])
                if cls_id < len(self.traffic_model_class_names):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = self.traffic_model_class_names[cls_id]
                    color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image
        
    def destroy_node(self):
        self.get_logger().info("Shutting down the thread pool.")
        # [수정] 종료 플래그를 설정하고, 이름이 변경된 실행기를 종료
        self._is_shutting_down = True
        self.yolo_executor.shutdown(wait=True)
        super().destroy_node()

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