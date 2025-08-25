#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: onnx_traffic.py
# DESCRIPTION:
# 1. [Hinton's ONNX Fix] PyTorch(.pt) 가중치를 ONNX(.onnx)로 변경하여 추론 가속.
# 2. [Hinton's Service Fix] /supply_distance 토픽 발행 로직을 PickPlace 서비스 클라이언트로 대체.
# 3. [Hinton's Reliability Fix] 연속성 및 거리 필터를 추가하여 오인식된 객체에 대한 서비스 요청 방지.

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
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge

from mtc_interfaces.srv import PickPlace

import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_traffic_node')
        self.get_logger().info("--- YOLO Vision Node (Hinton's ONNX Architecture with Reliability Filter) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"PyTorch detected device: {self.device}. ONNX Runtime will use best provider.")

        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value

        # [Hinton's Reliability Fix] 오인식 방지를 위한 파라미터 선언
        self.declare_parameter('detection_threshold', 5) # 연속 감지 횟수 조건
        self.declare_parameter('max_distance', 2.0)      # 최대 인식 거리 (미터)
        self.declare_parameter('min_distance', 0.3)      # 최소 인식 거리 (미터)
        self.declare_parameter('tracking_tolerance', 0.2)  # 동일 객체로 판단할 거리 허용치 (미터)
        
        self.DETECTION_THRESHOLD = self.get_parameter('detection_threshold').get_parameter_value().integer_value
        self.MAX_DISTANCE = self.get_parameter('max_distance').get_parameter_value().double_value
        self.MIN_DISTANCE = self.get_parameter('min_distance').get_parameter_value().double_value
        self.TRACKING_TOLERANCE = self.get_parameter('tracking_tolerance').get_parameter_value().double_value
        
        # [Hinton's Reliability Fix] 연속 감지 카운터 및 마지막 위치 저장 변수
        self.detection_counter = 0
        self.last_detected_position = None

        try:
            self.declare_parameter('supply_model_path', './tracking.onnx')
            self.declare_parameter('marker_model_path', './vision_enemy2.onnx')
            self.declare_parameter('traffic_model_path', './traffic_light.onnx')
            supply_model_path = self.get_parameter('supply_model_path').get_parameter_value().string_value
            marker_model_path = self.get_parameter('marker_model_path').get_parameter_value().string_value
            traffic_model_path = self.get_parameter('traffic_model_path').get_parameter_value().string_value
            self.supply_model = YOLO(supply_model_path, task='detect')
            self.marker_model = YOLO(marker_model_path, task='detect')
            self.traffic_detection_model = YOLO(traffic_model_path, task='detect')
            self.marker_class_names = ['A', 'E', 'Enemy', 'Heart', 'K', 'M', 'O', 'R', 'ROKA', 'Y']
            self.traffic_model_class_names = ['red', 'green']
            self.get_logger().info("✅ All ONNX models loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO models: {e}")
            self.destroy_node(); return

        self.intrinsics = None
        self.camera_info_sub = None

        self.pick_place_client = self.create_client(PickPlace, 'pick_place')
        while not self.pick_place_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('pick_place service not available, waiting...')
        self.service_call_in_progress = False

        self.status_pub = self.create_publisher(Bool, '/supply_status', 10)
        self.realsense_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/realsense/viz/compressed', 1)
        self.usb_cam_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam/viz/compressed', 1)
        self.led_pub = self.create_publisher(String, '/led_control', 10)
        self.traffic_pub = self.create_publisher(String, '/traffic_command', 10)
        self.resized_color_yolo = np.empty((self.proc_height, self.proc_width, 3), dtype=np.uint8)
        self.resized_depth = np.empty((self.proc_height, self.proc_width), dtype=np.uint16)
        self.status_msg = Bool()
        self.yolo_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='yolo_worker')
        self._is_shutting_down = False
        info_topic = "/camera/color/camera_info"
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, 10)
        self.get_logger().info(f"Waiting for CameraInfo on topic: {info_topic}")
        usb_cam_topic = 'camera1/image_compressed'
        self.usb_cam_sub = self.create_subscription(CompressedImage, usb_cam_topic, self.usb_cam_callback, 1)
        
    def camera_info_callback(self, info_msg):
        if self.intrinsics is not None: return
        self.get_logger().info("✅ CameraInfo received.")
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = info_msg.width; self.intrinsics.height = info_msg.height
        self.intrinsics.ppx = info_msg.k[2]; self.intrinsics.ppy = info_msg.k[5]
        self.intrinsics.fx = info_msg.k[0]; self.intrinsics.fy = info_msg.k[4]
        if info_msg.distortion_model == 'plumb_bob': self.intrinsics.model = rs2.distortion.brown_conrady
        elif info_msg.distortion_model == 'equidistant': self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in info_msg.d]
        self.initialize_image_sync()
        if self.camera_info_sub: self.destroy_subscription(self.camera_info_sub); self.camera_info_sub = None
        self.get_logger().info("CameraInfo subscription destroyed. Starting image synchronization.")

    def initialize_image_sync(self):
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub], queue_size=5, slop=0.2)
        self.ts.registerCallback(self.realsense_callback)
        self.get_logger().info("✅ YOLO Vision Node initialized successfully.")

    def realsense_callback(self, compressed_image_msg, depth_msg):
        if self.intrinsics is None or self._is_shutting_down: return
        try: self.yolo_executor.submit(self._process_realsense_data, compressed_image_msg, depth_msg)
        except Exception as e: self.get_logger().error(f"Failed to submit realsense task: {e}")
    
    def usb_cam_callback(self, compressed_msg):
        if self._is_shutting_down: return
        try: self.yolo_executor.submit(self._process_usb_cam_data, compressed_msg)
        except Exception as e: self.get_logger().error(f"Failed to submit usb_cam task: {e}")
            
    def _process_realsense_data(self, compressed_image_msg, depth_msg):
        try:
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            if cv_color is None or cv_depth_raw is None: self.get_logger().warn("Failed to decompress. Skip."); return
            cv2.resize(cv_depth_raw, (self.proc_width, self.proc_height), dst=self.resized_depth, interpolation=cv2.INTER_NEAREST)
            cv2.resize(cv_color, (self.proc_width, self.proc_height), dst=self.resized_color_yolo, interpolation=cv2.INTER_AREA)
            color_image_to_draw = cv_color.copy()
            supply_detected = self.run_supply_tracking(color_image_to_draw, self.resized_depth, self.resized_color_yolo)
            self.status_msg.data = supply_detected; self.status_pub.publish(self.status_msg)
            self.publish_compressed_viz(self.realsense_viz_pub, color_image_to_draw)
        except Exception as e: self.get_logger().error(f"Error in Realsense worker: {e}\n{traceback.format_exc()}")
            
    def _process_usb_cam_data(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None: self.get_logger().warn("Failed to decompress USB cam image."); return
            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            roka_found, enemy_found = False, False
            for r in results_marker:
                for box in r.boxes.cpu().numpy():
                    label = self.marker_class_names[int(box.cls[0])]
                    if label == 'ROKA': roka_found = True
                    elif label == 'Enemy': enemy_found = True
            led_msg = String(); led_msg.data = "ROKA" if roka_found else "ENEMY" if enemy_found else "NONE"
            self.led_pub.publish(led_msg)
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            results_traffic = self.traffic_detection_model(annotated_image, conf=0.5, iou=0.45, verbose=False)
            red_found, green_found = False, False
            for r in results_traffic:
                for box in r.boxes.cpu().numpy():
                    label = self.traffic_model_class_names[int(box.cls[0])]
                    if label == 'red': red_found = True
                    elif label == 'green': green_found = True
            traffic_msg = String()
            if red_found: traffic_msg.data = "stop"; self.traffic_pub.publish(traffic_msg)
            elif green_found: traffic_msg.data = "go"; self.traffic_pub.publish(traffic_msg)
            annotated_image = self.draw_traffic_detections(annotated_image, results_traffic)
            self.publish_compressed_viz(self.usb_cam_viz_pub, annotated_image)
        except Exception as e: self.get_logger().error(f"Error in USB Cam worker: {e}\n{traceback.format_exc()}")

    def pick_place_response_callback(self, future):
        try:
            response = future.result()
            if response.success: self.get_logger().info(f"✅ PickPlace service call successful: {response.message}")
            else: self.get_logger().warn(f"⚠️ PickPlace service call failed: {response.message}")
        except Exception as e: self.get_logger().error(f"Service call failed with exception: {e}")
        finally: self.service_call_in_progress = False

    def run_supply_tracking(self, color_image_to_draw, resized_depth_image, yolo_input_image):
        if self.intrinsics is None: return False
        results = self.supply_model(yolo_input_image, verbose=False)
        supply_found_this_frame = False; current_position = None
        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0]); cx_res, cy_res = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= cy_res < self.proc_height and 0 <= cx_res < self.proc_width:
                    depth_in_mm = resized_depth_image[cy_res, cx_res]
                    if depth_in_mm > 0:
                        supply_found_this_frame = True
                        orig_cx, orig_cy = int(cx_res * self.intrinsics.width / self.proc_width), int(cy_res * self.intrinsics.height / self.proc_height)
                        deprojected = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [orig_cx, orig_cy], depth_in_mm)
                        x, y, z = float(deprojected[2]/1000.0), float(-deprojected[0]/1000.0), float(-deprojected[1]/1000.0)
                        current_position = np.array([x, y, z])
                        label = f"Supply: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m"
                        orig_x1, orig_y1 = int(x1 * self.intrinsics.width / self.proc_width), int(y1 * self.intrinsics.height / self.proc_height)
                        orig_x2, orig_y2 = int(x2 * self.intrinsics.width / self.proc_width), int(y2 * self.intrinsics.height / self.proc_height)
                        cv2.rectangle(color_image_to_draw, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 255), 2)
                        cv2.putText(color_image_to_draw, label, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        break
        if supply_found_this_frame:
            if self.last_detected_position is not None and np.linalg.norm(current_position - self.last_detected_position) < self.TRACKING_TOLERANCE:
                self.detection_counter += 1
            else: self.detection_counter = 1
            self.last_detected_position = current_position
            if self.detection_counter >= self.DETECTION_THRESHOLD:
                distance = np.linalg.norm(current_position)
                if self.MIN_DISTANCE <= distance <= self.MAX_DISTANCE:
                    if not self.service_call_in_progress:
                        self.service_call_in_progress = True; request = PickPlace.Request()
                        request.x, request.y, request.z = current_position[0], current_position[1], current_position[2]
                        self.get_logger().info(f"Requesting PickPlace service for stable target at {distance:.2f}m.")
                        future = self.pick_place_client.call_async(request)
                        future.add_done_callback(self.pick_place_response_callback)
                else: self.get_logger().info(f"Stable target detected, but out of range ({distance:.2f}m).")
            else: self.get_logger().info(f"Tracking target... continuity: {self.detection_counter}/{self.DETECTION_THRESHOLD}")
        else: self.detection_counter = 0; self.last_detected_position = None
        return supply_found_this_frame

    def publish_compressed_viz(self, publisher, cv_image):
        msg = CompressedImage(); msg.header.stamp = self.get_clock().now().to_msg(); msg.format = "jpeg"
        success, encoded_image = cv2.imencode('.jpg', cv_image)
        if success: msg.data = encoded_image.tobytes(); publisher.publish(msg)

    def draw_marker_detections(self, image, results):
        for r in results:
            for box in r.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0]); conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.marker_class_names[cls_id] if cls_id < len(self.marker_class_names) else "Unknown"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    def draw_traffic_detections(self, image, results):
        for r in results:
            for box in r.boxes.cpu().numpy():
                cls_id = int(box.cls[0])
                if cls_id < len(self.traffic_model_class_names):
                    x1, y1, x2, y2 = map(int, box.xyxy[0]); conf = box.conf[0]
                    label = self.traffic_model_class_names[cls_id]
                    color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image

    def destroy_node(self):
        self.get_logger().info("Shutting down the thread pool.")
        self._is_shutting_down = True; self.yolo_executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args); node = YoloVisionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()