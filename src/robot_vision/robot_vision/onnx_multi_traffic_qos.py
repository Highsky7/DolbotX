#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: onnx_multi_traffic_qos_optimized.py
# AUTHOR: Guido (Optimized for Real-time Distributed Systems)
# DESCRIPTION:
# 1. [Hinton's ONNX Fix] PyTorch(.pt) 가중치를 ONNX(.onnx)로 변경하여 추론 가속.
# 2. [Hinton's Service Fix] /supply_distance 토픽 발행 로직을 PickPlace 서비스 클라이언트로 대체.
# 3. [Hinton's Reliability Fix] 연속성 및 거리 필터를 추가하여 오인식된 객체에 대한 서비스 요청 방지.
# 4. [Hinton's Dual Vision Fix] USB 카메라 입력 소스를 camera1과 camera2로 확장하여 인식 범위 및 신뢰성 향상.
# 5. [Guido's Real-time Fix] CameraInfo QoS를 VOLATILE로 수정하여 통신 안정성 확보.
# 6. [Guido's Latency Fix] 처리 지연을 막기 위해 잠금(Lock) 기반의 프레임 드롭(Frame Drop) 전략 적용.
# 7. [User's Refinement] 신호등 탐지 로직에 파라미터 기반의 크기, 위치, 시간적 일관성 필터 적용.
# 8. [Guido's Final Touch] FP16 추론 가속 적용 및 코드 간소화.

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

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node_guido_optimized')
        self.get_logger().info("--- YOLO Vision Node (Guido's Optimized Real-time Architecture) ---")
        
        self.realsense_lock = threading.Lock()
        self.usb_cam_locks = {'cam1': threading.Lock(), 'cam2': threading.Lock()}

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"PyTorch detected device: {self.device}. ONNX Runtime will use the best available provider.")
        
        # GPU 사용 시 반정밀도(FP16) 추론 활성화
        self.use_half = self.device == 'cuda'
        # self.use_half = False # 반정밀도 비활성화
        if self.use_half:
            self.get_logger().info("FP16 inference is enabled for CUDA device.")

        self.qos_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.qos_reliable_default = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.qos_camera_info = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE)
        
        # --- [사용자 요청] 신호등 탐지 파라미터 추가 ---
        self.declare_parameter('red_light_min_area', 100)
        self.declare_parameter('traffic_roi_top_ratio', 0.6)
        self.declare_parameter('red_light_confirmation_frames', 3)
        self.declare_parameter('red_light_tracking_tolerance', 50) # 픽셀 단위

        self.RED_LIGHT_MIN_AREA = self.get_parameter('red_light_min_area').get_parameter_value().integer_value
        self.TRAFFIC_ROI_TOP_RATIO = self.get_parameter('traffic_roi_top_ratio').get_parameter_value().double_value
        self.RED_LIGHT_CONFIRMATION_FRAMES = self.get_parameter('red_light_confirmation_frames').get_parameter_value().integer_value
        self.RED_LIGHT_TRACKING_TOLERANCE = self.get_parameter('red_light_tracking_tolerance').get_parameter_value().integer_value

        self.red_light_tracker = {
            'cam1': {'counter': 0, 'last_center': None},
            'cam2': {'counter': 0, 'last_center': None}
        }
        # --- 파라미터 추가 끝 ---

        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value
        
        self.declare_parameter('detection_threshold', 5)
        self.declare_parameter('max_distance', 2.0)
        self.declare_parameter('min_distance', 0.3)
        self.declare_parameter('tracking_tolerance', 0.2)
        
        self.DETECTION_THRESHOLD = self.get_parameter('detection_threshold').get_parameter_value().integer_value
        self.MAX_DISTANCE = self.get_parameter('max_distance').get_parameter_value().double_value
        self.MIN_DISTANCE = self.get_parameter('min_distance').get_parameter_value().double_value
        self.TRACKING_TOLERANCE = self.get_parameter('tracking_tolerance').get_parameter_value().double_value
        
        self.detection_counter = 0
        self.last_detected_position = None

        try:
            self.declare_parameter('supply_model_path', './tracking.onnx')
            self.declare_parameter('marker_model_path', './vision_enemy3.onnx')
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

        self.pick_place_client = self.create_client(PickPlace, 'pick_place')
        while not self.pick_place_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('pick_place service not available, waiting...')
        self.service_call_in_progress = False

        self.status_pub = self.create_publisher(Bool, '/supply_status', self.qos_reliable_default)
        self.realsense_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/realsense/viz/compressed', self.qos_sensor_data)
        self.led_pub = self.create_publisher(String, '/led_control', self.qos_reliable_default)
        self.traffic_pub = self.create_publisher(String, '/traffic_command', self.qos_reliable_default)
        self.usb_cam1_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam1/viz/compressed', self.qos_sensor_data)
        self.usb_cam2_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam2/viz/compressed', self.qos_sensor_data)

        self.resized_color_yolo = np.empty((self.proc_height, self.proc_width, 3), dtype=np.uint8)
        self.resized_depth = np.empty((self.proc_height, self.proc_width), dtype=np.uint16)
        
        self.yolo_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='yolo_worker')
        self._is_shutting_down = False

        info_topic = "/camera/color/camera_info"
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, self.qos_camera_info)
        self.get_logger().info(f"Waiting for CameraInfo on topic: {info_topic}")
        
        usb_cam1_topic = 'camera1/image_raw/compressed'
        self.usb_cam1_sub = self.create_subscription(
            CompressedImage, usb_cam1_topic, lambda msg: self.usb_cam_callback(msg, 'cam1'), qos_profile=self.qos_sensor_data)
        self.get_logger().info(f"Subscribing to USB Camera 1 on topic: {usb_cam1_topic}")
        
        usb_cam2_topic = 'camera2/image_raw/compressed'
        self.usb_cam2_sub = self.create_subscription(
            CompressedImage, usb_cam2_topic, lambda msg: self.usb_cam_callback(msg, 'cam2'), qos_profile=self.qos_sensor_data)
        self.get_logger().info(f"Subscribing to USB Camera 2 on topic: {usb_cam2_topic}")
        
    def camera_info_callback(self, info_msg):
        if self.intrinsics is not None: return
        self.get_logger().info("✅ CameraInfo received.")
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = info_msg.width; self.intrinsics.height = info_msg.height
        self.intrinsics.ppx = info_msg.k[2]; self.intrinsics.ppy = info_msg.k[5]
        self.intrinsics.fx = info_msg.k[0]; self.intrinsics.fy = info_msg.k[4]
        self.intrinsics.model = rs2.distortion.brown_conrady if info_msg.distortion_model == 'plumb_bob' else rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in info_msg.d]
        self.initialize_image_sync()
        if self.camera_info_sub: self.destroy_subscription(self.camera_info_sub); self.camera_info_sub = None
        self.get_logger().info("CameraInfo subscription destroyed. Starting image synchronization.")

    def initialize_image_sync(self):
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed', qos_profile=self.qos_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, "/camera/aligned_depth_to_color/image_raw", qos_profile=self.qos_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub], queue_size=5, slop=0.2)
        self.ts.registerCallback(self.realsense_callback)
        self.get_logger().info("✅ YOLO Vision Node initialized successfully.")

    def realsense_callback(self, compressed_image_msg, depth_msg):
        if self.intrinsics is None or self._is_shutting_down: return
        if self.realsense_lock.acquire(blocking=False):
            try:
                self.yolo_executor.submit(self._process_realsense_data, compressed_image_msg, depth_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to submit realsense task: {e}")
                self.realsense_lock.release()
        else:
            self.get_logger().warn("Dropping a Realsense frame, previous frame still processing.", throttle_duration_sec=1)

    def usb_cam_callback(self, compressed_msg, camera_id):
        if self._is_shutting_down: return
        lock = self.usb_cam_locks[camera_id]
        if lock.acquire(blocking=False):
            try:
                self.yolo_executor.submit(self._process_usb_cam_data, compressed_msg, camera_id)
            except Exception as e:
                self.get_logger().error(f"Failed to submit usb_cam task for {camera_id}: {e}")
                lock.release()
        else:
            self.get_logger().warn(f"Dropping a frame from {camera_id}, previous frame still processing.", throttle_duration_sec=1)

    def _process_realsense_data(self, compressed_image_msg, depth_msg):
        try:
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            if cv_color is None or cv_depth_raw is None: return
            
            cv2.resize(cv_depth_raw, (self.proc_width, self.proc_height), dst=self.resized_depth, interpolation=cv2.INTER_NEAREST)
            cv2.resize(cv_color, (self.proc_width, self.proc_height), dst=self.resized_color_yolo, interpolation=cv2.INTER_AREA)
            
            color_image_to_draw = cv_color.copy()
            supply_detected = self.run_supply_tracking(color_image_to_draw, self.resized_depth, self.resized_color_yolo)
            
            self.status_pub.publish(Bool(data=supply_detected))
            self.publish_compressed_viz(self.realsense_viz_pub, color_image_to_draw)
        except Exception as e:
            self.get_logger().error(f"Error in Realsense worker: {e}\n{traceback.format_exc()}")
        finally:
            self.realsense_lock.release()
            
    def _process_usb_cam_data(self, compressed_msg, camera_id):
        lock = self.usb_cam_locks[camera_id]
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None: return
            
            h, w, _ = cv_image.shape
            tracker = self.red_light_tracker[camera_id]

            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False, half=self.use_half)
            roka_found, enemy_found = False, False
            for r in results_marker:
                for box in r.boxes.cpu().numpy():
                    label = self.marker_class_names[int(box.cls[0])]
                    if label == 'ROKA': roka_found = True
                    elif label == 'Enemy': enemy_found = True
            
            led_data = "ROKA" if roka_found else "ENEMY" if enemy_found else "NONE"
            self.led_pub.publish(String(data=led_data))
            
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            
            results_traffic = self.traffic_detection_model(annotated_image, conf=0.5, iou=0.45, verbose=False, half=self.use_half)
            red_found, green_found = False, False
            
            best_red_candidate_center = None

            for r in results_traffic:
                for box_data in r.boxes.cpu().numpy():
                    label = self.traffic_model_class_names[int(box_data.cls[0])]
                    
                    if label == 'green':
                        green_found = True
                        continue

                    if label == 'red':
                        if box_data.conf[0] < 0.7: continue

                        box = box_data.xyxy[0].astype(int)
                        box_w = box[2] - box[0]; box_h = box[3] - box[1]

                        if (box_w * box_h) < self.RED_LIGHT_MIN_AREA: continue
                        
                        cy = (box[1] + box[3]) / 2
                        if cy > h * self.TRAFFIC_ROI_TOP_RATIO: continue
                        
                        current_center = np.array([(box[0] + box[2]) / 2, cy])
                        if best_red_candidate_center is None:
                             best_red_candidate_center = current_center
            
            if best_red_candidate_center is not None:
                if tracker['last_center'] is not None and \
                   np.linalg.norm(best_red_candidate_center - tracker['last_center']) < self.RED_LIGHT_TRACKING_TOLERANCE:
                    tracker['counter'] += 1
                else:
                    tracker['counter'] = 1
                tracker['last_center'] = best_red_candidate_center
            else:
                tracker['counter'] = 0
                tracker['last_center'] = None

            if tracker['counter'] >= self.RED_LIGHT_CONFIRMATION_FRAMES:
                red_found = True
            
            if red_found: 
                self.traffic_pub.publish(String(data="stop"))
            elif green_found: 
                self.traffic_pub.publish(String(data="go"))
            
            annotated_image = self.draw_traffic_detections(annotated_image, results_traffic)
            
            viz_publisher = self.usb_cam1_viz_pub if camera_id == 'cam1' else self.usb_cam2_viz_pub
            self.publish_compressed_viz(viz_publisher, annotated_image)
            
        except Exception as e:
            self.get_logger().error(f"Error in USB Cam worker ({camera_id}): {e}\n{traceback.format_exc()}")
        finally:
            lock.release()

    def pick_place_response_callback(self, future):
        try:
            response = future.result()
            if response.success: self.get_logger().info(f"✅ PickPlace service call successful: {response.message}")
            else: self.get_logger().warn(f"⚠️ PickPlace service call failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed with exception: {e}")
        finally:
            self.service_call_in_progress = False
            
    def run_supply_tracking(self, color_image_to_draw, resized_depth_image, yolo_input_image):
        if self.intrinsics is None: return False
        
        results = self.supply_model(yolo_input_image, verbose=False, half=self.use_half)
        supply_found_this_frame = False
        current_position = None
        
        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_res, cy_res = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= cy_res < self.proc_height and 0 <= cx_res < self.proc_width:
                    depth_in_mm = resized_depth_image[cy_res, cx_res]
                    if depth_in_mm > 0:
                        supply_found_this_frame = True
                        orig_cx = int(cx_res * self.intrinsics.width / self.proc_width)
                        orig_cy = int(cy_res * self.intrinsics.height / self.proc_height)
                        deprojected = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [orig_cx, orig_cy], depth_in_mm)
                        
                        x_coord = float(deprojected[2] / 1000.0)
                        y_coord = float(-deprojected[0] / 1000.0)
                        z_coord = float(-deprojected[1] / 1000.0)
                        current_position = np.array([x_coord, y_coord, z_coord])
                        
                        label = f"Supply: x={x_coord:.2f}m, y={y_coord:.2f}m, z={z_coord:.2f}m"

                        # --- 💡 수정된 부분 시작 💡 ---
                        # 원본 해상도에 맞게 좌표를 정확하게 스케일링합니다.
                        scale_w = self.intrinsics.width / self.proc_width
                        scale_h = self.intrinsics.height / self.proc_height
                        orig_x1 = int(x1 * scale_w)
                        orig_y1 = int(y1 * scale_h)
                        orig_x2 = int(x2 * scale_w)
                        orig_y2 = int(y2 * scale_h)
                        # --- 💡 수정된 부분 끝 💡 ---
                        
                        cv2.rectangle(color_image_to_draw, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 255), 2)
                        cv2.putText(color_image_to_draw, label, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        break

        if supply_found_this_frame:
            if self.last_detected_position is not None and \
            np.linalg.norm(current_position - self.last_detected_position) < self.TRACKING_TOLERANCE:
                self.detection_counter += 1
            else:
                self.detection_counter = 1
            self.last_detected_position = current_position

            if self.detection_counter >= self.DETECTION_THRESHOLD:
                distance = np.linalg.norm(current_position)
                if self.MIN_DISTANCE <= distance <= self.MAX_DISTANCE and not self.service_call_in_progress:
                    self.service_call_in_progress = True
                    request = PickPlace.Request()
                    request.x, request.y, request.z = current_position.tolist()
                    self.get_logger().info(f"Requesting PickPlace service for stable target at {distance:.2f}m.")
                    future = self.pick_place_client.call_async(request)
                    future.add_done_callback(self.pick_place_response_callback)
                elif not (self.MIN_DISTANCE <= distance <= self.MAX_DISTANCE):
                    self.get_logger().debug(f"Stable target detected, but out of range ({distance:.2f}m).")
            else:
                self.get_logger().debug(f"Tracking target... continuity: {self.detection_counter}/{self.DETECTION_THRESHOLD}")
        else:
            self.detection_counter = 0
            self.last_detected_position = None
            
        return supply_found_this_frame

    def publish_compressed_viz(self, publisher, cv_image):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
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
        self._is_shutting_down = True
        self.yolo_executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()