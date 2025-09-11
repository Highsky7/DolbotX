#!/usr/env/bin python3
# -*- coding: utf-8 -*-
# FILE: vision_nofilter.py
# AUTHOR: Guido (Optimized for Real-time Distributed Systems)
# REFINED BY: Hinton (for Robust State Integration & Logic Correction)
# DESCRIPTION:
# ... (설명 생략) ...
# 15. [Hinton's Update V3 for Robust Traffic Light Detection]
#     - 'red_light_tracker'를 단순 카운터에서 상태 기계(State Machine)로 변경.
#     - 'STATE_CLEAR', 'STATE_CANDIDATE', 'STATE_CONFIRMED_RED' 세 가지 상태를 정의하여
#       신호등 인식의 불확실성을 체계적으로 관리.
#     - 이력(Hysteresis) 원리를 도입: 'CONFIRMED_RED' 상태 진입 조건(N 프레임)과
#       해제 조건(M 프레임)을 비대칭적으로 설계하여 '깜빡임(flickering)' 현상을 원천적으로 방지.
#       이를 위해 'red_light_loss_tolerance_frames' 파라미터 추가.
#
# 16. [Hinton's Update V5 for Single-source Traffic Logic]
#     - 'traffic_light_camera_id' 파라미터를 추가하여 신호등 인식 전담 카메라를 지정.
#     - 지정된 카메라만 신호등 탐지 모델을 실행하고, 상태 기계를 업데이트하며, 명령을 발행하도록 로직 수정.
#     - 이를 통해 여러 카메라로부터의 명령어 충돌을 방지하고 시스템 리소스를 최적화.
#
# MODIFICATION 1: Removed temporal detection filter for immediate supply command issuance.
# MODIFICATION 2: Added a 3-second lock for the 'stop' command to ensure stable robot halt.

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

# ## MODIFICATION START: Duration import for timer ##
from rclpy.duration import Duration
# ## MODIFICATION END ##

from mtc_interfaces.srv import PickPlace

import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_traffic_node_no_qos_optimized')
        self.get_logger().info("--- YOLO Vision Node (Guido's Optimized, Hinton's Refinement V5) ---")
        
        self.realsense_lock = threading.Lock()
        self.usb_cam_locks = {'cam1': threading.Lock(), 'cam2': threading.Lock()}
        
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"PyTorch detected device: {self.device}. ONNX Runtime will use best provider.")

        self.use_half = self.device == 'cuda'
        if self.use_half:
            self.get_logger().info("FP16 inference is enabled for CUDA device.")

        # ## HINTON'S MODIFICATION START: State Machine Parameters for Traffic Light ##
        self.declare_parameter('red_light_min_area', 10000)
        self.declare_parameter('traffic_roi_top_ratio', 0.4)
        self.declare_parameter('red_light_confirmation_frames', 5) # N 값: RED 상태 확신 프레임
        self.declare_parameter('red_light_loss_tolerance_frames', 10) # M 값: RED 상태 해제 관용 프레임
        self.declare_parameter('red_light_tracking_tolerance', 50)
        
        # ## HINTON'S V5 UPDATE START: Designate a single camera for traffic light detection ##
        self.declare_parameter('traffic_light_camera_id', 'cam1') # 신호등 인식에 사용할 카메라 ID ('cam1' or 'cam2')
        # ## HINTON'S V5 UPDATE END ##

        self.RED_LIGHT_MIN_AREA = self.get_parameter('red_light_min_area').get_parameter_value().integer_value
        self.TRAFFIC_ROI_TOP_RATIO = self.get_parameter('traffic_roi_top_ratio').get_parameter_value().double_value
        self.RED_LIGHT_CONFIRMATION_FRAMES = self.get_parameter('red_light_confirmation_frames').get_parameter_value().integer_value
        self.RED_LIGHT_LOSS_TOLERANCE_FRAMES = self.get_parameter('red_light_loss_tolerance_frames').get_parameter_value().integer_value
        self.RED_LIGHT_TRACKING_TOLERANCE = self.get_parameter('red_light_tracking_tolerance').get_parameter_value().integer_value
        
        # ## HINTON'S V5 UPDATE START ##
        self.TRAFFIC_LIGHT_CAMERA_ID = self.get_parameter('traffic_light_camera_id').get_parameter_value().string_value
        self.get_logger().info(f"Primary camera for traffic light detection set to: '{self.TRAFFIC_LIGHT_CAMERA_ID}'")
        # ## HINTON'S V5 UPDATE END ##

        # 상태를 나타내는 상수 정의
        self.STATE_CLEAR = 0
        self.STATE_CANDIDATE = 1
        self.STATE_CONFIRMED_RED = 2

        # 각 카메라별 상태 추적기 초기화
        self.red_light_tracker = {
            'cam1': {'state': self.STATE_CLEAR, 'confirmation_counter': 0, 'loss_counter': 0, 'last_center': None},
            'cam2': {'state': self.STATE_CLEAR, 'confirmation_counter': 0, 'loss_counter': 0, 'last_center': None}
        }
        # ## HINTON'S MODIFICATION END ##

        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value

        try:
            self.declare_parameter('supply_model_path', './tracking2.onnx')
            self.declare_parameter('marker_model_path', './vision_enemy3.onnx')
            self.declare_parameter('traffic_model_path', './traffic_robo.onnx')
            supply_model_path = self.get_parameter('supply_model_path').get_parameter_value().string_value
            marker_model_path = self.get_parameter('marker_model_path').get_parameter_value().string_value
            traffic_model_path = self.get_parameter('traffic_model_path').get_parameter_value().string_value
            self.supply_model = YOLO(supply_model_path, task='detect')
            self.marker_model = YOLO(marker_model_path, task='detect')
            self.traffic_detection_model = YOLO(traffic_model_path, task='detect')
            self.marker_class_names = ['A', 'E', 'Enemy', 'Heart', 'K', 'M', 'O', 'R', 'ROKA', 'Y']
            self.traffic_model_class_names = ['green', 'red']
            self.get_logger().info("✅ All ONNX models loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO models: {e}")
            self.destroy_node(); return

        self.intrinsics = None
        self.camera_info_sub = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pick_place_client = self.create_client(PickPlace, '/pick_place_service')
        while not self.pick_place_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('pick_place service not available, waiting...')
        self.service_call_in_progress = False

        self.status_pub = self.create_publisher(Bool, '/supply_status', 10)
        self.realsense_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/realsense/viz/compressed', 10)
        self.led_pub = self.create_publisher(String, '/led_control', 10)
        self.traffic_pub = self.create_publisher(String, '/traffic_command', 10)
        self.supply_pub = self.create_publisher(String, '/supply_command', 10)
        self.usb_cam1_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam1/viz/compressed', 10)
        self.usb_cam2_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam2/viz/compressed', 10)

        self.resized_color_yolo = np.empty((self.proc_height, self.proc_width, 3), dtype=np.uint8)
        self.resized_depth = np.empty((self.proc_height, self.proc_width), dtype=np.uint16)

        self.yolo_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='yolo_worker')
        self._is_shutting_down = False
        
        self.stop_command_lock_until = None
        self.stop_command_duration = Duration(seconds=3.0)

        info_topic = "/camera/color/camera_info"
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, 10)
        self.get_logger().info(f"Waiting for CameraInfo on topic: {info_topic}")

        usb_cam1_topic = 'camera1/image_raw/compressed'
        self.usb_cam1_sub = self.create_subscription(
            CompressedImage, usb_cam1_topic, lambda msg: self.usb_cam_callback(msg, 'cam1'), 10)
        self.get_logger().info(f"Subscribing to USB Camera 1 on topic: {usb_cam1_topic}")
        
        usb_cam2_topic = 'camera2/image_raw/compressed'
        self.usb_cam2_sub = self.create_subscription(
            CompressedImage, usb_cam2_topic, lambda msg: self.usb_cam_callback(msg, 'cam2'), 10)
        self.get_logger().info(f"Subscribing to USB Camera 2 on topic: {usb_cam2_topic}")
        
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
            if cv_color is None or cv_depth_raw is None: self.get_logger().warn("Failed to decompress. Skip."); return
            
            cv2.resize(cv_depth_raw, (self.proc_width, self.proc_height), dst=self.resized_depth, interpolation=cv2.INTER_NEAREST)
            cv2.resize(cv_color, (self.proc_width, self.proc_height), dst=self.resized_color_yolo, interpolation=cv2.INTER_AREA)
            
            color_image_to_draw = cv_color.copy()
            supply_detected = self.run_supply_tracking(color_image_to_draw, self.resized_depth, self.resized_color_yolo, compressed_image_msg.header)
            self.status_pub.publish(Bool(data=supply_detected))
            self.publish_compressed_viz(self.realsense_viz_pub, color_image_to_draw)
        except Exception as e:
            self.get_logger().error(f"Error in Realsense worker: {e}\n{traceback.format_exc()}")
        finally:
            self.realsense_lock.release()
    
    # ## HINTON'S MODIFICATION START: Rewritten USB Camera Processor with State Machine & Single Source Logic ##
    def _process_usb_cam_data(self, compressed_msg, camera_id):
        lock = self.usb_cam_locks[camera_id]
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().warn(f"Failed to decompress USB cam image from {camera_id}.")
                return

            h, w, _ = cv_image.shape
            
            # --- 1. 마커 및 LED 처리 (모든 카메라에서 수행) ---
            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False, half=self.use_half)
            roka_found, enemy_found = False, False
            for r in results_marker:
                for box in r.boxes.cpu().numpy():
                    label = self.marker_class_names[int(box.cls[0])]
                    if label == 'ROKA': roka_found = True
                    elif label == 'Enemy': enemy_found = True
            
            led_data = "roka" if roka_found else "enemy" if enemy_found else "none"
            self.led_pub.publish(String(data=led_data))
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            
            # ## HINTON'S V5 UPDATE START: 지정된 카메라만 신호등 로직을 처리 ##
            if camera_id == self.TRAFFIC_LIGHT_CAMERA_ID:
                tracker = self.red_light_tracker[camera_id]

                # --- 2. 신호등 탐지 및 후보 선정 ---
                results_traffic = self.traffic_detection_model(annotated_image, conf=0.5, iou=0.45, verbose=False, half=self.use_half)
                
                green_found = False
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
                            
                            best_red_candidate_center = np.array([(box[0] + box[2]) / 2, cy])
                            break
                    if best_red_candidate_center is not None:
                        break
                
                # --- 3. 상태 기계(State Machine) 로직 ---
                red_light_detected_in_frame = False
                if best_red_candidate_center is not None:
                    if tracker['last_center'] is None or \
                       np.linalg.norm(best_red_candidate_center - tracker['last_center']) < self.RED_LIGHT_TRACKING_TOLERANCE:
                        red_light_detected_in_frame = True
                        tracker['last_center'] = best_red_candidate_center

                if tracker['state'] == self.STATE_CLEAR:
                    if red_light_detected_in_frame:
                        tracker['state'] = self.STATE_CANDIDATE
                        tracker['confirmation_counter'] = 1
                        tracker['loss_counter'] = 0
                        self.get_logger().debug(f"[{camera_id}] Red light candidate found. Entering CANDIDATE state.")
                
                elif tracker['state'] == self.STATE_CANDIDATE:
                    if red_light_detected_in_frame:
                        tracker['confirmation_counter'] += 1
                        if tracker['confirmation_counter'] >= self.RED_LIGHT_CONFIRMATION_FRAMES:
                            tracker['state'] = self.STATE_CONFIRMED_RED
                            tracker['loss_counter'] = 0
                            self.get_logger().info(f"[{camera_id}] Red light confirmed. Entering CONFIRMED_RED state.")
                    else:
                        tracker['state'] = self.STATE_CLEAR
                        tracker['confirmation_counter'] = 0
                        tracker['last_center'] = None
                        self.get_logger().debug(f"[{camera_id}] Candidate lost. Returning to CLEAR state.")

                elif tracker['state'] == self.STATE_CONFIRMED_RED:
                    if red_light_detected_in_frame:
                        tracker['loss_counter'] = 0
                    else:
                        tracker['loss_counter'] += 1
                        if tracker['loss_counter'] >= self.RED_LIGHT_LOSS_TOLERANCE_FRAMES:
                            tracker['state'] = self.STATE_CLEAR
                            tracker['confirmation_counter'] = 0
                            tracker['loss_counter'] = 0
                            tracker['last_center'] = None
                            self.get_logger().info(f"[{camera_id}] Red light lost for {self.RED_LIGHT_LOSS_TOLERANCE_FRAMES} frames. Returning to CLEAR state.")

                # --- 4. 최종 명령 발행 ---
                command_to_publish = None
                if tracker['state'] == self.STATE_CONFIRMED_RED:
                    command_to_publish = "stop"
                elif green_found:
                    command_to_publish = "go"

                if command_to_publish is not None:
                    self.traffic_pub.publish(String(data=command_to_publish))
                    
                # --- 5. 시각화 (신호등) ---
                annotated_image = self.draw_traffic_detections(annotated_image, results_traffic)
            # ## HINTON'S V5 UPDATE END ##
            
            # --- 6. 최종 이미지 발행 (모든 카메라 공통) ---
            viz_publisher = self.usb_cam1_viz_pub if camera_id == 'cam1' else self.usb_cam2_viz_pub
            self.publish_compressed_viz(viz_publisher, annotated_image)

        except Exception as e:
            self.get_logger().error(f"Error in USB Cam worker ({camera_id}): {e}\n{traceback.format_exc()}")
        finally:
            lock.release()
    # ## HINTON'S MODIFICATION END ##

    def pick_place_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"✅ PickPlace service successful: {response.message}. Publishing 'go'.")
                self.supply_pub.publish(String(data='go'))
            else:
                self.get_logger().warn(f"⚠️ PickPlace service failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed with exception: {e}")
        finally:
            self.service_call_in_progress = False
    
    def run_supply_tracking(self, color_image_to_draw, resized_depth_image, yolo_input_image, header):
        if self.intrinsics is None:
            return False
        
        results = self.supply_model(yolo_input_image, verbose=False, half=self.use_half)
        supply_found_this_frame = False
        transformed_position = None

        best_box = None
        max_conf = 0.0
        for box in results[0].boxes:
            if int(box.cls) == 0 and box.conf[0] > max_conf:
                max_conf = box.conf[0]
                best_box = box

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cx_res, cy_res = (x1 + x2) // 2, (y1 + y2) // 2
            
            if 0 <= cy_res < self.proc_height and 0 <= cx_res < self.proc_width:
                depth_in_mm = resized_depth_image[cy_res, cx_res]
                
                if depth_in_mm > 0:
                    supply_found_this_frame = True
                    orig_cx = int(cx_res * self.intrinsics.width / self.proc_width)
                    orig_cy = int(cy_res * self.intrinsics.height / self.proc_height)
                    deprojected = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [orig_cx, orig_cy], depth_in_mm)
                    optical_frame_coords = np.array([p / 1000.0 for p in deprojected])
                    
                    point_in_optical_frame = PointStamped()
                    point_in_optical_frame.header.frame_id = "camera_color_optical_frame"
                    point_in_optical_frame.header.stamp = header.stamp
                    point_in_optical_frame.point.x, point_in_optical_frame.point.y, point_in_optical_frame.point.z = optical_frame_coords
                    
                    target_frame = "camera_bottom_screw_frame"
                    
                    try:
                        transform = self.tf_buffer.lookup_transform(target_frame, point_in_optical_frame.header.frame_id, rclpy.time.Time())
                        point_in_target_frame = do_transform_point(point_in_optical_frame, transform)
                        transformed_position = np.array([point_in_target_frame.point.x, point_in_target_frame.point.y, point_in_target_frame.point.z])
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        self.get_logger().warn(f"Coordinate transform failed: {e}", throttle_duration_sec=5.0)
                        return False

                    label = f"x={transformed_position[0]:.2f}m, y={transformed_position[1]:.2f}m, z={transformed_position[2]:.2f}m"
                    scale_w = self.intrinsics.width / self.proc_width
                    scale_h = self.intrinsics.height / self.proc_height
                    orig_x1, orig_y1 = int(x1 * scale_w), int(y1 * scale_h)
                    orig_x2, orig_y2 = int(x2 * scale_w), int(y2 * scale_h)
                    cv2.rectangle(color_image_to_draw, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 255), 2)
                    cv2.putText(color_image_to_draw, label, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        current_time = self.get_clock().now()
        if self.stop_command_lock_until is not None and current_time >= self.stop_command_lock_until:
            self.get_logger().info("Stop command lock has expired.")
            self.stop_command_lock_until = None

        if self.stop_command_lock_until is not None:
            self.get_logger().debug("Stop command is locked. Publishing 'stop'.")
            self.supply_pub.publish(String(data='stop'))
            return supply_found_this_frame

        is_in_stop_zone = (supply_found_this_frame and 
                           transformed_position is not None and 
                           -0.1 <= transformed_position[2] <= 0.4)

        if is_in_stop_zone and not self.service_call_in_progress:
            z_coord = transformed_position[2]
            self.get_logger().info(f"Target in Z-range ({z_coord:.2f}m). Publishing 'stop' and initiating 3s lock.")
            self.supply_pub.publish(String(data='stop'))
            self.stop_command_lock_until = current_time + self.stop_command_duration
            
            self.service_call_in_progress = True
            request = PickPlace.Request()
            request.x, request.y, request.z = transformed_position.tolist()
            future = self.pick_place_client.call_async(request)
            future.add_done_callback(self.pick_place_response_callback)
        else:
            if not self.service_call_in_progress:
                self.supply_pub.publish(String(data='go'))
            
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
                x1, y1, x2, y2 = map(int, box.xyxy[0]); conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.marker_class_names[cls_id] if cls_id < len(self.marker_class_names) else "Unknown"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
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