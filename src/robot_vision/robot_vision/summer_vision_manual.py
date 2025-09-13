#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: summer_vision_modified.py
# DESCRIPTION:
# 이 노드는 로봇의 주행과 관련된 신호등과 보급상자 인식을 전담합니다.
# [수정됨] 각 기능(신호등, 보급상자)이 독립적인 스레드 풀에서 동작하여
# 한쪽 기능의 오류나 지연이 다른 기능에 영향을 주지 않도록 견고성이 향상되었습니다.
# [사용자 요청 수정] PickPlace 서비스 통신을 제거하고, 보급 상자의 TF 변환 좌표를 토픽으로 발행하여 시각화합니다.

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
from rclpy.duration import Duration

# --- [수정] 서비스 관련 import 제거 ---
# from mtc_interfaces.srv import PickPlace

import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class TrafficSupplyRobustNode(Node):
    def __init__(self):
        super().__init__('traffic_supply_robust_node')
        self.get_logger().info("--- [Robust & Modified] Traffic Light & Supply Box Detection Node ---")
        
        self.realsense_lock = threading.Lock()
        self.usb_cam_locks = {'cam1': threading.Lock(), 'cam2': threading.Lock()}
        
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        self.use_half = self.device == 'cuda'

        # 신호등 상태 기계 파라미터 선언
        self.declare_parameter('red_light_min_area', 10000)
        self.declare_parameter('traffic_roi_top_ratio', 0.7)
        self.declare_parameter('red_light_confirmation_frames', 3)
        self.declare_parameter('red_light_loss_tolerance_frames', 7)
        self.declare_parameter('red_light_tracking_tolerance', 50)
        self.declare_parameter('traffic_light_camera_id', 'cam1')

        self.RED_LIGHT_MIN_AREA = self.get_parameter('red_light_min_area').get_parameter_value().integer_value
        self.TRAFFIC_ROI_TOP_RATIO = self.get_parameter('traffic_roi_top_ratio').get_parameter_value().double_value
        self.RED_LIGHT_CONFIRMATION_FRAMES = self.get_parameter('red_light_confirmation_frames').get_parameter_value().integer_value
        self.RED_LIGHT_LOSS_TOLERANCE_FRAMES = self.get_parameter('red_light_loss_tolerance_frames').get_parameter_value().integer_value
        self.RED_LIGHT_TRACKING_TOLERANCE = self.get_parameter('red_light_tracking_tolerance').get_parameter_value().integer_value
        self.TRAFFIC_LIGHT_CAMERA_ID = self.get_parameter('traffic_light_camera_id').get_parameter_value().string_value
        self.get_logger().info(f"Primary camera for traffic light detection: '{self.TRAFFIC_LIGHT_CAMERA_ID}'")

        # 신호등 상태 정의
        self.STATE_CLEAR = 0
        self.STATE_CANDIDATE = 1
        self.STATE_CONFIRMED_RED = 2

        # 카메라별 신호등 상태 추적기
        self.red_light_tracker = {
            'cam1': {'state': self.STATE_CLEAR, 'confirmation_counter': 0, 'loss_counter': 0, 'last_center': None},
            'cam2': {'state': self.STATE_CLEAR, 'confirmation_counter': 0, 'loss_counter': 0, 'last_center': None}
        }
        
        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value
        
        # 모델 로딩 분리
        self.supply_model = None
        self.traffic_detection_model = None
        self.supply_model_ready = False
        self.traffic_model_ready = False

        try:
            self.declare_parameter('supply_model_path', './tracking2.onnx')
            supply_model_path = self.get_parameter('supply_model_path').get_parameter_value().string_value
            self.supply_model = YOLO(supply_model_path, task='detect')
            self.supply_model_ready = True
            self.get_logger().info("✅ Supply ONNX model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load Supply model: {e}. Supply detection will be disabled.")

        try:
            self.declare_parameter('traffic_model_path', './traffic_robo2.onnx')
            traffic_model_path = self.get_parameter('traffic_model_path').get_parameter_value().string_value
            self.traffic_detection_model = YOLO(traffic_model_path, task='detect')
            self.traffic_model_class_names = ['green', 'red']
            self.traffic_model_ready = True
            self.get_logger().info("✅ Traffic ONNX model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load Traffic model: {e}. Traffic light detection will be disabled.")

        self.intrinsics = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- [수정] 서비스 클라이언트 관련 코드 제거 ---
        # self.pick_place_client = self.create_client(PickPlace, '/pick_place_service')
        # while not self.pick_place_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('pick_place service not available, waiting...')
        # self.service_call_in_progress = False

        # 퍼블리셔 설정
        self.status_pub = self.create_publisher(Bool, '/supply_status', 10)
        self.traffic_pub = self.create_publisher(String, '/traffic_command', 10)
        
        # --- [수정] /supply_command 퍼블리셔 제거하고 좌표 퍼블리셔 추가 ---
        # self.supply_pub = self.create_publisher(String, '/supply_command', 10)
        self.supply_position_pub = self.create_publisher(PointStamped, '/supply_box_position', 10)
        
        self.realsense_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/realsense/viz/compressed', 10)
        self.usb_cam1_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam1/viz/compressed', 10)
        self.usb_cam2_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam2/viz/compressed', 10)

        self.resized_color_yolo = np.empty((self.proc_height, self.proc_width, 3), dtype=np.uint8)
        self.resized_depth = np.empty((self.proc_height, self.proc_width), dtype=np.uint16)

        # 스레드 풀 분리
        self.supply_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='supply_worker')
        self.traffic_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='traffic_worker')
        
        self._is_shutting_down = False
        
        # 구독자 설정
        self.camera_info_sub = self.create_subscription(CameraInfo, "/camera/color/camera_info", self.camera_info_callback, 10)
        self.get_logger().info("Waiting for CameraInfo...")

        self.usb_cam1_sub = self.create_subscription(CompressedImage, 'camera1/image_raw/compressed', lambda msg: self.usb_cam_callback(msg, 'cam1'), 10)
        self.usb_cam2_sub = self.create_subscription(CompressedImage, 'camera2/image_raw/compressed', lambda msg: self.usb_cam_callback(msg, 'cam2'), 10)
        
    def camera_info_callback(self, info_msg):
        if self.intrinsics is not None: return
        self.get_logger().info("✅ CameraInfo received.")
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = info_msg.width; self.intrinsics.height = info_msg.height
        self.intrinsics.ppx = info_msg.k[2]; self.intrinsics.ppy = info_msg.k[5]
        self.intrinsics.fx = info_msg.k[0]; self.intrinsics.fy = info_msg.k[4]
        self.intrinsics.model = rs2.distortion.brown_conrady if info_msg.distortion_model == 'plumb_bob' else rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = list(info_msg.d)
        self.initialize_image_sync()
        self.destroy_subscription(self.camera_info_sub)

    def initialize_image_sync(self):
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub], queue_size=5, slop=0.2)
        self.ts.registerCallback(self.realsense_callback)
        self.get_logger().info("✅ Traffic & Supply Node initialized successfully.")

    def realsense_callback(self, compressed_image_msg, depth_msg):
        if self.intrinsics is None or self._is_shutting_down: return
        if self.realsense_lock.acquire(blocking=False):
            try:
                self.supply_executor.submit(self._process_realsense_data, compressed_image_msg, depth_msg)
            finally:
                pass # The lock is released in the worker thread
        else:
            self.get_logger().warn("Dropping a Realsense frame, processing is busy.", throttle_duration_sec=1)

    def usb_cam_callback(self, compressed_msg, camera_id):
        if self._is_shutting_down: return
        lock = self.usb_cam_locks[camera_id]
        if lock.acquire(blocking=False):
            try:
                self.traffic_executor.submit(self._process_usb_cam_data, compressed_msg, camera_id)
            finally:
                 pass # The lock is released in the worker thread
        else:
            self.get_logger().warn(f"Dropping a frame from {camera_id}, processing is busy.", throttle_duration_sec=1)

    def _process_realsense_data(self, compressed_image_msg, depth_msg):
        if not self.supply_model_ready:
            self.realsense_lock.release()
            return
        try:
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            if cv_color is None or cv_depth_raw is None: return
            
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
    
    def _process_usb_cam_data(self, compressed_msg, camera_id):
        lock = self.usb_cam_locks[camera_id]
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None: return

            annotated_image = cv_image.copy()

            if self.traffic_model_ready and camera_id == self.TRAFFIC_LIGHT_CAMERA_ID:
                tracker = self.red_light_tracker[camera_id]
                results_traffic = self.traffic_detection_model(cv_image, conf=0.5, iou=0.45, verbose=False, half=self.use_half)
                
                green_found = False
                best_red_candidate_center = None
                
                h, w, _ = cv_image.shape
                for r in results_traffic:
                    for box_data in r.boxes.cpu().numpy():
                        label = self.traffic_model_class_names[int(box_data.cls[0])]
                        if label == 'green':
                            green_found = True
                            continue
                        if label == 'red' and box_data.conf[0] >= 0.7:
                            box = box_data.xyxy[0].astype(int)
                            if (box[2] - box[0]) * (box[3] - box[1]) < self.RED_LIGHT_MIN_AREA: continue
                            cy = (box[1] + box[3]) / 2
                            if cy > h * self.TRAFFIC_ROI_TOP_RATIO: continue
                            best_red_candidate_center = np.array([(box[0] + box[2]) / 2, cy])
                            break
                    if best_red_candidate_center is not None: break
                
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
                elif tracker['state'] == self.STATE_CANDIDATE:
                    if red_light_detected_in_frame:
                        tracker['confirmation_counter'] += 1
                        if tracker['confirmation_counter'] >= self.RED_LIGHT_CONFIRMATION_FRAMES:
                            tracker['state'] = self.STATE_CONFIRMED_RED
                            self.get_logger().info(f"[{camera_id}] Red light confirmed.")
                    else:
                        tracker['state'] = self.STATE_CLEAR
                        tracker['last_center'] = None
                elif tracker['state'] == self.STATE_CONFIRMED_RED:
                    if not red_light_detected_in_frame:
                        tracker['loss_counter'] += 1
                        if tracker['loss_counter'] >= self.RED_LIGHT_LOSS_TOLERANCE_FRAMES:
                            tracker['state'] = self.STATE_CLEAR
                            tracker['last_center'] = None
                            self.get_logger().info(f"[{camera_id}] Red light lost.")
                    else:
                        tracker['loss_counter'] = 0

                command_to_publish = "stop" if tracker['state'] == self.STATE_CONFIRMED_RED else "go" if green_found else None
                if command_to_publish:
                    self.traffic_pub.publish(String(data=command_to_publish))
                    
                annotated_image = self.draw_traffic_detections(cv_image, results_traffic)

            viz_publisher = self.usb_cam1_viz_pub if camera_id == 'cam1' else self.usb_cam2_viz_pub
            self.publish_compressed_viz(viz_publisher, annotated_image)

        except Exception as e:
            self.get_logger().error(f"Error in USB Cam worker ({camera_id}): {e}\n{traceback.format_exc()}")
        finally:
            lock.release()

    def run_supply_tracking(self, color_image_to_draw, resized_depth_image, yolo_input_image, header):
        if self.intrinsics is None: return False
        
        results = self.supply_model(yolo_input_image, verbose=False, half=self.use_half)
        supply_found_this_frame = False

        best_box = max(results[0].boxes, key=lambda box: box.conf[0], default=None)

        if best_box is not None and int(best_box.cls) == 0:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cx_res, cy_res = (x1 + x2) // 2, (y1 + y2) // 2
            
            if 0 <= cy_res < self.proc_height and 0 <= cx_res < self.proc_width:
                depth_in_mm = resized_depth_image[cy_res, cx_res]
                
                if depth_in_mm > 0:
                    supply_found_this_frame = True
                    orig_cx = int(cx_res * self.intrinsics.width / self.proc_width)
                    orig_cy = int(cy_res * self.intrinsics.height / self.proc_height)
                    optical_frame_coords = np.array([p / 1000.0 for p in rs2.rs2_deproject_pixel_to_point(self.intrinsics, [orig_cx, orig_cy], depth_in_mm)])
                    
                    point_in_optical_frame = PointStamped()
                    point_in_optical_frame.header.frame_id = "camera_color_optical_frame"
                    point_in_optical_frame.header.stamp = header.stamp
                    point_in_optical_frame.point.x, point_in_optical_frame.point.y, point_in_optical_frame.point.z = optical_frame_coords
                    
                    try:
                        transform = self.tf_buffer.lookup_transform("camera_bottom_screw_frame", point_in_optical_frame.header.frame_id, rclpy.time.Time())
                        point_in_target_frame = do_transform_point(point_in_optical_frame, transform)
                        
                        # --- [수정] 서비스 호출 대신 좌표를 토픽으로 발행 ---
                        self.supply_position_pub.publish(point_in_target_frame)
                        
                        transformed_position = np.array([point_in_target_frame.point.x, point_in_target_frame.point.y, point_in_target_frame.point.z])
                        
                        label = f"x={transformed_position[0]:.2f}m, y={transformed_position[1]:.2f}m, z={transformed_position[2]:.2f}m"
                        scale_w, scale_h = self.intrinsics.width / self.proc_width, self.intrinsics.height / self.proc_height
                        orig_x1, orig_y1, orig_x2, orig_y2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
                        cv2.rectangle(color_image_to_draw, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 255), 2)
                        cv2.putText(color_image_to_draw, label, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        self.get_logger().warn(f"Coordinate transform failed: {e}", throttle_duration_sec=5.0)
                        return False

        # --- [수정] 서비스 및 `stop` 명령 관련 로직 전체 제거 ---

        return supply_found_this_frame

    # --- [수정] 서비스 콜백 함수 제거 ---
    # def pick_place_response_callback(self, future):
    #     ...

    def publish_compressed_viz(self, publisher, cv_image):
        msg = CompressedImage(format="jpeg")
        msg.header.stamp = self.get_clock().now().to_msg()
        success, encoded_image = cv2.imencode('.jpg', cv_image)
        if success:
            msg.data = encoded_image.tobytes()
            publisher.publish(msg)

    def draw_traffic_detections(self, image, results):
        for r in results:
            for box in r.boxes.cpu().numpy():
                cls_id = int(box.cls[0])
                if cls_id < len(self.traffic_model_class_names):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = self.traffic_model_class_names[cls_id]
                    color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"{label}: {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image

    def destroy_node(self):
        self.get_logger().info("Shutting down the thread pools.")
        self._is_shutting_down = True
        self.supply_executor.shutdown(wait=True)
        self.traffic_executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TrafficSupplyRobustNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()