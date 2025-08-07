#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: yolo_path_planning_node_integrated.py
# DESCRIPTION: YOLO 모델을 이용한 주행 영역 인식과 경로 생성을 하나의 노드로 통합한 버전입니다.

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros
from tf_transformations import quaternion_matrix
import message_filters
import math
import torch
from ultralytics import YOLO

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge

class YoloPathPlanningNode(Node):
    def __init__(self):
        super().__init__('yolo_path_planning_pp_node')
        self.get_logger().info("--- YOLO Path Planning Node (Integrated Mask Generation) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using compute device: {self.device}")

        # --- 경로 계획 및 Pure Pursuit 파라미터 ---
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('path_lookahead', 3.0)
        self.declare_parameter('num_path_points', 20)
        self.declare_parameter('smoothing_factor', 0.4)
        self.declare_parameter('pp_lookahead_distance', 1.0)
        self.declare_parameter('wheelbase', 0.58)
        
        self.robot_base_frame = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        self.path_lookahead = self.get_parameter('path_lookahead').get_parameter_value().double_value
        self.num_path_points = self.get_parameter('num_path_points').get_parameter_value().integer_value
        self.smoothing_factor = self.get_parameter('smoothing_factor').get_parameter_value().double_value
        self.pp_lookahead_distance = self.get_parameter('pp_lookahead_distance').get_parameter_value().double_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value

        # --- [추가] YOLO 주행 영역 모델 파라미터 ---
        try:
            self.declare_parameter('yolo_model_path', './weights.pt')
            self.declare_parameter('yolo_confidence', 0.5)
            self.declare_parameter('drivable_class_index', 0)
            
            yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
            self.path_model = YOLO(yolo_model_path).to(self.device)
            self.yolo_confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
            self.drivable_class_index = self.get_parameter('drivable_class_index').get_parameter_value().integer_value
            self.get_logger().info(f"Successfully loaded YOLO path model from: {yolo_model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO path model: {e}")
            self.destroy_node()
            return
            
        self.scaled_camera_intrinsics = None
        self.smoothed_path_points_3d = None

        # --- TF & Publisher/Subscriber 설정 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.path_pub = self.create_publisher(Path, '/competition_path_yolo', 1)
        self.steer_pub = self.create_publisher(Float64, '/steering_angle', 1)
        # [추가] 디버깅용 마스크 Publisher
        self.mask_pub_debug = self.create_publisher(Image, '/path_planning/yolo/mask_debug', 1)
        self.viz_pub = self.create_publisher(CompressedImage, '/path_planning/yolo/viz/compressed', 1)

        # [수정] Raw 카메라 토픽을 직접 구독
        realsense_img_topic = '/camera/color/image_raw/compressed'
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        info_topic = "/camera/color/camera_info"
        
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, realsense_img_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub, info_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.planning_callback)
        
        self.get_logger().info("✅ YOLO Path Planning Node (Integrated) initialized successfully.")

    def planning_callback(self, compressed_img_msg, depth_msg, info_msg):
        try:
            # 1. 이미지 처리 및 주행 가능 영역 마스크 생성
            np_arr = np.frombuffer(compressed_img_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            proc_height, proc_width, _ = cv_color.shape
            
            # YOLO 모델을 사용하여 마스크 생성
            cv_mask = self.create_yolo_drivable_mask(cv_color, proc_width, proc_height)
            if cv_mask is None:
                self.get_logger().warn("Drivable area mask could not be generated.", throttle_duration_sec=2.0)
                return

            # 디버깅용으로 생성된 마스크 발행
            self.mask_pub_debug.publish(self.bridge.cv2_to_imgmsg(cv_mask, "mono8"))
            # --- [신규] 시각화 이미지 생성 및 발행 로직 ---
            viz_image = cv_color.copy()
            # 마스크 영역을 초록색으로 표시
            viz_image[cv_mask > 0] = cv2.addWeighted(viz_image[cv_mask > 0], 0.5, np.full_like(viz_image[cv_mask > 0], (0, 255, 0)), 0.5, 0)
            viz_msg = self.bridge.cv2_to_compressed_imgmsg(viz_image)
            viz_msg.header = compressed_img_msg.header
            self.viz_pub.publish(viz_msg)
            # --- 시각화 로직 종료 ---
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            if self.scaled_camera_intrinsics is None:
                self.scale_camera_info(depth_msg, info_msg)

            # 2. 마스크로부터 3D 경로점 생성
            contours, _ = cv2.findContours(cv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return

            contour_points_2d = np.concatenate(contours, axis=0).squeeze(axis=1)
            points_3d = self.unproject_contours_to_3d(contour_points_2d, cv_depth)
            
            if points_3d.shape[0] < 50: return
            
            # 3. 경로 추종 및 조향각 계산
            self.generate_and_follow_path(points_3d, compressed_img_msg.header)

        except Exception as e:
            self.get_logger().error(f"Error in planning callback: {e}", exc_info=True)

    # --- [신규] YOLO 주행 가능 영역 마스크 생성 함수 ---
    def create_yolo_drivable_mask(self, color_image, width, height):
        results = self.path_model(color_image, conf=self.yolo_confidence, verbose=False)
        result = results[0]
        if result.masks is None: return None
        
        final_mask = np.zeros((height, width), dtype=np.uint8)
        # drivable_class_index에 해당하는 클래스만 필터링
        drivable_indices = np.where(result.boxes.cls.cpu().numpy() == self.drivable_class_index)[0]
        if len(drivable_indices) == 0: return None
        
        for idx in drivable_indices:
            mask_data = result.masks.data.cpu().numpy()[idx]
            # 모델 출력 마스크 크기가 이미지 크기와 다를 수 있으므로 리사이즈
            resized_mask = cv2.resize(mask_data, (width, height), interpolation=cv2.INTER_NEAREST)
            # 마스크를 0-255 범위로 변환하고 final_mask에 중첩
            final_mask = np.maximum(final_mask, (resized_mask * 255).astype(np.uint8))
            
        return final_mask

    def scale_camera_info(self, depth_msg, info_msg):
        proc_width, proc_height = depth_msg.width, depth_msg.height
        orig_width, orig_height = info_msg.width, info_msg.height
        
        if orig_width == 0 or orig_height == 0:
            self.get_logger().warn("Original camera info width/height is zero, cannot scale intrinsics yet.")
            return

        scale_x = proc_width / orig_width
        scale_y = proc_height / orig_height
        self.scaled_camera_intrinsics = {
            'fx': info_msg.k[0] * scale_x, 'fy': info_msg.k[4] * scale_y,
            'ppx': info_msg.k[2] * scale_x, 'ppy': info_msg.k[5] * scale_y
        }
        self.get_logger().info(f"Path planner intrinsics scaled: {self.scaled_camera_intrinsics}")

    def unproject_contours_to_3d(self, contour_points, cv_depth):
        if self.scaled_camera_intrinsics is None: return np.array([])
        u, v = contour_points[:, 0], contour_points[:, 1]
        depths = cv_depth[v, u]
        valid = depths > 0
        u, v, depths = u[valid], v[valid], depths[valid]
        if len(u) == 0: return np.array([])
        
        fx, fy = self.scaled_camera_intrinsics['fx'], self.scaled_camera_intrinsics['fy']
        cx, cy = self.scaled_camera_intrinsics['ppx'], self.scaled_camera_intrinsics['ppy']
        
        z = depths / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.vstack((x, y, z)).T

    def generate_and_follow_path(self, points_3d_cam, header):
        try:
            transform = self.tf_buffer.lookup_transform(self.robot_base_frame, header.frame_id, header.stamp, rclpy.duration.Duration(seconds=0.2))
            trans_matrix = self.transform_to_matrix(transform)
            points_hom = np.hstack((points_3d_cam, np.ones((points_3d_cam.shape[0], 1))))
            points_3d_robot = (trans_matrix @ points_hom.T).T[:, :3]
            
            valid_indices = (points_3d_robot[:, 0] > 0.1) & (points_3d_robot[:, 0] < self.path_lookahead)
            if np.sum(valid_indices) < 20: return
            
            x, y = points_3d_robot[valid_indices, 0], points_3d_robot[valid_indices, 1]
            coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(coeffs)
            
            path_x = np.linspace(0.0, x.max(), self.num_path_points)
            path_y = poly(path_x)
            
            raw_path = []
            for px, py in zip(path_x, path_y):
                dists = np.linalg.norm(points_3d_robot[:, :2] - np.array([px, py]), axis=1)
                nearby_pts = points_3d_robot[dists < 0.15]
                if nearby_pts.shape[0] > 3:
                    raw_path.append(np.array([px, py, np.median(nearby_pts[:, 2])]))
            
            if len(raw_path) < self.num_path_points / 2: return
            
            if self.smoothed_path_points_3d is None or len(self.smoothed_path_points_3d) != len(raw_path):
                self.smoothed_path_points_3d = np.array(raw_path)
            else:
                self.smoothed_path_points_3d = self.smoothing_factor * np.array(raw_path) + (1 - self.smoothing_factor) * self.smoothed_path_points_3d
            
            path_msg = Path(header=header)
            path_msg.header.frame_id = self.robot_base_frame
            for p in self.smoothed_path_points_3d:
                pose = PoseStamped(header=path_msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p[0], p[1], p[2]
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)
            
            self.calculate_and_publish_steering(self.smoothed_path_points_3d)

        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)

    def calculate_and_publish_steering(self, path_points):
        dists = np.linalg.norm(path_points[:, :2], axis=1)
        goal_idx = np.argmin(np.abs(dists - self.pp_lookahead_distance))
        goal_point = path_points[goal_idx]
        goal_x, goal_y = goal_point[0], goal_point[1]
        
        alpha = math.atan2(goal_y, goal_x)
        steering_angle = math.atan2(2.0 * self.wheelbase * math.sin(alpha), self.pp_lookahead_distance)
        
        steer_msg = Float64()
        steer_msg.data = steering_angle
        self.steer_pub.publish(steer_msg)
        self.get_logger().info(f"Published Steering Angle: {math.degrees(steering_angle):.2f} deg", throttle_duration_sec=1.0)

    def transform_to_matrix(self, t):
        trans, rot = t.transform.translation, t.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        mat[:3, 3] = [trans.x, trans.y, trans.z]
        return mat

def main(args=None):
    rclpy.init(args=args)
    node = YoloPathPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()