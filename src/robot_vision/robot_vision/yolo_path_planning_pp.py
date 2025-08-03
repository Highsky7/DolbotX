#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: yolo_path_planning_node.py (Corrected and Integrated with Pure Pursuit)

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros
from tf_transformations import quaternion_matrix
import message_filters
import math

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64  # [추가] 조향각 메시지 타입
from cv_bridge import CvBridge

class YoloPathPlanningNode(Node):
    def __init__(self):
        super().__init__('yolo_path_planning_pp_node')
        self.get_logger().info("--- YOLO Path Planning Node with Pure Pursuit Control ---")
        self.bridge = CvBridge()

        # --- 기존 파라미터 ---
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('path_lookahead', 3.0)
        self.declare_parameter('num_path_points', 20)
        self.declare_parameter('smoothing_factor', 0.4)
        self.robot_base_frame = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        self.path_lookahead = self.get_parameter('path_lookahead').get_parameter_value().double_value
        self.num_path_points = self.get_parameter('num_path_points').get_parameter_value().integer_value
        self.smoothing_factor = self.get_parameter('smoothing_factor').get_parameter_value().double_value

        # --- [추가] Pure Pursuit 파라미터 ---
        self.declare_parameter('pp_lookahead_distance', 1.0)
        self.declare_parameter('wheelbase', 0.58)
        self.pp_lookahead_distance = self.get_parameter('pp_lookahead_distance').get_parameter_value().double_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value

        self.scaled_camera_intrinsics = None
        self.smoothed_path_points_3d = None

        # --- TF & Publisher/Subscriber 설정 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # [수정] Publisher 추가: 생성된 경로(디버깅용)와 최종 조향각
        self.path_pub = self.create_publisher(Path, '/competition_path_yolo', 1)
        self.steer_pub = self.create_publisher(Float64, '/steering_angle', 1)

        mask_topic = '/path_planning/yolo/mask'
        depth_topic = '/path_planning/yolo/depth'
        info_topic = '/path_planning/yolo/info'
        mask_sub = message_filters.Subscriber(self, Image, mask_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([mask_sub, depth_sub, info_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.planning_callback)
        
        self.get_logger().info(f"Subscribing to intermediate topics: {mask_topic}, {depth_topic}, {info_topic}")
        self.get_logger().info("✅ YOLO Path Planning Node with Pure Pursuit initialized successfully.")

    def planning_callback(self, mask_msg, depth_msg, info_msg):
        try:
            cv_mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            if self.scaled_camera_intrinsics is None:
                self.scale_camera_info(depth_msg, info_msg)

            contours, _ = cv2.findContours(cv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return

            contour_points_2d = np.concatenate(contours, axis=0).squeeze(axis=1)
            points_3d = self.unproject_contours_to_3d(contour_points_2d, cv_depth)
            
            if points_3d.shape[0] < 50: return
            
            # [수정] 함수 이름 변경 및 로직 흐름 명확화
            self.generate_and_follow_path(points_3d, mask_msg.header)

        except Exception as e:
            self.get_logger().error(f"Error in planning callback: {e}", exc_info=True)

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
            
            # --- 경로 시각화 메시지 발행 (디버깅용) ---
            path_msg = Path(header=header)
            path_msg.header.frame_id = self.robot_base_frame
            for p in self.smoothed_path_points_3d:
                pose = PoseStamped(header=path_msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p[0], p[1], p[2]
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)
            
            # --- [추가] Pure Pursuit 로직 호출 ---
            self.calculate_and_publish_steering(self.smoothed_path_points_3d)

        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)

    # --- [신규] Pure Pursuit 조향각 계산 및 발행 함수 ---
    def calculate_and_publish_steering(self, path_points):
        """
        생성된 경로(base_link 기준)를 받아 Pure Pursuit 알고리즘으로 조향각을 계산합니다.
        """
        # 1. 목표 지점(Goal Point) 찾기
        # 로봇(0,0)에서 각 경로점까지의 거리를 계산
        dists = np.linalg.norm(path_points[:, :2], axis=1)
        
        # 전방 주시 거리(lookahead_distance)와 가장 가까운 경로점을 목표 지점으로 선택
        # np.argmin은 |dists - Ld|가 최소가 되는 인덱스를 반환
        goal_idx = np.argmin(np.abs(dists - self.pp_lookahead_distance))
        goal_point = path_points[goal_idx]
        
        # 2. 조향각 계산
        # 목표 지점 (goal_x, goal_y)는 로봇의 base_link 좌표계에 있음
        goal_x, goal_y = goal_point[0], goal_point[1]
        
        # 로봇의 현재 헤딩과 목표 지점 사이의 각도(alpha) 계산
        # alpha는 로봇 전방 방향과 목표 지점을 잇는 선 사이의 각도
        alpha = math.atan2(goal_y, goal_x)
        
        # Pure Pursuit 공식: delta = atan(2 * L * sin(alpha) / |V|)
        # 여기서는 |V|(속도) 대신 전방 주시 거리를 사용하여 곡률을 계산
        # 조향각(delta) = atan2(2 * wheelbase * sin(alpha), lookahead_distance)
        steering_angle = math.atan2(2.0 * self.wheelbase * math.sin(alpha), self.pp_lookahead_distance)
        
        # 3. 조향각 발행
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
