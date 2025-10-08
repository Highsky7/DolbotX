#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for Fused Drivable Area Segmentation and Path Planning.

This script implements a ROS2 node for the "Summer" mission profile. It uses
three separate YOLO segmentation models to identify multiple types of drivable
surfaces: standard track, sand, and stone. The outputs of these models are
fused into a single unified drivable area mask.

From this fused mask, the node generates a smooth, navigable path using a
3rd-order Bézier curve and calculates the required steering angle for the robot
to follow this path using the Pure Pursuit algorithm.

Key Features:
- **Triple-Model Fusion:** Combines results from three distinct YOLO models.
- **Bézier Curve Path Planning:** Generates a robust path for complex terrain.
- **Efficient Processing:** A multi-threaded architecture offloads heavy computation.
- **Intelligent Control Point Selection:** Automatically determines Bézier control
  points within a defined Region of Interest (ROI).
- **Pure Pursuit Control:** Calculates steering commands based on a lookahead distance.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, Bool
from cv_bridge import CvBridge
import traceback

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


def generate_bezier_curve(p0, p1, p2, p3, num_points=50):
    """
    Generate a 3rd-order Bézier curve from four control points.

    Args:
        p0 (tuple): The starting point (x, y).
        p1 (tuple): The first control point (handle for p0).
        p2 (tuple): The second control point (handle for p3).
        p3 (tuple): The end point (x, y).
        num_points (int): The number of points to generate on the curve.

    Returns:
        np.ndarray: A numpy array of shape (num_points, 2) representing
                    the points on the Bézier curve.
    """
    t = np.linspace(0, 1, num_points)
    t_1 = 1.0 - t
    x = t_1**3 * p0[0] + 3 * t_1**2 * t * p1[0] + 3 * t_1 * t**2 * p2[0] + t**3 * p3[0]
    y = t_1**3 * p0[1] + 3 * t_1**2 * t * p1[1] + 3 * t_1 * t**2 * p2[1] + t**3 * p3[1]
    return np.vstack((x, y)).T


def morph_close(binary_mask, ksize=5):
    """
    Apply a morphological closing operation to a binary mask.

    Args:
        binary_mask (np.ndarray): The input binary image (mask).
        ksize (int): The size of the square kernel for the operation.

    Returns:
        np.ndarray: The binary mask after the closing operation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


def remove_small_components(binary_mask, min_size=300):
    """
    Remove small connected components from a binary mask.

    Args:
        binary_mask (np.ndarray): The input binary image (mask).
        min_size (int): The minimum area of a component to be kept.

    Returns:
        np.ndarray: A new binary mask with small components removed.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    if num_labels > 1:
        largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[largest_component_label, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == largest_component_label] = 255
    return cleaned


class YoloBevFusedDrivableAreaNode(Node):
    """
    A ROS2 node for fused drivable area detection and path planning.

    This node subscribes to a compressed image topic, performs a BEV
    transformation, and uses three separate YOLO models (for regular
    drivable area, sand, and stone) to segment the environment. The resulting
    masks are fused, a path is calculated via a Bézier curve, and a steering
    angle is published.

    Attributes:
        drive_area_model (YOLO): Model for general drivable surfaces.
        sand_model (YOLO): Model for detecting sand.
        stone_model (YOLO): Model for detecting stone/gravel.
        ... (and other attributes similar to YoloBevDrivableAreaNode)
    """
    _MORPH_KSIZE = 7
    _MIN_AREA_SIZE = 15000

    _ROI_TOP_Y_RATIO = 0.0
    _ROI_BOTTOM_Y_RATIO = 1.0
    _ROI_WIDTH_RATIO = 1.0

    _BEZIER_HANDLE_RATIO = 0.5

    def __init__(self):
        """
        Initialize the YoloBevFusedDrivableAreaNode.

        This sets up the node, declares and gets parameters for three YOLO
        models, loads all models and BEV parameters, and creates publishers,
        subscribers, and the thread pool executor.
        """
        super().__init__('yolo_bev_fused_drivable_area_node')
        self.get_logger().info("--- YOLO BEV Fused Drivable Area Node (Hinton's Triple-Fusion Bézier Architecture) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.declare_parameter('drive_area_model_path', './drive_area2.onnx')
        self.declare_parameter('sand_model_path', './sand.onnx')
        self.declare_parameter('stone_model_path', './stone.onnx')
        self.declare_parameter('drive_area_confidence', 0.5)
        self.declare_parameter('sand_confidence', 0.5)
        self.declare_parameter('stone_confidence', 0.5)
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('wheelbase', 0.6)
        self.declare_parameter('camera_to_rear_axle_offset', 0.27)
        self.declare_parameter('lookahead_distance', 0.66)

        # Get Parameters
        params = {name: self.get_parameter(name).get_parameter_value() for name in self.get_parameters_and_include_prefix('')}

        self.drive_area_confidence = params['drive_area_confidence'].double_value
        self.sand_confidence = params['sand_confidence'].double_value
        self.stone_confidence = params['stone_confidence'].double_value
        self.L = params['wheelbase'].double_value
        self.CAMERA_TO_REAR_AXLE_OFFSET = params['camera_to_rear_axle_offset'].double_value
        self.lookahead_distance = params['lookahead_distance'].double_value

        try:
            self.drive_area_model = YOLO(params['drive_area_model_path'].string_value, task='segment')
            self.sand_model = YOLO(params['sand_model_path'].string_value, task='segment')
            self.stone_model = YOLO(params['stone_model_path'].string_value, task='segment')

            bev_params = np.load(params['bev_param_file'].string_value)
            self.bev_h, self.bev_w = int(bev_params['warp_h']), int(bev_params['warp_w'])
            self.M_bev = cv2.getPerspectiveTransform(bev_params['src_points'], bev_params['dst_points'])
            self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.002609375, 0.66, 0.0011171875
            self.get_logger().info(f"✅ Triple ONNX models and resources loaded on [{self.device}].")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to load resources: {e}")
            rclpy.shutdown()
            return

        self.planning_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='planning_worker')
        self._is_shutting_down = False

        self.steer_pub = self.create_publisher(Float64, '/steering_angle', 10)
        self.viz_pub = self.create_publisher(CompressedImage, '/path_planning/drivable_area/viz/compressed', 10)
        self.status_pub = self.create_publisher(Bool, '/path_planning/drivable_area/status', 10)

        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.img_sub = self.create_subscription(CompressedImage, '/camera3/image_raw/compressed', self.planning_callback, qos_profile)
        self.get_logger().info("✅ Node initialized.")

    def planning_callback(self, msg):
        """
        Receive a compressed image and submit it for processing.

        Args:
            msg (CompressedImage): The incoming ROS2 message.
        """
        if self._is_shutting_down: return
        self.planning_executor.submit(self._process_planning_data, msg.data)

    def _process_planning_data(self, compressed_data):
        """
        Execute the full path planning pipeline in a worker thread.

        This function decodes the image, runs BEV, performs inference with
        three separate models, fuses their masks, calculates the steering
        angle, and publishes the results.

        Args:
            compressed_data (bytes): The raw byte data from the message.
        """
        try:
            np_arr = np.frombuffer(compressed_data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_color is None: return

            bev_image = self.do_bev_transform(cv_color)

            # Run inference with all three models
            drive_area_results = self.drive_area_model(bev_image, conf=self.drive_area_confidence, verbose=False)
            sand_results = self.sand_model(bev_image, conf=self.sand_confidence, verbose=False)
            stone_results = self.stone_model(bev_image, conf=self.stone_confidence, verbose=False)

            # Extract and fuse the masks
            drive_area_mask = self.extract_mask(drive_area_results)
            sand_mask = self.extract_mask(sand_results)
            stone_mask = self.extract_mask(stone_results)

            temp_mask = cv2.bitwise_or(drive_area_mask, sand_mask)
            unified_mask = cv2.bitwise_or(temp_mask, stone_mask)

            filtered_mask = self.filter_drivable_mask(unified_mask)
            steering_angle_rad, viz_data = self.calculate_steering_from_area(filtered_mask)

            steer_msg = Float64()
            steer_msg.data = steering_angle_rad if steering_angle_rad is not None else 0.0
            self.steer_pub.publish(steer_msg)

            if self.viz_pub.get_subscription_count() > 0:
                final_viz_angle = steering_angle_rad if steering_angle_rad is not None else 0.0
                self.publish_visualization(bev_image, filtered_mask, viz_data, final_viz_angle)
        except Exception:
            self.get_logger().error(f"Error in planning worker:\n{traceback.format_exc()}")

    def extract_mask(self, results):
        """
        Extract a single binary mask from YOLO model results.

        Args:
            results: The output from a YOLO model's segmentation task.

        Returns:
            np.ndarray: A binary mask where detected areas are 255.
        """
        mask = np.zeros(results[0].orig_shape, dtype=np.uint8)
        if results[0].masks:
            # Combine all detected masks into a single one
            mask = np.max(np.array([m.cpu().numpy() for m in results[0].masks.data]), axis=0)
            mask = (mask * 255).astype(np.uint8)
        return mask

    def do_bev_transform(self, image):
        """
        Apply a perspective transformation to get a Bird's-Eye View.

        Args:
            image (np.ndarray): The input image from the camera.

        Returns:
            np.ndarray: The transformed BEV image.
        """
        return cv2.warpPerspective(image, self.M_bev, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)

    def filter_drivable_mask(self, bev_mask):
        """
        Clean up the raw fused segmentation mask.

        Args:
            bev_mask (np.ndarray): The raw binary mask from the fusion.

        Returns:
            np.ndarray: The cleaned binary mask.
        """
        f1 = morph_close(bev_mask, ksize=self._MORPH_KSIZE)
        return remove_small_components(f1, min_size=self._MIN_AREA_SIZE)

    def image_to_vehicle(self, pt_bev):
        """
        Convert a point from BEV image coordinates to vehicle coordinates.

        Args:
            pt_bev (tuple): A tuple (u, v) of pixel coordinates in the BEV image.

        Returns:
            tuple: A tuple (x, y) of coordinates in the vehicle frame (meters).
        """
        u, v = pt_bev
        y_cam = (self.bev_w / 2 - u) * self.m_per_pixel_x
        x_cam = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        return x_cam - self.CAMERA_TO_REAR_AXLE_OFFSET, y_cam

    def calculate_steering_from_area(self, area_mask):
        """
        Calculate the steering angle from the fused drivable area mask.

        Note: The implementation is identical to the non-fused version.

        Args:
            area_mask (np.ndarray): The cleaned binary mask of the drivable area.

        Returns:
            tuple: A tuple containing the steering angle (rad) and viz data.
        """
        roi_top_y = int(self.bev_h * self._ROI_TOP_Y_RATIO)
        roi_bottom_y = int(self.bev_h * self._ROI_BOTTOM_Y_RATIO) - 1
        roi_half_width = int((self.bev_w * self._ROI_WIDTH_RATIO) / 2)
        roi_center_x = self.bev_w // 2
        roi_left_x = roi_center_x - roi_half_width
        roi_right_x = roi_center_x + roi_half_width

        roi_mask = np.zeros_like(area_mask)
        cv2.rectangle(roi_mask, (roi_left_x, roi_top_y), (roi_right_x, roi_bottom_y), 255, -1)
        roi_area_mask = cv2.bitwise_and(area_mask, area_mask, mask=roi_mask)

        is_detected = np.any(roi_area_mask)
        self.status_pub.publish(Bool(data=bool(is_detected)))
        viz_data = {'roi_coords': (roi_left_x, roi_top_y, roi_right_x, roi_bottom_y)}

        if not is_detected:
            return None, viz_data

        p0 = (roi_center_x, roi_bottom_y)
        top_points_y, top_points_x = np.where(roi_area_mask[roi_top_y:roi_top_y+10, :] > 0)
        p3 = (int(np.mean(top_points_x)), roi_top_y) if len(top_points_x) > 0 else (roi_center_x, roi_top_y)

        roi_height = roi_bottom_y - roi_top_y
        handle_offset = int(roi_height * self._BEZIER_HANDLE_RATIO)
        p1 = (p0[0], p0[1] - handle_offset)
        p2 = (p3[0], p3[1] + handle_offset)

        path_points = generate_bezier_curve(p0, p1, p2, p3)
        x_bev_coords, y_bev_coords = path_points[:, 0], path_points[:, 1]

        viz_data.update({'bezier_points': path_points, 'control_points': [p0, p1, p2, p3]})

        x_veh, y_veh = self.image_to_vehicle((x_bev_coords, y_bev_coords))
        dist_from_ego = np.sqrt(x_veh**2 + y_veh**2)

        goal_idx_candidates = np.where(np.abs(dist_from_ego - self.lookahead_distance) < 0.2)[0]

        goal_idx = None
        if len(goal_idx_candidates) > 0:
            goal_idx = goal_idx_candidates[np.argmin(np.abs(dist_from_ego[goal_idx_candidates] - self.lookahead_distance))]
        elif len(dist_from_ego) > 0:
            goal_idx = np.argmax(dist_from_ego)

        if goal_idx is not None:
            actual_lookahead_dist = dist_from_ego[goal_idx]
            if actual_lookahead_dist > 0.1:
                x_goal, y_goal = x_veh[goal_idx], y_veh[goal_idx]
                alpha = math.atan2(y_goal, x_goal)
                steering_angle = math.atan2(2.0 * self.L * math.sin(alpha), actual_lookahead_dist)
                goal_point_bev = (int(x_bev_coords[goal_idx]), int(y_bev_coords[goal_idx]))
                viz_data['goal_point_bev'] = goal_point_bev
                return steering_angle, viz_data

        return None, viz_data

    def publish_visualization(self, bev_image, area_mask, viz_data, steering_angle_rad):
        """
        Create and publish a visualization image.

        Note: The implementation is identical to the non-fused version.

        Args:
            bev_image (np.ndarray): The base BEV image.
            area_mask (np.ndarray): The binary mask of the drivable area.
            viz_data (dict): A dictionary containing all visualization elements.
            steering_angle_rad (float): The final steering angle in radians.
        """
        viz_image = bev_image.copy()
        green_overlay = np.zeros_like(viz_image)
        green_overlay[area_mask > 0] = (0, 255, 0)
        viz_image = cv2.addWeighted(viz_image, 1, green_overlay, 0.4, 0)

        if 'roi_coords' in viz_data:
            x1, y1, x2, y2 = viz_data['roi_coords']
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if 'bezier_points' in viz_data:
            path_points = viz_data['bezier_points'].astype(np.int32)
            cv2.polylines(viz_image, [path_points], isClosed=False, color=(255, 255, 0), thickness=3)
        if 'control_points' in viz_data:
            p0, p1, p2, p3 = viz_data['control_points']
            cv2.line(viz_image, p0, p1, (255, 0, 255), 2)
            cv2.line(viz_image, p2, p3, (255, 0, 255), 2)
            for p in viz_data['control_points']:
                cv2.circle(viz_image, p, 8, (255, 0, 255), -1)

        if 'goal_point_bev' in viz_data:
            cv2.circle(viz_image, viz_data['goal_point_bev'], 10, (0, 0, 255), -1)

        steer_text = f"Steer: {math.degrees(steering_angle_rad):.1f} deg"
        cv2.putText(viz_image, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        _, jpeg_buffer = cv2.imencode('.jpg', viz_image)
        viz_msg = CompressedImage(data=jpeg_buffer.tobytes(), format="jpeg")
        viz_msg.header.stamp = self.get_clock().now().to_msg()
        self.viz_pub.publish(viz_msg)

    def destroy_node(self):
        """Clean up resources before shutting down the node."""
        self.get_logger().info("Shutting down the planning thread pool.")
        self._is_shutting_down = True
        self.planning_executor.shutdown(wait=True)
        super().destroy_node()


def main(args=None):
    """The main entry point for the ROS2 node."""
    rclpy.init(args=args)
    node = YoloBevFusedDrivableAreaNode()
    if rclpy.ok():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()