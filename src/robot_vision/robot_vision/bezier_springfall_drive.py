#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for Drivable Area Segmentation and Path Planning.

This script implements a ROS2 node that uses a YOLO segmentation model to
identify the drivable area from a Bird's-Eye View (BEV) perspective.
It then generates a smooth, navigable path using a 3rd-order Bézier curve
and calculates the required steering angle for the robot to follow this path
using the Pure Pursuit algorithm.

Key Features:
- **Bézier Curve Path Planning:** Generates a robust path that handles sharp turns.
- **Efficient Processing:** A multi-threaded architecture offloads heavy computation
  from the ROS2 callback, preventing communication delays.
- **Intelligent Control Point Selection:** Automatically determines Bézier control
  points within a defined Region of Interest (ROI).
- **Pure Pursuit Control:** Calculates steering commands based on a lookahead distance.
- **Fail-Safe Mechanism:** Publishes a neutral steering angle (0.0) if no
  drivable area is detected.
- **Dynamic Visualization:** Publishes a compressed image topic for real-time
  visualization of the path planning process, but only if there is a subscriber.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import threading

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

    # 3rd-order Bézier curve formula
    x = t_1**3 * p0[0] + 3 * t_1**2 * t * p1[0] + 3 * t_1 * t**2 * p2[0] + t**3 * p3[0]
    y = t_1**3 * p0[1] + 3 * t_1**2 * t * p1[1] + 3 * t_1 * t**2 * p2[1] + t**3 * p3[1]

    return np.vstack((x, y)).T


def morph_close(binary_mask, ksize=5):
    """
    Apply a morphological closing operation to a binary mask.

    This helps to close small holes inside the foreground objects.

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

    This function finds all connected components (blobs) in the binary mask
    and removes any that are smaller than a specified minimum size. It keeps
    the largest component if it meets the minimum size criteria.

    Args:
        binary_mask (np.ndarray): The input binary image (mask).
        min_size (int): The minimum area of a component to be kept.

    Returns:
        np.ndarray: A new binary mask with small components removed.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    if num_labels > 1:
        # Find the label of the largest component, ignoring background (label 0)
        largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # Keep it only if it's larger than min_size
        if stats[largest_component_label, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == largest_component_label] = 255
    return cleaned


class YoloBevDrivableAreaNode(Node):
    """
    A ROS2 node for drivable area detection and path planning.

    This node subscribes to a compressed image topic, performs a Bird's-Eye-View
    (BEV) transformation, uses a YOLO model to segment the drivable area,
    calculates a smooth path using a Bézier curve, and publishes a steering
    angle command.

    Attributes:
        _MORPH_KSIZE (int): Kernel size for morphological closing.
        _MIN_AREA_SIZE (int): Minimum area to keep a detected component.
        _ROI_TOP_Y_RATIO (float): Ratio for the top of the ROI.
        _ROI_BOTTOM_Y_RATIO (float): Ratio for the bottom of the ROI.
        _ROI_WIDTH_RATIO (float): Ratio for the width of the ROI.
        _BEZIER_HANDLE_RATIO (float): Ratio to control Bézier curve smoothness.
        bridge (CvBridge): ROS2 to OpenCV image converter.
        device (str): The computation device ('cuda' or 'cpu').
        yolo_confidence (float): Confidence threshold for YOLO model.
        L (float): The wheelbase of the vehicle in meters.
        CAMERA_TO_REAR_AXLE_OFFSET (float): Offset from camera to rear axle.
        lookahead_distance (float): Lookahead distance for Pure Pursuit.
        model (YOLO): The loaded YOLO segmentation model.
        bev_h (int): Height of the BEV image.
        bev_w (int): Width of the BEV image.
        M_bev (np.ndarray): Perspective transformation matrix for BEV.
        planning_executor (ThreadPoolExecutor): For running planning in a worker thread.
        steer_pub (Publisher): Publishes the calculated steering angle.
        viz_pub (Publisher): Publishes the visualization image.
        status_pub (Publisher): Publishes the detection status.
        img_sub (Subscription): Subscribes to the raw compressed image topic.
    """
    _MORPH_KSIZE = 7
    _MIN_AREA_SIZE = 15000

    _ROI_TOP_Y_RATIO = 0.0
    _ROI_BOTTOM_Y_RATIO = 1.0
    _ROI_WIDTH_RATIO = 1.0

    # Parameter for adjusting Bézier curve control point positions
    _BEZIER_HANDLE_RATIO = 0.5  # Larger values make the curve smoother (0.0 ~ 1.0)

    def __init__(self):
        """
        Initialize the YoloBevDrivableAreaNode.

        This involves setting up the node, declaring and getting parameters,
        loading the YOLO model and BEV transformation matrix, and creating
        publishers, subscribers, and the thread pool executor.
        """
        super().__init__('yolo_bev_drivable_area_node')
        self.get_logger().info("--- YOLO BEV Drivable Area Planning Node (Hinton's Bézier Curve Architecture) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.declare_parameter('yolo_model_path', './drive_area2.onnx')
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('wheelbase', 0.6)
        self.declare_parameter('camera_to_rear_axle_offset', 0.27)
        self.declare_parameter('lookahead_distance', 0.66)

        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.CAMERA_TO_REAR_AXLE_OFFSET = self.get_parameter('camera_to_rear_axle_offset').get_parameter_value().double_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value

        try:
            self.model = YOLO(yolo_model_path, task='segment')
            bev_params = np.load(bev_param_file)
            self.bev_h, self.bev_w = int(bev_params['warp_h']), int(bev_params['warp_w'])
            self.M_bev = cv2.getPerspectiveTransform(bev_params['src_points'], bev_params['dst_points'])
            self.get_logger().info("✅ BEV transformation matrix calculated.")
            self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.002609375, 0.66, 0.0011171875
            self.get_logger().info(f"✅ ONNX model and all resources loaded on [{self.device}].")
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
        self.get_logger().info("✅ Node initialized and subscribing to image topic.")

    def planning_callback(self, compressed_img_msg):
        """
        Receive a compressed image and submit it for processing.

        This is the main subscription callback. To avoid blocking the ROS2
        executor, it offloads the actual processing to a worker thread via
        a ThreadPoolExecutor.

        Args:
            compressed_img_msg (CompressedImage): The incoming ROS2 message.
        """
        if self._is_shutting_down: return
        try:
            self.planning_executor.submit(self._process_planning_data, compressed_img_msg.data)
        except Exception as e:
            self.get_logger().warn(f"Failed to submit planning task: {e}")

    def _process_planning_data(self, compressed_data_buffer):
        """
        Execute the full path planning pipeline in a worker thread.

        This function decodes the image, runs the BEV transformation, performs
        YOLO inference, filters the resulting mask, calculates the steering
        angle, and publishes the results and visualization.

        Args:
            compressed_data_buffer (bytes): The raw byte data from the
                                            CompressedImage message.
        """
        try:
            np_arr = np.frombuffer(compressed_data_buffer, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_color is None: return

            bev_image = self.do_bev_transform(cv_color)
            results = self.model(bev_image, conf=self.yolo_confidence, verbose=False)

            combined_mask = np.zeros(results[0].orig_shape, dtype=np.uint8)
            if results[0].masks:
                # Combine masks from all detected objects into one
                combined_mask = np.max(np.array([m.cpu().numpy() for m in results[0].masks.data]), axis=0)
                combined_mask = (combined_mask * 255).astype(np.uint8)

            filtered_mask = self.filter_drivable_mask(combined_mask)
            steering_angle_rad, viz_data = self.calculate_steering_from_area(filtered_mask)

            steer_msg = Float64()
            steer_msg.data = steering_angle_rad if steering_angle_rad is not None else 0.0
            self.steer_pub.publish(steer_msg)

            # Only perform visualization if there are subscribers
            if self.viz_pub.get_subscription_count() > 0:
                final_viz_angle = steering_angle_rad if steering_angle_rad is not None else 0.0
                self.publish_visualization(bev_image, filtered_mask, viz_data, final_viz_angle)
        except Exception:
            self.get_logger().error(f"Error in planning worker:\n{traceback.format_exc()}")

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
        Clean up the raw YOLO segmentation mask.

        This applies morphological closing to fill holes and then removes
        small, noisy components to get a clean drivable area mask.

        Args:
            bev_mask (np.ndarray): The raw binary mask from the model.

        Returns:
            np.ndarray: The cleaned binary mask.
        """
        f1 = morph_close(bev_mask, ksize=self._MORPH_KSIZE)
        return remove_small_components(f1, min_size=self._MIN_AREA_SIZE)

    def image_to_vehicle(self, pt_bev):
        """
        Convert a point from BEV image coordinates to vehicle coordinates.

        The vehicle's coordinate system is defined with the origin at the
        center of the rear axle, X-axis forward, and Y-axis to the left.

        Args:
            pt_bev (tuple): A tuple (u, v) representing the pixel coordinates
                            in the BEV image.

        Returns:
            tuple: A tuple (x, y) representing the coordinates in meters in the
                   vehicle's reference frame.
        """
        u, v = pt_bev
        # Convert pixel coordinates to meters relative to the camera
        y_cam = (self.bev_w / 2 - u) * self.m_per_pixel_x
        x_cam = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        # Adjust for the offset between the camera and the rear axle
        return x_cam - self.CAMERA_TO_REAR_AXLE_OFFSET, y_cam

    def calculate_steering_from_area(self, area_mask):
        """
        Calculate the steering angle from the drivable area mask.

        This is the core path planning logic. It defines an ROI, generates a
        Bézier curve path through the center of the detected area, finds a
        goal point on this path using the Pure Pursuit algorithm's lookahead
        distance, and computes the required steering angle to reach that point.

        Args:
            area_mask (np.ndarray): The cleaned binary mask of the drivable area.

        Returns:
            tuple: A tuple containing:
                - (float | None): The calculated steering angle in radians, or
                  None if no path is found.
                - (dict): A dictionary containing data for visualization.
        """
        roi_top_y = int(self.bev_h * self._ROI_TOP_Y_RATIO)
        roi_bottom_y = int(self.bev_h * self._ROI_BOTTOM_Y_RATIO) - 1
        roi_half_width = int((self.bev_w * self._ROI_WIDTH_RATIO) / 2)
        roi_center_x = self.bev_w // 2
        roi_left_x = roi_center_x - roi_half_width
        roi_right_x = roi_center_x + roi_half_width

        # Create a mask for the ROI
        roi_mask = np.zeros_like(area_mask)
        cv2.rectangle(roi_mask, (roi_left_x, roi_top_y), (roi_right_x, roi_bottom_y), 255, -1)
        roi_area_mask = cv2.bitwise_and(area_mask, area_mask, mask=roi_mask)

        is_detected = np.any(roi_area_mask)
        self.status_pub.publish(Bool(data=bool(is_detected)))
        viz_data = {'roi_coords': (roi_left_x, roi_top_y, roi_right_x, roi_bottom_y)}

        if not is_detected:
            return None, viz_data

        # === Bézier Curve Path Generation ===
        # 1. Define start (P0) and end (P3) points.
        p0 = (roi_center_x, roi_bottom_y)

        # Find the average x-coordinate of detected points at the top of the ROI
        top_points_y, top_points_x = np.where(roi_area_mask[roi_top_y:roi_top_y+10, :] > 0)
        if len(top_points_x) == 0:
             self.get_logger().warn("No points at ROI top. Using center as endpoint.", throttle_duration_sec=2)
             p3 = (roi_center_x, roi_top_y)
        else:
             p3 = (int(np.mean(top_points_x)), roi_top_y)

        # 2. Define control handles (P1, P2).
        roi_height = roi_bottom_y - roi_top_y
        handle_offset = int(roi_height * self._BEZIER_HANDLE_RATIO)
        p1 = (p0[0], p0[1] - handle_offset)
        p2 = (p3[0], p3[1] + handle_offset)

        # 3. Generate the Bézier curve.
        path_points = generate_bezier_curve(p0, p1, p2, p3)
        x_bev_coords, y_bev_coords = path_points[:, 0], path_points[:, 1]

        viz_data.update({'bezier_points': path_points, 'control_points': [p0, p1, p2, p3]})
        # =================================

        # Convert path from image coordinates to vehicle coordinates
        x_veh, y_veh = self.image_to_vehicle((x_bev_coords, y_bev_coords))
        dist_from_ego = np.sqrt(x_veh**2 + y_veh**2)

        # Find the point on the path that is closest to the lookahead distance
        goal_idx_candidates = np.where(np.abs(dist_from_ego - self.lookahead_distance) < 0.2)[0]

        goal_idx = None
        if len(goal_idx_candidates) > 0:
            # Select the best candidate
            goal_idx = goal_idx_candidates[np.argmin(np.abs(dist_from_ego[goal_idx_candidates] - self.lookahead_distance))]
        elif len(dist_from_ego) > 0:
            # If path is shorter than lookahead, use the farthest point on the path
            self.get_logger().warn(f"Path shorter than lookahead. Using farthest point.", throttle_duration_sec=2)
            goal_idx = np.argmax(dist_from_ego)

        if goal_idx is not None:
            actual_lookahead_dist = dist_from_ego[goal_idx]
            if actual_lookahead_dist > 0.1: # Threshold to avoid instability
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

        This image overlays the detected drivable area, the ROI, the
        Bézier path and its control points, the target goal point, and the
        calculated steering angle onto the BEV image.

        Args:
            bev_image (np.ndarray): The base BEV image.
            area_mask (np.ndarray): The binary mask of the drivable area.
            viz_data (dict): A dictionary containing all visualization elements
                             (ROI, path, points, etc.).
            steering_angle_rad (float): The final steering angle in radians.
        """
        viz_image = bev_image.copy()
        green_overlay = np.zeros_like(viz_image)
        green_overlay[area_mask > 0] = (0, 255, 0)
        viz_image = cv2.addWeighted(viz_image, 1, green_overlay, 0.4, 0)

        # Visualize ROI
        if 'roi_coords' in viz_data:
            x1, y1, x2, y2 = viz_data['roi_coords']
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Visualize Bézier curve and control points
        if 'bezier_points' in viz_data:
            path_points = viz_data['bezier_points'].astype(np.int32)
            cv2.polylines(viz_image, [path_points], isClosed=False, color=(255, 255, 0), thickness=3)
        if 'control_points' in viz_data:
            p0, p1, p2, p3 = viz_data['control_points']
            cv2.line(viz_image, p0, p1, (255, 0, 255), 2)
            cv2.line(viz_image, p2, p3, (255, 0, 255), 2)
            for p in viz_data['control_points']:
                cv2.circle(viz_image, p, 8, (255, 0, 255), -1)

        # Visualize goal point
        if 'goal_point_bev' in viz_data:
            cv2.circle(viz_image, viz_data['goal_point_bev'], 10, (0, 0, 255), -1)

        steer_text = f"Steer: {math.degrees(steering_angle_rad):.1f} deg"
        cv2.putText(viz_image, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        _, jpeg_buffer = cv2.imencode('.jpg', viz_image)
        viz_msg = CompressedImage(data=jpeg_buffer.tobytes(), format="jpeg")
        viz_msg.header.stamp = self.get_clock().now().to_msg()
        self.viz_pub.publish(viz_msg)

    def destroy_node(self):
        """
        Clean up resources before shutting down the node.

        This specifically ensures that the worker thread pool is shut down
        gracefully.
        """
        self.get_logger().info("Shutting down the planning thread pool.")
        self._is_shutting_down = True
        self.planning_executor.shutdown(wait=True)
        super().destroy_node()


def main(args=None):
    """The main entry point for the ROS2 node."""
    rclpy.init(args=args)
    node = YoloBevDrivableAreaNode()
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