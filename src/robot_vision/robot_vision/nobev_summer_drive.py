#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for Fused Drivable Area Detection without BEV Transformation.

This script is tailored for the "Summer" mission profile. It operates directly
in the camera's image space, avoiding a Bird's-Eye-View (BEV) transformation.
It uses three separate YOLO segmentation models to identify multiple types of
drivable surfaces: standard track, sand, and stone. The outputs of these
models are fused into a single unified drivable area mask.

Control is achieved using a simple proportional controller based on the
centroid of the fused drivable area within a predefined Region of Interest (ROI).

Key Features:
- **No BEV:** All processing is done on the raw camera image.
- **Triple-Model Fusion:** Combines results from three distinct YOLO models
  (drivable area, sand, stone) into one mask.
- **Proportional Steering Control:** Simple steering logic based on the
  centroid of the detected drivable area.
- **Region of Interest (ROI):** Focuses processing on the lower part of the
  image to improve stability.
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


class YoloTripleFusedDrivableAreaNode(Node):
    """
    A ROS2 node for fused drivable area detection in camera space.

    This node loads three separate YOLO models, fuses their segmentation
    results, and computes a steering command based on the centroid of the
    combined drivable area.
    """
    _MORPH_KSIZE = 7
    _MIN_AREA_SIZE = 15000  # This may need adjustment for the original image size.

    # ROI for path detection (as a ratio of image dimensions)
    _ROI_TOP_Y_RATIO = 0.5
    _ROI_BOTTOM_Y_RATIO = 1.0
    _ROI_WIDTH_RATIO = 1.0

    def __init__(self):
        """
        Initialize the YoloTripleFusedDrivableAreaNode.

        This sets up the node, loads the three YOLO models (drivable area,
        sand, stone), and creates the necessary publishers and subscribers.
        """
        super().__init__('yolo_triple_fused_drivable_area_node')
        self.get_logger().info("--- YOLO Triple-Fused Drivable Area Node (BEV-Removed, Camera-Space Control with ROI) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Remove BEV-related parameters, add steering_gain
        self.declare_parameter('drive_area_model_path', './drive_area2.onnx')
        self.declare_parameter('sand_model_path', './sand.onnx')
        self.declare_parameter('stone_model_path', './stone.onnx')
        self.declare_parameter('drive_area_confidence', 0.5)
        self.declare_parameter('sand_confidence', 0.5)
        self.declare_parameter('stone_confidence', 0.5)
        self.declare_parameter('steering_gain', 0.003)  # Proportional gain (Kp) for steering

        drive_area_model_path = self.get_parameter('drive_area_model_path').get_parameter_value().string_value
        sand_model_path = self.get_parameter('sand_model_path').get_parameter_value().string_value
        stone_model_path = self.get_parameter('stone_model_path').get_parameter_value().string_value
        self.drive_area_confidence = self.get_parameter('drive_area_confidence').get_parameter_value().double_value
        self.sand_confidence = self.get_parameter('sand_confidence').get_parameter_value().double_value
        self.stone_confidence = self.get_parameter('stone_confidence').get_parameter_value().double_value
        self.steering_gain = self.get_parameter('steering_gain').get_parameter_value().double_value

        try:
            self.drive_area_model = YOLO(drive_area_model_path, task='segment')
            self.sand_model = YOLO(sand_model_path, task='segment')
            self.stone_model = YOLO(stone_model_path, task='segment')
            self.get_logger().info(f"✅ Triple ONNX models and all resources loaded on [{self.device}].")

        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to load resources: {e}")
            rclpy.shutdown()
            return

        self.planning_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='planning_worker')
        self._is_shutting_down = False

        self.steer_pub = self.create_publisher(Float64, '/steering_angle', 10)
        self.viz_pub = self.create_publisher(CompressedImage, '/path_planning/drivable_area/viz/compressed', 10)
        self.status_pub = self.create_publisher(Bool, '/path_planning/drivable_area/status', 10)

        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        logitech_img_topic = '/camera3/image_raw/compressed'
        self.img_sub = self.create_subscription(
            CompressedImage,
            logitech_img_topic,
            self.planning_callback,
            qos_profile_sensor_data
        )
        self.get_logger().info(f"✅ Node initialized. Subscribing to '{logitech_img_topic}' with RELIABLE QoS.")

    def planning_callback(self, compressed_img_msg):
        """
        Handle incoming compressed image messages and submit for processing.

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
        Process a single image frame in a worker thread.

        This function decodes the image, runs three YOLO models, fuses their
        masks, calculates the steering angle, and publishes results.

        Args:
            compressed_data_buffer (bytes): The raw byte data from the message.
        """
        try:
            np_arr = np.frombuffer(compressed_data_buffer, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_color is None:
                self.get_logger().warn("Failed to decode image.")
                return

            # Use the original image directly, no BEV transformation
            image_for_model = cv_color

            drive_area_results = self.drive_area_model(image_for_model, conf=self.drive_area_confidence, verbose=False)
            sand_results = self.sand_model(image_for_model, conf=self.sand_confidence, verbose=False)
            stone_results = self.stone_model(image_for_model, conf=self.stone_confidence, verbose=False)

            # Create and fuse masks from the three models
            drive_area_mask = np.zeros(drive_area_results[0].orig_shape, dtype=np.uint8)
            if drive_area_results[0].masks is not None and len(drive_area_results[0].masks.data) > 0:
                drive_area_mask = np.max(np.array([m.cpu().numpy() for m in drive_area_results[0].masks.data]), axis=0)
                drive_area_mask = (drive_area_mask * 255).astype(np.uint8)

            sand_mask = np.zeros(sand_results[0].orig_shape, dtype=np.uint8)
            if sand_results[0].masks is not None and len(sand_results[0].masks.data) > 0:
                sand_mask = np.max(np.array([m.cpu().numpy() for m in sand_results[0].masks.data]), axis=0)
                sand_mask = (sand_mask * 255).astype(np.uint8)

            stone_mask = np.zeros(stone_results[0].orig_shape, dtype=np.uint8)
            if stone_results[0].masks is not None and len(stone_results[0].masks.data) > 0:
                stone_mask = np.max(np.array([m.cpu().numpy() for m in stone_results[0].masks.data]), axis=0)
                stone_mask = (stone_mask * 255).astype(np.uint8)

            temp_mask = cv2.bitwise_or(drive_area_mask, sand_mask)
            unified_mask = cv2.bitwise_or(temp_mask, stone_mask)

            filtered_mask = self.filter_drivable_mask(unified_mask)
            steering_angle, viz_data = self.calculate_steering_from_area(filtered_mask)

            steer_msg = Float64()
            if steering_angle is not None:
                steer_msg.data = steering_angle
            else:
                steer_msg.data = 0.0
            self.steer_pub.publish(steer_msg)

            if self.viz_pub.get_subscription_count() > 0:
                final_viz_angle = steering_angle if steering_angle is not None else 0.0
                self.publish_visualization(cv_color, filtered_mask, viz_data, final_viz_angle)

        except Exception:
            self.get_logger().error(f"Error in planning worker:\n{traceback.format_exc()}")

    def filter_drivable_mask(self, mask):
        """
        Clean up the raw fused segmentation mask.

        Args:
            mask (np.ndarray): The raw binary mask from the fusion.

        Returns:
            np.ndarray: The cleaned binary mask.
        """
        f1 = morph_close(mask, ksize=self._MORPH_KSIZE)
        f2 = remove_small_components(f1, min_size=self._MIN_AREA_SIZE)
        return f2

    def calculate_steering_from_area(self, area_mask):
        """
        Calculate steering angle based on the centroid of the fused area.

        This method is identical to the one in `nobev_springfall_drive.py`.

        Args:
            area_mask (np.ndarray): The cleaned binary mask of the drivable area.

        Returns:
            tuple: A tuple containing the steering angle and visualization data.
        """
        h, w = area_mask.shape[:2]

        # Calculate ROI coordinates based on image dimensions
        roi_top_y = int(h * self._ROI_TOP_Y_RATIO)
        roi_bottom_y = int(h * self._ROI_BOTTOM_Y_RATIO)
        roi_half_width = int((w * self._ROI_WIDTH_RATIO) / 2)
        roi_center_x = w // 2
        roi_left_x = roi_center_x - roi_half_width
        roi_right_x = roi_center_x + roi_half_width

        roi_mask = np.zeros_like(area_mask)
        cv2.rectangle(roi_mask, (roi_left_x, roi_top_y), (roi_right_x, roi_bottom_y), 255, -1)

        roi_area_mask = cv2.bitwise_and(area_mask, area_mask, mask=roi_mask)

        viz_data = {'roi_coords': (roi_left_x, roi_top_y, roi_right_x, roi_bottom_y), 'centroid': None}

        is_detected = np.any(roi_area_mask)
        self.status_pub.publish(Bool(data=bool(is_detected)))
        if not is_detected:
            return None, viz_data

        # Find the largest contour (drivable area) within the ROI
        contours, _ = cv2.findContours(roi_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, viz_data

        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, viz_data

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        viz_data['centroid'] = (cx, cy)

        # Calculate the error between the centroid and the image center
        error = cx - (w // 2)

        # Calculate steering angle using proportional control
        steering_angle = -self.steering_gain * float(error)

        return steering_angle, viz_data

    def publish_visualization(self, image, area_mask, viz_data, steering_angle_rad):
        """
        Create and publish a visualization image.

        Args:
            image (np.ndarray): The base camera image.
            area_mask (np.ndarray): The binary mask of the drivable area.
            viz_data (dict): A dictionary containing visualization elements.
            steering_angle_rad (float): The final steering angle in radians.
        """
        viz_image = image.copy()

        green_overlay = np.zeros_like(viz_image)
        green_overlay[area_mask > 0] = (0, 255, 0)
        viz_image = cv2.addWeighted(viz_image, 1, green_overlay, 0.4, 0)

        if 'roi_coords' in viz_data:
            x1, y1, x2, y2 = viz_data['roi_coords']
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if viz_data.get('centroid') is not None:
            cv2.circle(viz_image, viz_data['centroid'], 10, (0, 0, 255), -1)

        steer_deg = math.degrees(steering_angle_rad)
        steer_text = f"Steer: {steer_deg:.1f} deg"
        cv2.putText(viz_image, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        _, jpeg_buffer = cv2.imencode('.jpg', viz_image)
        viz_msg = CompressedImage()
        viz_msg.header.stamp = self.get_clock().now().to_msg()
        viz_msg.format = "jpeg"
        viz_msg.data = jpeg_buffer.tobytes()

        self.viz_pub.publish(viz_msg)

    def destroy_node(self):
        """Cleanly shut down the node and its resources."""
        self.get_logger().info("Shutting down the planning thread pool.")
        self._is_shutting_down = True
        self.planning_executor.shutdown(wait=True)
        super().destroy_node()


def main(args=None):
    """The main entry point for the node."""
    rclpy.init(args=args)
    node = YoloTripleFusedDrivableAreaNode()
    if rclpy.ok():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Keyboard interrupt, shutting down.")
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()