#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for Visual Marker Detection from Multiple USB Cameras.

This node is dedicated to recognizing various visual markers using two separate
USB cameras. It leverages a YOLO object detection model ('vision_marker2.onnx')
to perform inference.

The detection results are visualized with enhanced readability and published
to separate topics for each camera stream.

Key Visualization Enhancements:
- Assigns a unique color to each marker class for immediate identification.
- Adds a filled background to text labels for maximum readability under all
  lighting conditions.
- Uses bold, black text for high contrast and visibility.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge


class VisionMarkerDetectorNode(Node):
    """
    Detects visual markers from two USB cameras and publishes visualizations.

    This node loads a YOLO model to detect a predefined set of visual markers.
    It subscribes to two separate compressed image topics, processes them in
    parallel using a thread pool, and publishes annotated images showing the
    detections.
    """

    def __init__(self):
        """
        Initialize the VisionMarkerDetectorNode.

        This sets up the node, loads the YOLO model, defines class names and
        colors, creates publishers for visualization, and sets up subscribers
        for two camera feeds.
        """
        super().__init__('vision_marker_detector_node')
        self.get_logger().info("--- Vision Marker Detection Node (Enhanced by Hinton) ---")

        self.usb_cam_locks = {'cam1': threading.Lock(), 'cam2': threading.Lock()}

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        self.use_half = self.device == 'cuda'

        try:
            # Load the vision marker detection model
            self.declare_parameter('marker_model_path', './vision_marker2.onnx')
            marker_model_path = self.get_parameter('marker_model_path').get_parameter_value().string_value
            self.marker_model = YOLO(marker_model_path, task='detect')
            self.marker_class_names = ['A', 'E', 'Heart', 'K', 'M', 'O', 'R', 'Y']

            # Define unique colors for each class for visualization (BGR format)
            self.marker_colors = {
                'A': (255, 0, 0),      # Blue
                'E': (0, 255, 0),      # Green
                'Heart': (0, 0, 255),    # Red
                'K': (255, 255, 0),    # Cyan
                'M': (255, 0, 255),    # Magenta
                'O': (0, 165, 255),    # Orange
                'R': (128, 0, 128),    # Purple
                'Y': (0, 255, 255)     # Yellow
            }
            self.get_logger().info("✅ Vision Marker ONNX model loaded successfully.")
            self.get_logger().info("✅ Enhanced visualization color palette is active.")

        except Exception as e:
            self.get_logger().error(f"Failed to load Vision Marker YOLO model: {e}")
            self.destroy_node(); return

        # Setup publishers (for visualization only)
        self.usb_cam1_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam1_marker/viz/compressed', 10)
        self.usb_cam2_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam2_marker/viz/compressed', 10)

        self.yolo_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='marker_worker')
        self._is_shutting_down = False

        # Setup subscribers
        usb_cam1_topic = 'camera1/image_raw/compressed'
        self.usb_cam1_sub = self.create_subscription(
            CompressedImage, usb_cam1_topic, lambda msg: self.usb_cam_callback(msg, 'cam1'), 10)

        usb_cam2_topic = 'camera2/image_raw/compressed'
        self.usb_cam2_sub = self.create_subscription(
            CompressedImage, usb_cam2_topic, lambda msg: self.usb_cam_callback(msg, 'cam2'), 10)

        self.get_logger().info("✅ Vision Marker Node initialized successfully.")

    def usb_cam_callback(self, compressed_msg, camera_id):
        """
        Handle incoming compressed image messages from a USB camera.

        This callback uses a non-blocking lock to prevent queuing up frames if
        the processing for a given camera is backlogged. It submits the image
        data to a thread pool for processing.

        Args:
            compressed_msg (CompressedImage): The ROS message containing the image.
            camera_id (str): The identifier for the camera (e.g., 'cam1').
        """
        if self._is_shutting_down: return
        lock = self.usb_cam_locks[camera_id]
        if lock.acquire(blocking=False):
            try:
                self.yolo_executor.submit(self._process_usb_cam_data, compressed_msg, camera_id)
            finally:
                pass  # The lock is released in the worker thread
        else:
            self.get_logger().warn(f"Dropping a frame from {camera_id}, processing is busy.", throttle_duration_sec=1)

    def _process_usb_cam_data(self, compressed_msg, camera_id):
        """
        Process a single image frame in a worker thread.

        This function decodes the image, runs the YOLO model for marker
        detection, creates a visualization image, and publishes it. The lock
        for the camera is released upon completion.

        Args:
            compressed_msg (CompressedImage): The ROS message containing the image.
            camera_id (str): The identifier for the camera.
        """
        lock = self.usb_cam_locks[camera_id]
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().warn(f"Failed to decompress USB cam image from {camera_id}.")
                return

            # Detect markers
            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False, half=self.use_half)

            # Create and publish visualization image
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            viz_publisher = self.usb_cam1_viz_pub if camera_id == 'cam1' else self.usb_cam2_viz_pub
            self.publish_compressed_viz(viz_publisher, annotated_image)

        except Exception as e:
            self.get_logger().error(f"Error in Vision Marker USB Cam worker ({camera_id}): {e}\n{traceback.format_exc()}")
        finally:
            lock.release()

    def publish_compressed_viz(self, publisher, cv_image):
        """
        Encode an image as JPEG and publish it as a CompressedImage message.

        Args:
            publisher (Publisher): The ROS publisher to use.
            cv_image (np.ndarray): The OpenCV image to publish.
        """
        msg = CompressedImage(format="jpeg")
        msg.header.stamp = self.get_clock().now().to_msg()
        success, encoded_image = cv2.imencode('.jpg', cv_image)
        if success:
            msg.data = encoded_image.tobytes()
            publisher.publish(msg)

    def draw_marker_detections(self, image, results):
        """
        Draw detection bounding boxes and labels on an image with enhanced styling.

        This enhanced visualization function:
        1. Applies a unique color to each marker class.
        2. Adds a filled background rectangle to the text label for readability.
        3. Uses bold, black text for maximum visibility.

        Args:
            image (np.ndarray): The image to draw on.
            results: The YOLO detection results.

        Returns:
            np.ndarray: The image with annotations drawn on it.
        """
        for r in results:
            for box in r.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])

                if cls_id >= len(self.marker_class_names):
                    continue

                label = self.marker_class_names[cls_id]
                # Get the unique color for the class, defaulting to gray
                color = self.marker_colors.get(label, (128, 128, 128))

                # 1. Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # 2. Prepare the text label and background
                text = f"{label}: {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6

                # --- Enhanced font thickness ---
                font_thickness = 2

                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

                # Calculate coordinates for the text background rectangle
                # Display above the box if there's space, otherwise below
                if y1 - text_h - baseline > 0:
                    text_bg_y1 = y1 - text_h - baseline - 2
                    text_bg_y2 = y1
                    text_y = y1 - baseline // 2 - 2
                else:
                    text_bg_y1 = y2
                    text_bg_y2 = y2 + text_h + baseline + 2
                    text_y = y2 + text_h

                cv2.rectangle(image, (x1, text_bg_y1), (x1 + text_w, text_bg_y2), color, cv2.FILLED)

                # --- Use black text color for maximum visibility ---
                cv2.putText(image, text, (x1, text_y), font, font_scale, (0, 0, 0), font_thickness)

        return image

    def destroy_node(self):
        """Cleanly shut down the node and its resources."""
        self.get_logger().info("Shutting down the thread pool.")
        self._is_shutting_down = True
        self.yolo_executor.shutdown(wait=True)
        super().destroy_node()


def main(args=None):
    """The main entry point for the node."""
    rclpy.init(args=args)
    node = VisionMarkerDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()