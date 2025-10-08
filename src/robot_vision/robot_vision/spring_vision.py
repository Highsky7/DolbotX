#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for Friendly and Enemy Uniform Detection.

This node is responsible for identifying friendly ('ROKA') and enemy ('Enemy')
uniforms using two separate USB cameras. It performs inference using the
'roka_enemy.onnx' model and publishes an LED control signal to the
'/led_control' topic based on the identification results.
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


class RokaEnemyDetectorNode(Node):
    """
    Detects ROKA (friendly) and Enemy uniforms from two USB cameras.

    This node loads a YOLO model to detect uniforms. It processes two camera
    feeds in parallel, determines if friend or foe is present, publishes a
    corresponding command to control an LED, and also publishes annotated
    visualization images.
    """

    def __init__(self):
        """
        Initialize the RokaEnemyDetectorNode.

        This sets up the node, loads the YOLO model, creates publishers for
        LED control and visualization, and sets up subscribers for two
        camera feeds with a multi-threaded executor.
        """
        super().__init__('roka_enemy_detector_node')
        self.get_logger().info("--- ROKA & Enemy Detection Node ---")

        self.usb_cam_locks = {'cam1': threading.Lock(), 'cam2': threading.Lock()}

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        self.use_half = self.device == 'cuda'

        try:
            # Load the friend/foe detection model
            self.declare_parameter('roka_enemy_model_path', './roka_enemy.onnx')
            roka_enemy_model_path = self.get_parameter('roka_enemy_model_path').get_parameter_value().string_value
            self.roka_enemy_model = YOLO(roka_enemy_model_path, task='detect')

            # IMPORTANT: This class name order must exactly match the order
            # in which the roka_enemy.onnx model was trained.
            # e.g., if class 0 is Enemy and class 1 is ROKA.
            self.roka_enemy_class_names = ['Enemy', 'ROKA']
            self.get_logger().info("✅ ROKA/Enemy ONNX model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load ROKA/Enemy YOLO model: {e}")
            self.destroy_node(); return

        # Setup publishers
        self.led_pub = self.create_publisher(String, '/led_control', 10)
        self.usb_cam1_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam1_roka_enemy/viz/compressed', 10)
        self.usb_cam2_viz_pub = self.create_publisher(CompressedImage, '/unified_vision/usb_cam2_roka_enemy/viz/compressed', 10)

        self.yolo_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='roka_enemy_worker')
        self._is_shutting_down = False

        # Setup subscribers
        usb_cam1_topic = 'camera1/image_raw/compressed'
        self.usb_cam1_sub = self.create_subscription(
            CompressedImage, usb_cam1_topic, lambda msg: self.usb_cam_callback(msg, 'cam1'), 10)

        usb_cam2_topic = 'camera2/image_raw/compressed'
        self.usb_cam2_sub = self.create_subscription(
            CompressedImage, usb_cam2_topic, lambda msg: self.usb_cam_callback(msg, 'cam2'), 10)

        self.get_logger().info("✅ ROKA & Enemy Node initialized successfully.")

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

        This function decodes the image, runs the YOLO model for uniform
        detection, determines the LED status, creates a visualization image,
        and publishes the results.

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

            # Detect friend or foe
            results = self.roka_enemy_model(cv_image, conf=0.5, iou=0.45, verbose=False, half=self.use_half)
            roka_found, enemy_found = False, False
            for r in results:
                for box in r.boxes.cpu().numpy():
                    cls_id = int(box.cls[0])
                    if cls_id < len(self.roka_enemy_class_names):
                        label = self.roka_enemy_class_names[cls_id]
                        if label == 'ROKA':
                            roka_found = True
                        elif label == 'Enemy':
                            enemy_found = True

            # Publish LED control message (ROKA has priority)
            led_data = "roka" if roka_found else "enemy" if enemy_found else "none"
            self.led_pub.publish(String(data=led_data))

            # Create and publish visualization image
            annotated_image = self.draw_detections(cv_image, results)
            viz_publisher = self.usb_cam1_viz_pub if camera_id == 'cam1' else self.usb_cam2_viz_pub
            self.publish_compressed_viz(viz_publisher, annotated_image)

        except Exception as e:
            self.get_logger().error(f"Error in ROKA/Enemy USB Cam worker ({camera_id}): {e}\n{traceback.format_exc()}")
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

    def draw_detections(self, image, results):
        """
        Draw detection bounding boxes and labels on an image.

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
                label = self.roka_enemy_class_names[cls_id] if cls_id < len(self.roka_enemy_class_names) else "Unknown"
                color = (0, 255, 0) if label == 'ROKA' else (0, 0, 255) if label == 'Enemy' else (200, 200, 200)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
    node = RokaEnemyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()