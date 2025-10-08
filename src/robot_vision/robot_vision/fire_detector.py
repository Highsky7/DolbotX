#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for Fire and Door Handle Detection.

This node subscribes to a compressed image topic from a USB camera and uses
two separate YOLO models to detect 'fire' and 'door_handle' objects.
It then annotates the image with bounding boxes for both types of detections
and publishes the result as a compressed image for visualization.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class FireAndDoorDetector(Node):
    """
    A ROS2 node that detects fire and door handles in an image stream.

    This node loads two YOLO models, one for fire detection and one for door
    handle detection. It processes incoming images, runs both models, draws
t   he results on the image, and publishes the annotated image.
    """

    def __init__(self):
        """
        Initialize the FireAndDoorDetector node.

        This sets up the node, loads the two YOLO models, and creates the
        necessary publishers and subscribers with appropriate QoS settings.
        """
        super().__init__('fire_and_door_detector')
        self.get_logger().info('🔥🚪 Start Fire and Door Handle Detector.')

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using compute device: {self.device}")

        # Define QoS profile for sensor data (e.g., images)
        self.qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Declare parameters for image processing size
        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value

        try:
            # Load fire detection model
            self.declare_parameter('fire_detector_model_path', './fire.pt')
            fire_detector_model_path = self.get_parameter('fire_detector_model_path').get_parameter_value().string_value
            self.fire_detector_model = YOLO(fire_detector_model_path).to(self.device)
            self.fire_detector_class_names = ['Fire']

            # Load door handle detection model
            self.declare_parameter('door_handle_model_path', './door_handle.pt')
            door_handle_model_path = self.get_parameter('door_handle_model_path').get_parameter_value().string_value
            self.door_handle_model = YOLO(door_handle_model_path).to(self.device)
            self.door_handle_class_names = ['door_handle']

        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO models: {e}")
            self.destroy_node()
            return

        # Declare publisher with QoS profile
        self.detector_viz_pub = self.create_publisher(
            CompressedImage,
            'fire_and_door_detector/compressed',
            qos_profile=self.qos_profile_sensor_data
        )

        # Declare subscriber with QoS profile
        usb_cam_topic = 'camera1/image_compressed'
        self.usb_cam_sub = self.create_subscription(
            CompressedImage,
            usb_cam_topic,
            self.usb_cam_callback,
            qos_profile=self.qos_profile_sensor_data
        )

        self.get_logger().info("✅ Fire and Door Handle Detector Node initialized successfully.")

    def usb_cam_callback(self, compressed_msg):
        """
        Process an incoming image message from the USB camera.

        This function decodes the compressed image, runs both the fire and
        door handle detection models, draws the results, and publishes the
        final annotated image.

        Args:
            compressed_msg (CompressedImage): The ROS message with image data.
        """
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 1. Run fire detection model and draw results
            results_fire = self.fire_detector_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_fire_detections(cv_image, results_fire)

            # 2. Run door handle detection model and draw results
            results_door_handle = self.door_handle_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_door_handle_detections(annotated_image, results_door_handle)

            # 3. Publish the final annotated image
            self.publish_compressed_viz(self.detector_viz_pub, annotated_image)
        except Exception as e:
            self.get_logger().error(f"Error in USB Cam callback: {e}")

    def publish_compressed_viz(self, publisher, cv_image):
        """
        Publish an OpenCV image as a compressed ROS message.

        Args:
            publisher (Publisher): The publisher to use for sending the message.
            cv_image (np.ndarray): The image to be published.
        """
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
        publisher.publish(msg)

    def draw_fire_detections(self, image, results):
        """
        Draw bounding boxes for fire detections on the image.

        Args:
            image (np.ndarray): The image to draw on.
            results: The results from the YOLO fire detection model.

        Returns:
            np.ndarray: The image with fire detections annotated.
        """
        for result in results:
            for box in result.boxes.cpu().numpy():
                cls_id = int(box.cls[0])
                if cls_id < len(self.fire_detector_class_names):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = self.fire_detector_class_names[cls_id]
                    # Draw fire detections in red
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return image

    def draw_door_handle_detections(self, image, results):
        """
        Draw bounding boxes for door handle detections on the image.

        Args:
            image (np.ndarray): The image to draw on.
            results: The results from the YOLO door handle detection model.

        Returns:
            np.ndarray: The image with door handle detections annotated.
        """
        for result in results:
            for box in result.boxes.cpu().numpy():
                cls_id = int(box.cls[0])
                if cls_id < len(self.door_handle_class_names):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = self.door_handle_class_names[cls_id]
                    # Draw door handle detections in blue
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image


def main(args=None):
    """Main function to initialize and run the ROS2 node."""
    rclpy.init(args=args)
    node = FireAndDoorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()