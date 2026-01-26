#!/usr/env/bin python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for Distributed RealSense Vision Processing (Color Compressed Only).

This node is designed to run as a separate process for each camera to maximize
parallelism and minimize latency. It consumes only compressed color images,
performs 2D YOLO detection, and republishes the visualized result as a compressed image.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import sys
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO
import traceback

class DistributedRealsenseNode(Node):
    def __init__(self):
        super().__init__('distributed_realsense_node')
        
        # Parameters
        self.declare_parameter('camera_namespace', 'camera1')
        self.declare_parameter('model_path', './tracking.onnx')
        
        # Get and normalize namespace
        raw_ns = self.get_parameter('camera_namespace').get_parameter_value().string_value
        self.camera_namespace = raw_ns.strip('/')
        
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        self.get_logger().info(f"--- Distributed RealSense Node for /{self.camera_namespace}/ ---")
        
        # Initialize libraries
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Device: {self.device}")
        
        # Load Model
        try:
            self.model = YOLO(self.model_path, task='detect')
            self.get_logger().info(f"Allocated YOLO model from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model from '{self.model_path}': {e}")
            # Do NOT call self.destroy_node() here, just raise or exit
            sys.exit(1)
            
        # Topics - Precise construction as requested
        # Input: /{camera_namespace}/color/image_raw/compressed
        sub_topic = f'/{self.camera_namespace}/color/image_raw/compressed'
        
        # Output: /{camera_namespace}/viz/compressed
        pub_topic = f'/{self.camera_namespace}/viz/compressed'
        
        self.get_logger().info(f"Subscribing to: {sub_topic}")
        self.get_logger().info(f"Publishing to:  {pub_topic}")
        
        # Communication
        self.sub = self.create_subscription(
            CompressedImage,
            sub_topic,
            self.image_callback,
            10
        )
        self.pub = self.create_publisher(CompressedImage, pub_topic, 10)
        
    def image_callback(self, msg):
        try:
            # 1. Decode Compressed Image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().warn("Failed to decode image")
                return

            # 2. Inference
            results = self.model(cv_image, verbose=False)
            
            # 3. Draw & Annotate
            annotated_image = cv_image.copy()
            for r in results:
                for box in r.boxes.cpu().numpy():
                    # Check class if needed. Supply model usually has class 0 for target.
                    if int(box.cls[0]) == 0: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        
                        # Draw Box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # Label
                        label = f"Supply: {conf:.2f}"
                        cv2.putText(annotated_image, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 4. Encode & Publish
            viz_msg = CompressedImage()
            viz_msg.header = msg.header
            viz_msg.format = "jpeg"
            
            # Use default JPEG quality or specify params suitable for rqt streaming
            success, encoded_data = cv2.imencode('.jpg', annotated_image)
            if success:
                viz_msg.data = encoded_data.tobytes()
                self.pub.publish(viz_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}\n{traceback.format_exc()}")

def main(args=None):
    rclpy.init(args=args)
    node = DistributedRealsenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
