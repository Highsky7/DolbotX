#!/usr/env/bin python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for Distributed RealSense Vision Processing (Color Compressed Relay Only).

This node runs as a separate process for each camera.
It subscribes to compressed color images and republishes them to a visualization topic.
This version has NO dependency on Torch or YOLO, functioning purely as a lightweight relay
to ensure smooth visualization in rqt_image_view.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class DistributedRealsenseNode(Node):
    def __init__(self):
        super().__init__('distributed_realsense_node')
        
        # Parameters
        self.declare_parameter('camera_namespace', 'camera1')
        
        # Get and normalize namespace
        raw_ns = self.get_parameter('camera_namespace').get_parameter_value().string_value
        self.camera_namespace = raw_ns.strip('/')
        
        self.get_logger().info(f"--- Distributed RealSense Relay Node for /{self.camera_namespace}/ ---")
        
        # Topics
        # Input: /{camera_namespace}/color/image_raw/compressed
        sub_topic = f'/{self.camera_namespace}/color/image_raw/compressed'
        
        # Output: /{camera_namespace}/viz/compressed
        pub_topic = f'/{self.camera_namespace}/viz/compressed'
        
        self.get_logger().info(f"Subscribing to: {sub_topic}")
        self.get_logger().info(f"Relaying to:    {pub_topic}")
        
        # Communication
        self.sub = self.create_subscription(
            CompressedImage,
            sub_topic,
            self.image_callback,
            10
        )
        self.pub = self.create_publisher(CompressedImage, pub_topic, 10)
        
    def image_callback(self, msg):
        """
        Directly republish the compressed image message to the viz topic.
        This avoids decoding/encoding overhead since no drawing is required.
        """
        try:
            # Create a new message (or just publish the same one, but safer to copy shell)
            viz_msg = CompressedImage()
            viz_msg.header = msg.header
            viz_msg.format = msg.format
            viz_msg.data = msg.data
            
            self.pub.publish(viz_msg)
                
        except Exception as e:
            self.get_logger().error(f"Relay error: {e}")

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
