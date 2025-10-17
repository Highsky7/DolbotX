#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: bev_recorder.py
# AUTHOR: DolbotX Team
# DESCRIPTION:
# BEV data generation and recording utility.
# 1. Reuses the exact BEV transform from 'onnx_path_planning_pp.py' for full data parity.
# 2. Writes high-quality video using the standard MP4V codec.
# 3. Safely releases the video file on shutdown to prevent corruption.
# 4. Exposes parameters via the ROS 2 parameter system (bev_param_file, output_path, fps).
# 5. Subscribes with a Best Effort QoS profile suited for real-time image streams.

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import traceback

from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class BEVRecorderNode(Node):
    """
    Subscribe to a camera topic, convert frames to BEV (Bird's-Eye View),
    and persist the results to a video file.
    """
    def __init__(self):
        super().__init__('bev_recorder_node')
        self.get_logger().info("--- BEV data generation and recording node ---")
        
        # === Declare and fetch parameters ===
        # Parameters can be overridden from launch files or the command line.
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('output_path', '~/bev_output1.mp4')
        self.declare_parameter('fps', 30.0)
        
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        output_path_str = self.get_parameter('output_path').get_parameter_value().string_value
        # Expand '~' to the user home directory.
        self.output_path = os.path.expanduser(output_path_str) 
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        
        self.get_logger().info(f"BEV parameter file: {bev_param_file}")
        self.get_logger().info(f"Video output path: {self.output_path}")
        self.get_logger().info(f"Video FPS: {self.fps}")

        self.video_writer = None

        # === Load BEV transform parameters ===
        # Mirrors the logic from 'onnx_path_planning_pp.py' for guaranteed parity.
        try:
            self.get_logger().info(f"Loading BEV parameters from: {bev_param_file}")
            bev_params = np.load(bev_param_file)
            self.src_points = bev_params['src_points']
            self.dst_points = bev_params['dst_points']
            self.bev_h = int(bev_params['warp_h'])
            self.bev_w = int(bev_params['warp_w'])
            self.M_bev = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.get_logger().info("✅ BEV transformation matrix calculated successfully.")
        except FileNotFoundError:
            self.get_logger().error(f"FATAL: BEV parameter file not found at '{bev_param_file}'. Shutting down.")
            rclpy.shutdown()
            return
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to load BEV parameters: {e}")
            rclpy.shutdown()
            return

        # === Initialize the video writer ===
        # MP4V is a broadly supported codec that yields high-quality output.
        try:
            # Create the output directory on demand.
            output_dir = os.path.dirname(self.output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.get_logger().info(f"Created output directory: {output_dir}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.bev_w, self.bev_h))
            if not self.video_writer.isOpened():
                raise IOError("Cannot open video writer.")
            self.get_logger().info(f"✅ Video writer initialized. Recording to '{self.output_path}'")
        except (IOError, Exception) as e:
            self.get_logger().error(f"FATAL: Failed to initialize VideoWriter: {e}")
            self.get_logger().error("Please check file permissions and codec support.")
            rclpy.shutdown()
            return

        # === Configure the image subscriber ===
        # Reuse the planner QoS settings to keep timing aligned.
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        img_topic = '/camera3/image_raw/compressed'
        self.img_sub = self.create_subscription(
            CompressedImage, 
            img_topic, 
            self.image_callback, 
            qos_profile_sensor_data
        )
        self.get_logger().info(f"✅ Node initialized. Subscribing to '{img_topic}'.")

    def image_callback(self, compressed_img_msg):
        """
        Convert each incoming image into BEV space and append it to the video.
        """
        if self.video_writer is None or not self.video_writer.isOpened():
            self.get_logger().warn("Video writer is not ready. Skipping frame.", throttle_duration_sec=5)
            return

        try:
            # 1. Decode the compressed image into an OpenCV array.
            np_arr = np.frombuffer(compressed_img_msg.data, np.uint8)
            cv_color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_color_image is None:
                self.get_logger().warn("Failed to decode compressed image.", throttle_duration_sec=5)
                return

            # 2. Run the BEV warp (core logic).
            bev_image = cv2.warpPerspective(
                cv_color_image, 
                self.M_bev, 
                (self.bev_w, self.bev_h), 
                flags=cv2.INTER_LINEAR
            )

            # 3. Append the warped BEV frame to the video file.
            self.video_writer.write(bev_image)

        except Exception:
            self.get_logger().error(f"Error in image_callback:\n{traceback.format_exc()}")

    def destroy_node(self):
        """
        Close the video file safely when the node shuts down.
        This prevents corruption of the recorded dataset.
        """
        self.get_logger().info("Shutting down node...")
        if self.video_writer and self.video_writer.isOpened():
            self.get_logger().info(f"Finalizing video file at '{self.output_path}'...")
            self.video_writer.release()
            self.get_logger().info("✅ Video file saved successfully.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    # Spin only if the BEV parameters and video writer were initialized correctly.
    node = BEVRecorderNode()
    if rclpy.ok() and hasattr(node, 'M_bev') and node.video_writer is not None:
        try: 
            rclpy.spin(node)
        except KeyboardInterrupt: 
            node.get_logger().info("Keyboard interrupt detected.")
        finally: 
            # Tear down the node and release all resources.
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
    main()
