#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: unified_recorder.py
# AUTHOR: DolbotX Team
# DESCRIPTION:
# Unified multi-camera data recording node.
# 1. Subscribes to multiple camera topics from a single ROS 2 node to keep capture in sync.
# 2. Applies BEV (Bird's-Eye View) conversion for a specific topic (camera3) while recording.
# 3. Stores camera topics and output paths as parameter lists for easy launch-time customization.
# 4. Releases all video writers cleanly on shutdown to preserve recordings.
# 5. Mixes CvBridge decoding with direct NumPy processing for flexibility and stability.

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import traceback
from datetime import datetime
from functools import partial

from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge

class UnifiedRecorderNode(Node):
    """
    Unified recorder that subscribes to multiple camera topics and writes
    synchronized video files, applying BEV transformation to a designated feed.
    """
    def __init__(self):
        super().__init__('unified_recorder_node')
        self.get_logger().info("--- Unified multi-camera recorder ---")

        # === 1. Declare and fetch parameters ===
        self.declare_parameter('camera_topics', 
            ['/camera1/image_raw/compressed', 
             '/camera2/image_raw/compressed', 
             '/camera/color/image_raw/compressed',
             '/camera3/image_raw/compressed'])
        self.declare_parameter('output_dir', '~/ros2_recordings/unified')
        self.declare_parameter('output_filenames', 
            ['camera1.mp4', 'camera2.mp4', 'realsense.mp4', 'bev_output.mp4'])
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('bev_target_topic', '/camera3/image_raw/compressed')

        self.camera_topics = self.get_parameter('camera_topics').get_parameter_value().string_array_value
        output_dir_str = self.get_parameter('output_dir').get_parameter_value().string_value
        self.output_filenames = self.get_parameter('output_filenames').get_parameter_value().string_array_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        self.bev_target_topic = self.get_parameter('bev_target_topic').get_parameter_value().string_value

        self.output_dir = os.path.expanduser(output_dir_str)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.get_logger().info(f"Created output directory: {self.output_dir}")

        if len(self.camera_topics) != len(self.output_filenames):
            self.get_logger().error("FATAL: The number of camera_topics must match the number of output_filenames. Shutting down.")
            rclpy.shutdown()
            return
            
        self.get_logger().info(f"Subscribing to topics: {self.camera_topics}")
        self.get_logger().info(f"Outputting to files: {self.output_filenames}")
        self.get_logger().info(f"Video FPS set to: {self.fps}")

        # === 2. Load BEV transform parameters (for camera3) ===
        self.M_bev, self.bev_w, self.bev_h = None, None, None
        if self.bev_target_topic in self.camera_topics:
            try:
                self.get_logger().info(f"Loading BEV parameters from: {bev_param_file}")
                bev_params = np.load(bev_param_file)
                self.src_points = bev_params['src_points']
                self.dst_points = bev_params['dst_points']
                self.bev_h = int(bev_params['warp_h'])
                self.bev_w = int(bev_params['warp_w'])
                self.M_bev = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
                self.get_logger().info("✅ BEV transformation matrix calculated successfully.")
            except Exception as e:
                self.get_logger().error(f"FATAL: Failed to load BEV parameters: {e}. Shutting down.")
                rclpy.shutdown()
                return
        
        # === 3. Initialize state ===
        self.bridge = CvBridge()
        self.video_writers = {}  # Map topic name -> VideoWriter
        self.is_recording_started = False
        self.frame_counters = {topic: 0 for topic in self.camera_topics}

        # === 4. Create subscribers for each topic ===
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10 # Extra buffer for multiple streams
        )

        for topic in self.camera_topics:
            # Use partial to pass the topic name into the callback
            callback_with_topic = partial(self.image_callback, topic_name=topic)
            self.create_subscription(
                CompressedImage,
                topic,
                callback_with_topic,
                qos_profile_sensor_data
            )
        
        self.get_logger().info("✅ Unified recorder node initialized successfully. Waiting for images...")

    def image_callback(self, msg, topic_name):
        """
        Shared callback for all camera topics; dispatch based on topic name.
        """
        # On the first frame, initialize writers and start recording
        if not self.is_recording_started:
            self.initialize_all_video_writers(msg, topic_name)
            if not self.is_recording_started: # Abort if initialization failed
                return

        # Ensure this topic has a ready video writer
        writer = self.video_writers.get(topic_name)
        if writer is None or not writer.isOpened():
            self.get_logger().warn(f"Video writer for '{topic_name}' is not ready. Skipping frame.", throttle_duration_sec=5)
            return

        try:
            # === Decode frame and apply BEV when required ===
            processed_image = None
            if topic_name == self.bev_target_topic and self.M_bev is not None:
                # BEV conversion logic (mirrors bev_recorder.py)
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_color_image is not None:
                    processed_image = cv2.warpPerspective(
                        cv_color_image, self.M_bev, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR
                    )
            else:
                # Standard decoding path via CvBridge
                processed_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            # === Write the frame ===
            if processed_image is not None:
                writer.write(processed_image)
                self.frame_counters[topic_name] += 1
            else:
                self.get_logger().warn(f"Failed to decode or process image from {topic_name}", throttle_duration_sec=5)

        except Exception:
            self.get_logger().error(f"Error processing frame from {topic_name}:\n{traceback.format_exc()}")

    def initialize_all_video_writers(self, initial_msg, initial_topic):
        """
        Initialize all VideoWriter instances once the first image arrives.
        This function runs exactly once.
        """
        self.get_logger().info("First image received. Initializing all video writers for synchronized recording...")
        
        # Stamp all filenames with a shared timestamp to keep the set grouped
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        try:
            for topic, filename_template in zip(self.camera_topics, self.output_filenames):
                # Build filename, e.g., camera1_20250910_153000.mp4
                base, ext = os.path.splitext(filename_template)
                filename = f"{base}_{timestamp}{ext}"
                output_path = os.path.join(self.output_dir, filename)

                width, height = -1, -1

                # Use the predefined BEV dimensions when applicable
                if topic == self.bev_target_topic and self.bev_w is not None:
                    width, height = self.bev_w, self.bev_h
                else:
                    # Derive other resolutions dynamically from the first frame
                    temp_image = self.bridge.compressed_imgmsg_to_cv2(initial_msg, "bgr8")
                    h, w, _ = temp_image.shape
                    width, height = w, h
                
                if width > 0 and height > 0:
                    writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
                    if not writer.isOpened():
                        raise IOError(f"Cannot open video writer for {output_path}")
                    self.video_writers[topic] = writer
                    self.get_logger().info(f"✅ Initialized recorder for '{topic}' -> '{output_path}'")
                else:
                    raise ValueError(f"Invalid image dimensions for topic {topic}")

            self.is_recording_started = True
            self.get_logger().info("🚀 All recorders are active. Synchronized recording has started!")

        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize video writers: {e}")
            self.get_logger().error("Please check image topics, file permissions, and codec support.")
            # If initialization failed, clean up any partially created writers
            for writer in self.video_writers.values():
                if writer.isOpened():
                    writer.release()
            self.video_writers = {}
            rclpy.shutdown()

    def destroy_node(self):
        """
        Close all video files safely when the node shuts down.
        """
        self.get_logger().info("Shutting down node and finalizing videos...")
        for topic, writer in self.video_writers.items():
            if writer and writer.isOpened():
                output_path = "N/A"
                # This is a simplification; in a real scenario, you'd store the full path.
                for filename in self.output_filenames:
                    if os.path.splitext(filename)[0] in topic:
                         output_path = f"{self.output_dir}/{filename}"
                         break
                
                writer.release()
                self.get_logger().info(f"✅ Video for '{topic}' saved successfully. Total frames: {self.frame_counters[topic]}")
        self.get_logger().info("All video files have been saved.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedRecorderNode()
    if rclpy.ok(): # Ensure the node initialized correctly
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Keyboard interrupt detected.")
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
    main()
