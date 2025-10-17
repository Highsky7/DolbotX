#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility script: configure BEV (Bird's-Eye View) parameters for ROS 2 with
automatic right-side alignment.
----------------------------------------------------------------
Subscribes to the ROS 2 topic (/camera/color/image_raw/compressed) and lets
you pick four source points from live video.

- Manually select the four source points required for the BEV transform.
- Align the right-side points vertically with their left-side counterparts.
- Persist the chosen coordinates (src_points) into both NPZ and TXT files.

Press 's' after selection to save the BEV parameters.
"""

import cv2
import numpy as np
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class BevParamSetterNode(Node):
    def __init__(self):
        super().__init__('bev_param_setter_node')
        self.get_logger().info('BEV Parameter Setter ROS2 Node has been started.')

        # BEV parameters
        self.warp_w = 640
        self.warp_h = 640
        self.out_npz_file = 'bev_params.npz'
        self.out_txt_file = 'selected_bev_src_points.txt'

        # ROS 2 subscriber
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera3/image_raw/compressed',  # Requested topic
            self.image_callback,
            10
        )

        # State variables tracked at the instance level
        self.cv_image = None
        self.src_points = []
        self.max_points = 4

        # Configure OpenCV windows and mouse callbacks
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Original", self.mouse_callback, self)

        self.print_instructions()

    def print_instructions(self):
        self.get_logger().info("\n[INSTRUCTIONS]")
        self.get_logger().info("Waiting for images from the ROS 2 topic...")
        self.get_logger().info("Select four points on the original image with the left mouse button.")
        self.get_logger().info("Click order: 1. left-bottom -> 2. right-bottom -> 3. left-top -> 4. right-top")
        self.get_logger().info("✨ Right-side points automatically match the left-side y coordinate.")
        self.get_logger().info("'r' key: reset (clear selected points)")
        self.get_logger().info("'s' key: save BEV parameters and exit")
        self.get_logger().info("'q' key: exit without saving\n")

    def image_callback(self, msg):
        """Receive an image message and update self.cv_image."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Failed to decode image: {e}')

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events."""
        node_instance = param  # self
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(node_instance.src_points) < node_instance.max_points:
                point_order = ["Left-Bottom", "Right-Bottom", "Left-Top", "Right-Top"]
                current_point_index = len(node_instance.src_points)
                final_point = (x, y)

                # Auto-align the y coordinate for the 2nd and 4th points.
                if current_point_index == 1:   # Right-bottom
                    if len(node_instance.src_points) > 0:
                        y_bottom = node_instance.src_points[0][1]
                        final_point = (x, y_bottom)
                elif current_point_index == 3: # Right-top
                    if len(node_instance.src_points) > 2:
                        y_top = node_instance.src_points[2][1]
                        final_point = (x, y_top)

                node_instance.src_points.append(final_point)
                self.get_logger().info(f"Added {point_order[current_point_index]} point: {final_point} ({len(node_instance.src_points)}/{node_instance.max_points})")

                if len(node_instance.src_points) == node_instance.max_points:
                    self.get_logger().info("All four points selected. Press 's' to save or 'r' to reset.")
            else:
                self.get_logger().warn("Four points are already selected. Press 'r' to reset or 's' to save.")

    def save_params(self):
        """Persist the BEV parameters to disk."""
        if len(self.src_points) < self.max_points:
            self.get_logger().warn("Select all four points before saving.")
            return False

        self.get_logger().info("'s' pressed -> saving BEV parameters and exiting")
        
        dst_points_default = np.float32([
            [0, self.warp_h],          # Left-bottom
            [self.warp_w, self.warp_h],# Right-bottom
            [0, 0],                    # Left-top
            [self.warp_w, 0]           # Right-top
        ])
        
        src_arr = np.float32(self.src_points)
        
        # Save NPZ file
        np.savez(self.out_npz_file,
                 src_points=src_arr,
                 dst_points=dst_points_default,
                 warp_w=self.warp_w,
                 warp_h=self.warp_h)
        self.get_logger().info(f"Saved BEV parameters to '{self.out_npz_file}'.")

        # Save TXT file
        point_labels = ["Left-Bottom", "Right-Bottom", "Left-Top", "Right-Top"]
        try:
            with open(self.out_txt_file, 'w') as f:
                f.write("# Selected BEV Source Points (x, y) for original image\n")
                f.write("# Order: Left-Bottom, Right-Bottom, Left-Top, Right-Top\n")
                for i, point in enumerate(self.src_points):
                    f.write(f"{point[0]}, {point[1]} # {point_labels[i]}\n")
            self.get_logger().info(f"Saved selected coordinates to '{self.out_txt_file}'.")
        except Exception as e:
            self.get_logger().error(f"Failed to write TXT file: {e}")
            
        return True

    def run(self):
        """Main loop: process ROS 2 events, refresh windows, and handle input."""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.cv_image is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.get_logger().info("'q' pressed -> exiting")
                    break
                continue

            disp = self.cv_image.copy()

            # Draw the selected points and connecting polygon
            point_labels = ["1 (L-Bot)", "2 (R-Bot)", "3 (L-Top)", "4 (R-Top)"]
            for i, pt in enumerate(self.src_points):
                cv2.circle(disp, pt, 5, (0, 255, 0), -1)
                label = point_labels[i] if i < len(point_labels) else f"{i+1}"
                cv2.putText(disp, label, (pt[0] + 5, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(self.src_points) == 4:
                cv2.polylines(disp, [np.array(self.src_points, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

            cv2.imshow("Original", disp)

            # Compute and display the BEV projection
            bev_result = np.zeros((self.warp_h, self.warp_w, 3), dtype=np.uint8)
            if len(self.src_points) == 4:
                src_np = np.float32(self.src_points)
                dst_points_default = np.float32([
                    [0, self.warp_h], [self.warp_w, self.warp_h],
                    [0, 0], [self.warp_w, 0]
                ])
                M = cv2.getPerspectiveTransform(src_np, dst_points_default)
                bev_result = cv2.warpPerspective(self.cv_image, M, (self.warp_w, self.warp_h))
            cv2.imshow("BEV", bev_result)

            # Handle key input
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self.get_logger().info("'q' pressed -> exiting without saving")
                break
            elif key == ord('r'):
                self.get_logger().info("'r' pressed -> clearing all points")
                self.src_points = []
            elif key == ord('s'):
                if self.save_params():
                    break  # Exit loop after successful save

        # Clean up windows and ROS resources
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        self.get_logger().info("bev_utils_ros2.py finished.")

def main(args=None):
    rclpy.init(args=args)
    bev_node = BevParamSetterNode()
    bev_node.run()

if __name__ == '__main__':
    main()
