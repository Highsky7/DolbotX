#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class HsvPickerNode(Node):
    """
    Node that subscribes to a ROS 2 topic and inspects HSV values.
    """
    def __init__(self):
        super().__init__('hsv_picker_node')
        self.get_logger().info('HSV Picker ROS2 Node has been started.')

        # Set up the ROS 2 subscription
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',  # Requested topic
            self.image_callback,
            10)
        
        self.cv_image = None
        self.window_name = 'HSV Picker - Click on the track'
        cv2.namedWindow(self.window_name)
        # Pass self to the mouse callback so it can access the latest frame
        cv2.setMouseCallback(self.window_name, self.get_hsv_value, self)

        self.get_logger().info("Waiting for images from the ROS 2 topic...")
        self.get_logger().info("Click on the track to inspect HSV values. Press 'q' to exit.")

    def get_hsv_value(self, event, x, y, flags, param):
        """
        Mouse callback that prints the BGR and HSV values of the clicked pixel.
        """
        # Use the node instance passed as param
        node_instance = param
        if event == cv2.EVENT_LBUTTONDOWN:
            if node_instance.cv_image is not None:
                # Convert the BGR pixel to HSV
                hsv_pixel = cv2.cvtColor(np.uint8([[node_instance.cv_image[y, x]]]), cv2.COLOR_BGR2HSV)
                # Log the sampled value
                self.get_logger().info(f"Clicked Pixel BGR: {node_instance.cv_image[y, x]}, HSV: {hsv_pixel[0][0]}")
            else:
                self.get_logger().warn("No image received yet.")

    def image_callback(self, msg):
        """
        Callback that decodes incoming image messages.
        """
        try:
            # Decode the compressed image into an OpenCV matrix
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Failed to decode image: {e}')
            return

    def run(self):
        """
        Main loop: update the display and handle user input.
        """
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)  # Process ROS 2 callbacks
            
            if self.cv_image is not None:
                cv2.imshow(self.window_name, self.cv_image)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("'q' pressed -> exiting")
                break

        # Clean up before shutting down
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    hsv_picker_node = HsvPickerNode()
    hsv_picker_node.run()

if __name__ == '__main__':
    main()
