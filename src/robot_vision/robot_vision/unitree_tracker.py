#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for 3D Object Tracking using YOLO and RealSense.

This node uses an Intel RealSense D435i camera and a 'unitree_go2.onnx'
YOLO model to detect a target object. It calculates the object's 3D coordinates
in the camera's optical frame and then transforms these coordinates into the
'camera_bottom_screw_frame' using the TF2 library.

The final transformed coordinates are published to the '/target_xy' topic as a
`geometry_msgs/msg/Point` message. The node also publishes a compressed image
topic for visualization.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import message_filters
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge

import tf2_ros
from tf2_geometry_msgs import do_transform_point


import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2


class YoloObjectLocatorNode(Node):
    """
    Detects and localizes a target object in 3D space using YOLO and depth data.
    """

    def __init__(self):
        """
        Initialize the YoloObjectLocatorNode.

        This sets up the node, loads the YOLO model, initializes TF2, and
        creates the necessary publishers and subscribers for image processing
        and coordinate publishing.
        """
        super().__init__('yolo_object_locator_node_tf')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for 3D Object Tracking using YOLO and RealSense.

This node uses an Intel RealSense D435i camera and a 'unitree_go2.onnx'
YOLO model to detect a target object. It calculates the object's 3D coordinates
in the camera's optical frame and then transforms these coordinates into the
'camera_bottom_screw_frame' using the TF2 library.

The final transformed coordinates are published to the '/target_xy' topic as a
`geometry_msgs/msg/Point` message. The node also publishes a compressed image
topic for visualization.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import message_filters
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge

import tf2_ros
from tf2_geometry_msgs import do_transform_point


import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2


class YoloObjectLocatorNode(Node):
    """
    Detects and localizes a target object in 3D space using YOLO and depth data.
    """
    def __init__(self):
        """
        Initialize the YoloObjectLocatorNode.

        This sets up the node, loads the YOLO model, initializes TF2, and
        creates the necessary publishers and subscribers for image processing
        and coordinate publishing.
        """
        super().__init__('yolo_object_locator_node_tf')
        self.get_logger().info("--- YOLO Object Locator Node for unitree_go2 (with TF Transformation) ---")

        # Lock object for concurrency control
        self.processing_lock = threading.Lock()

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")
        
        # Load the ONNX model
        try:
            self.declare_parameter('model_path', './unitree_go2.onnx')
            model_path = self.get_parameter('model_path').get_parameter_value().string_value
            self.model = YOLO(model_path, task='detect')
            self.get_logger().info(f"✅ YOLO ONNX model loaded successfully from: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            self.destroy_node()
            return

        # Variable to store camera intrinsic parameters
        self.intrinsics = None
        self.camera_info_sub = None
        
        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Image processing size parameters for performance
        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value

        # Publisher for the target's 3D coordinates
        self.target_pub = self.create_publisher(Point, '/target_xy', 10)
        
        # Publisher for the visualization image
        self.viz_pub = self.create_publisher(CompressedImage, '/yolo_locator/viz/compressed', 10)

        self.yolo_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='yolo_worker')
        self._is_shutting_down = False

        # Subscribe to the RealSense camera info topic to get intrinsics once
        info_topic = "/camera/color/camera_info"
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, 10)
        self.get_logger().info(f"Waiting for CameraInfo on topic: {info_topic}")

    def camera_info_callback(self, info_msg):
        """
        Receive camera intrinsic parameters once and store them.

        Args:
            info_msg (CameraInfo): The message containing camera intrinsics.
        """
        if self.intrinsics is not None:
            return
            
        try:
            self.get_logger().info("✅ CameraInfo received.")
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = info_msg.width
            self.intrinsics.height = info_msg.height
            self.intrinsics.ppx = info_msg.k[2]
            self.intrinsics.ppy = info_msg.k[5]
            self.intrinsics.fx = info_msg.k[0]
            self.intrinsics.fy = info_msg.k[4]
            
            if info_msg.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif info_msg.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            else:
                self.intrinsics.model = rs2.distortion.none

            self.intrinsics.coeffs = [i for i in info_msg.d]
            
            # After receiving camera info, set up the synchronized image callback
            self.initialize_image_sync()

            # Unsubscribe from the camera info topic
            if self.camera_info_sub:
                self.destroy_subscription(self.camera_info_sub)
                self.camera_info_sub = None
                self.get_logger().info("CameraInfo subscription destroyed. Starting image synchronization.")
        except Exception as e:
            self.get_logger().error(f"Error in camera_info_callback: {e}")

    def initialize_image_sync(self):
        """Set up message_filters to synchronize color and depth images."""
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        # Use ApproximateTimeSynchronizer to group messages with similar timestamps
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [realsense_img_sub, depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.image_callback)
        self.get_logger().info("✅ YOLO Object Locator Node initialized successfully.")

    def image_callback(self, compressed_image_msg, depth_msg):
        """
        Main callback for synchronized image messages.

        Args:
            compressed_image_msg (CompressedImage): The compressed color image message.
            depth_msg (Image): The depth image message.
        """
        if self.intrinsics is None or self._is_shutting_down:
            return

        # If the previous frame is still being processed, skip the current one
        if self.processing_lock.acquire(blocking=False):
            try:
                self.yolo_executor.submit(self._process_images, compressed_image_msg, depth_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to submit image processing task: {e}")
                self.processing_lock.release()
        else:
            self.get_logger().warn("Dropping a frame, previous frame is still processing.", throttle_duration_sec=1)

    def _process_images(self, compressed_image_msg, depth_msg):
        """
        Core logic for YOLO inference, coordinate calculation, and TF transform.
        This runs in a separate thread.

        Args:
            compressed_image_msg (CompressedImage): The compressed color image message.
            depth_msg (Image): The depth image message.
        """
        try:
            # 1. Decode and convert images
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            if cv_color is None or cv_depth_raw is None:
                self.get_logger().warn("Failed to decode image. Skipping frame.")
                return

            # Copy image for visualization
            viz_image = cv_color.copy()

            # Resize color image for YOLO input
            resized_color_yolo = cv2.resize(cv_color, (self.proc_width, self.proc_height), interpolation=cv2.INTER_AREA)

            # 2. Run YOLO model inference
            results = self.model(resized_color_yolo, verbose=False, device=self.device)

            # 3. Find the object with the highest confidence
            best_box = None
            max_conf = 0.0
            for box in results[0].boxes:
                if box.conf[0] > max_conf:
                    max_conf = box.conf[0]
                    best_box = box

            # 4. If an object is detected, calculate and publish its coordinates
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cx_res, cy_res = (x1 + x2) // 2, (y1 + y2) // 2

                scale_w = self.intrinsics.width / self.proc_width
                scale_h = self.intrinsics.height / self.proc_height
                orig_cx = int(cx_res * scale_w)
                orig_cy = int(cy_res * scale_h)

                if 0 <= orig_cy < self.intrinsics.height and 0 <= orig_cx < self.intrinsics.width:
                    depth_in_mm = cv_depth_raw[orig_cy, orig_cx]
                    
                    if depth_in_mm > 0:
                        # 5. Deproject 2D pixel to 3D point (relative to camera optical frame)
                        deprojected = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [orig_cx, orig_cy], depth_in_mm)
                        optical_frame_coords = np.array([p / 1000.0 for p in deprojected])
                        
                        # TF Transformation Logic
                        point_in_optical_frame = PointStamped()
                        point_in_optical_frame.header.frame_id = "camera_color_optical_frame"
                        point_in_optical_frame.header.stamp = compressed_image_msg.header.stamp
                        point_in_optical_frame.point.x, point_in_optical_frame.point.y, point_in_optical_frame.point.z = optical_frame_coords
                        
                        target_frame = "camera_bottom_screw_frame"
                        transformed_position = None
                        
                        try:
                            transform = self.tf_buffer.lookup_transform(
                                target_frame, 
                                point_in_optical_frame.header.frame_id, 
                                rclpy.time.Time()
                            )
                            point_in_target_frame = do_transform_point(point_in_optical_frame, transform)
                            transformed_position = np.array([
                                point_in_target_frame.point.x, 
                                point_in_target_frame.point.y, 
                                point_in_target_frame.point.z
                            ])
                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                            self.get_logger().warn(f"Coordinate transform failed: {e}", throttle_duration_sec=5.0)
                            return # Exit function on transform failure
                        
                        if transformed_position is not None:
                            # 6. Publish the transformed coordinates as a Point message
                            target_point_msg = Point()
                            target_point_msg.x = transformed_position[0]
                            target_point_msg.y = transformed_position[1]
                            target_point_msg.z = transformed_position[2]
                            self.target_pub.publish(target_point_msg)

                            # 7. Draw bounding box and transformed coordinates for visualization
                            orig_x1, orig_y1 = int(x1 * scale_w), int(y1 * scale_h)
                            orig_x2, orig_y2 = int(x2 * scale_w), int(y2 * scale_h)
                            
                            label = f"Target(TF): x={transformed_position[0]:.2f}, y={transformed_position[1]:.2f}, z={transformed_position[2]:.2f} m"
                            cv2.rectangle(viz_image, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
                            cv2.putText(viz_image, label, (orig_x1, orig_y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Publish the visualization image
            self.publish_compressed_viz(viz_image)

        except Exception as e:
            self.get_logger().error(f"Error in image processing worker: {e}\n{traceback.format_exc()}")
        finally:
            # Release the lock after processing is complete
            self.processing_lock.release()

    def publish_compressed_viz(self, cv_image):
        """
        Publish the visualization image as a compressed JPEG.

        Args:
            cv_image (np.ndarray): The image to be published.
        """
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        success, encoded_image = cv2.imencode('.jpg', cv_image)
        if success:
            msg.data = np.array(encoded_image).tobytes()
            self.viz_pub.publish(msg)

    def destroy_node(self):
        """Safely shut down the node and its thread pool."""
        self.get_logger().info("Shutting down the thread pool.")
        self._is_shutting_down = True
        self.yolo_executor.shutdown(wait=True)
        super().destroy_node()


def main(args=None):
    """Main entry point for the ROS2 node."""
    rclpy.init(args=args)
    node = YoloObjectLocatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()