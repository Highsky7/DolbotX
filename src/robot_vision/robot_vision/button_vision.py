#!/usr/bin/env python3
"""
ROS2 Node for Colored Button Detection and 3D Localization.

This node uses a RealSense camera and a YOLO object detection model to find
colored buttons ('blue', 'green', 'yellow'). Upon detecting a button, it
calculates its 3D position in the robot's coordinate frame.

When a button is detected within a specific Z-height range, it calls a
`PickPlace` service with the button's 3D coordinates. This is used to
trigger a robotic arm or other actuator. The node includes throttling to
prevent spamming the service.

It subscribes to:
- `/camera/color/image_raw/compressed`: The color image stream.
- `/camera/aligned_depth_to_color/image_raw`: The depth image, aligned to the color frame.
- `/camera/color/camera_info`: Camera intrinsic parameters.

It publishes:
- `/button_vision/viz/compressed`: A visualization image with detections.
- `/supply_command`: A command string when a service call is made.

It acts as a client for:
- `/pick_place_service`: A service to send the 3D coordinates of a detected button.
"""

import threading
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import cv2
import message_filters
import numpy as np
import pyrealsense2 as rs2  # type: ignore
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import torch
from ultralytics import YOLO

if not hasattr(rs2, "intrinsics"):
    import pyrealsense2.pyrealsense2 as rs2  # type: ignore

from mtc_interfaces.srv import PickPlace

BUTTON_CLASS_NAMES = ['blue', 'green', 'yellow']
BUTTON_CLASS_COLORS = {
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
}


@dataclass
class SupplyDetectionResult:
    """
    Container describing a single detection and its 3D localization.

    Attributes:
        found (bool): Whether a valid detection was made.
        confidence (float): The confidence score of the detection.
        class_id (Optional[int]): The class ID of the detected object.
        resized_bbox (Optional[Tuple[int, int, int, int]]): The bounding box
            coordinates (x1, y1, x2, y2) in the resized processing image.
        original_bbox (Optional[Tuple[int, int, int, int]]): The bounding box
            coordinates scaled back to the original image dimensions.
        position (Optional[np.ndarray]): The 3D position (x, y, z) of the
            object in the target TF frame.
        depth_m (Optional[float]): The depth of the object in meters.
    """

    found: bool
    confidence: float = 0.0
    class_id: Optional[int] = None
    resized_bbox: Optional[Tuple[int, int, int, int]] = None
    original_bbox: Optional[Tuple[int, int, int, int]] = None
    position: Optional[np.ndarray] = None
    depth_m: Optional[float] = None


class SupplyTransformError(RuntimeError):
    """Raised when the TF transform required for estimation fails."""


def _select_best_box(results: Sequence, expected_class: int) -> Optional[object]:
    """
    Return the highest confidence box for the desired class from YOLO results.

    Args:
        results (Sequence): The output from a YOLO model prediction.
        expected_class (int): The integer class ID to search for.

    Returns:
        Optional[object]: The box object with the highest confidence for the
                          specified class, or None if no such box is found.
    """
    if not results:
        return None

    first_result = results[0]
    boxes = getattr(first_result, "boxes", None)
    if boxes is None:
        return None

    candidates = [box for box in boxes if int(box.cls[0]) == expected_class]
    if not candidates:
        return None

    return max(candidates, key=lambda box: float(box.conf[0]))


def compute_supply_detection(
    results: Sequence,
    depth_image: np.ndarray,
    intrinsics: rs2.intrinsics,
    proc_width: int,
    proc_height: int,
    header_stamp,
    tf_buffer: tf2_ros.Buffer,
    optical_frame_id: str = "camera_color_optical_frame",
    target_frame_id: str = "camera_bottom_screw_frame",
    expected_class: int = 0,
) -> SupplyDetectionResult:
    """
    Compute the 3D pose of the best detection for a given class.

    This function finds the best bounding box for the `expected_class`,
    extracts the depth at its center, deprojects the 2D pixel to a 3D point
    in the camera's frame, and then transforms that point into the
    `target_frame_id`.

    Args:
        results (Sequence): The YOLO model's output.
        depth_image (np.ndarray): The depth image corresponding to the results.
        intrinsics (rs2.intrinsics): The camera's intrinsic parameters.
        proc_width (int): The width of the processed image.
        proc_height (int): The height of the processed image.
        header_stamp: The timestamp from the image message header.
        tf_buffer (tf2_ros.Buffer): The TF buffer for coordinate transforms.
        optical_frame_id (str): The TF frame of the camera's optical center.
        target_frame_id (str): The target TF frame for the final position.
        expected_class (int): The class ID to look for.

    Returns:
        SupplyDetectionResult: A data object containing the results of the
                               computation.

    Raises:
        SupplyTransformError: If the TF lookup fails.
    """
    if intrinsics is None:
        return SupplyDetectionResult(found=False)

    best_box = _select_best_box(results, expected_class)
    if best_box is None:
        return SupplyDetectionResult(found=False)

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cx_res = (x1 + x2) // 2
    cy_res = (y1 + y2) // 2

    if not (0 <= cx_res < proc_width and 0 <= cy_res < proc_height):
        return SupplyDetectionResult(found=False)

    depth_mm = int(depth_image[cy_res, cx_res])
    if depth_mm <= 0:
        return SupplyDetectionResult(
            found=False,
            confidence=float(best_box.conf[0]),
            class_id=expected_class,
            resized_bbox=(x1, y1, x2, y2),
        )

    scale_w = float(intrinsics.width) / float(proc_width)
    scale_h = float(intrinsics.height) / float(proc_height)

    orig_cx = int(cx_res * scale_w)
    orig_cy = int(cy_res * scale_h)

    point_cam = rs2.rs2_deproject_pixel_to_point(
        intrinsics,
        [orig_cx, orig_cy],
        depth_mm,
    )
    point_cam_m = np.asarray(point_cam, dtype=np.float64) / 1000.0

    stamped_point = PointStamped()
    stamped_point.header.frame_id = optical_frame_id
    stamped_point.header.stamp = header_stamp
    stamped_point.point.x = float(point_cam_m[0])
    stamped_point.point.y = float(point_cam_m[1])
    stamped_point.point.z = float(point_cam_m[2])

    lookup_time = Time.from_msg(header_stamp) if header_stamp is not None else Time()

    try:
        transform = tf_buffer.lookup_transform(target_frame_id, optical_frame_id, lookup_time)
        transformed_point = do_transform_point(stamped_point, transform)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
        raise SupplyTransformError(str(exc)) from exc

    position = np.array(
        [
            transformed_point.point.x,
            transformed_point.point.y,
            transformed_point.point.z,
        ],
        dtype=np.float64,
    )

    original_bbox = (
        int(x1 * scale_w),
        int(y1 * scale_h),
        int(x2 * scale_w),
        int(y2 * scale_h),
    )

    return SupplyDetectionResult(
        found=True,
        confidence=float(best_box.conf[0]),
        class_id=expected_class,
        resized_bbox=(x1, y1, x2, y2),
        original_bbox=original_bbox,
        position=position,
        depth_m=float(depth_mm) / 1000.0,
    )


class ButtonVisionNode(Node):
    """
    Detects colored buttons and sends their 3D positions via PickPlace service.

    This class sets up the ROS2 node, loads the ML model, subscribes to camera
    topics, and creates a service client. It processes incoming frames to find
    buttons and trigger actions.
    """

    def __init__(self) -> None:
        """Initialize the ButtonVisionNode."""
        super().__init__('button_vision_node')
        self.get_logger().info('--- Button Vision Node initialising ---')

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half = self.device == 'cuda'
        self.get_logger().info(f'Using device: {self.device}')

        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value

        self.button_model = None
        self.button_model_ready = False
        try:
            self.declare_parameter('button_model_path', './button_color.onnx')
            model_path = self.get_parameter('button_model_path').get_parameter_value().string_value
            self.button_model = YOLO(model_path, task='detect')
            self.button_model_ready = True
            self.get_logger().info('✅ button_color ONNX model loaded successfully.')
        except Exception as exc:  # pragma: no cover - defensive logging
            self.get_logger().error(f'Failed to load button_color model: {exc}. Button detection disabled.')

        self.intrinsics = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pick_place_client = self.create_client(PickPlace, '/pick_place_service')
        while not self.pick_place_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('pick_place service not available, waiting...')

        self.button_service_in_progress: Dict[str, bool] = {name: False for name in BUTTON_CLASS_NAMES}
        self.button_last_sent_time: Dict[str, Optional[rclpy.time.Time]] = {name: None for name in BUTTON_CLASS_NAMES}
        self.button_send_interval = Duration(seconds=1.0)
        self.service_z_min = self.declare_parameter('button_service_z_min', -0.2).value
        self.service_z_max = self.declare_parameter('button_service_z_max', 0.6).value

        self.viz_pub = self.create_publisher(CompressedImage, '/button_vision/viz/compressed', 10)
        self.command_pub = self.create_publisher(String, '/supply_command', 10)

        self.processing_lock = threading.Lock()
        self.button_executor = threading.Thread

        self.resized_color_yolo = np.empty((self.proc_height, self.proc_width, 3), dtype=np.uint8)
        self.resized_depth = np.empty((self.proc_height, self.proc_width), dtype=np.uint16)

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)
        self.get_logger().info('Waiting for CameraInfo...')

        self._is_shutting_down = False

    def destroy_node(self) -> bool:
        """Shut down the node cleanly."""
        self._is_shutting_down = True
        return super().destroy_node()

    def camera_info_callback(self, info_msg: CameraInfo) -> None:
        """
        Receive camera intrinsics and initialize the rest of the node.

        This callback is triggered once, then the subscription is destroyed.
        It sets up the `pyrealsense2.intrinsics` object and then calls
        `initialize_sync` to start the main processing loop.

        Args:
            info_msg (CameraInfo): The camera info message.
        """
        if self.intrinsics is not None:
            return

        self.get_logger().info('✅ CameraInfo received for button vision.')
        import pyrealsense2 as rs2
        if not hasattr(rs2, 'intrinsics'):
            import pyrealsense2.pyrealsense2 as rs2  # type: ignore
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = info_msg.width
        self.intrinsics.height = info_msg.height
        self.intrinsics.ppx = info_msg.k[2]
        self.intrinsics.ppy = info_msg.k[5]
        self.intrinsics.fx = info_msg.k[0]
        self.intrinsics.fy = info_msg.k[4]
        self.intrinsics.model = rs2.distortion.brown_conrady if info_msg.distortion_model == 'plumb_bob' else rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = list(info_msg.d)

        self.initialize_sync()
        self.destroy_subscription(self.camera_info_sub)

    def initialize_sync(self) -> None:
        """Initialize the synchronized subscribers for color and depth images."""
        color_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5, slop=0.15)
        self.sync.registerCallback(self.realsense_callback)
        self.get_logger().info('✅ Button vision synchronizer initialised.')

    def realsense_callback(self, compressed_image_msg: CompressedImage, depth_msg: Image) -> None:
        """
        Handle synchronized color and depth image messages.

        This is the main entry point for frame processing. It uses a lock to
        prevent concurrent processing of frames.

        Args:
            compressed_image_msg (CompressedImage): The compressed color image.
            depth_msg (Image): The depth image.
        """
        if not self.button_model_ready or self.intrinsics is None or self._is_shutting_down:
            return

        if not self.processing_lock.acquire(blocking=False):
            self.get_logger().warn('Dropping button frame: processing busy.', throttle_duration_sec=1.0)
            return

        try:
            self._process_frame(compressed_image_msg, depth_msg)
        finally:
            self.processing_lock.release()

    def _process_frame(self, compressed_image_msg: CompressedImage, depth_msg: Image) -> None:
        """
        Process a single pair of color and depth frames.

        Decodes images, resizes them, runs the YOLO model, computes 3D
        positions for each detected button class, and triggers visualization
        and service calls.

        Args:
            compressed_image_msg (CompressedImage): The compressed color image.
            depth_msg (Image): The depth image.
        """
        np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        depth_image_raw = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        if color_image is None or depth_image_raw is None:
            return

        cv2.resize(depth_image_raw, (self.proc_width, self.proc_height), dst=self.resized_depth, interpolation=cv2.INTER_NEAREST)
        cv2.resize(color_image, (self.proc_width, self.proc_height), dst=self.resized_color_yolo, interpolation=cv2.INTER_AREA)

        results = self.button_model(self.resized_color_yolo, verbose=False, half=self.use_half)
        annotated_image = color_image.copy()

        for class_id, class_name in enumerate(BUTTON_CLASS_NAMES):
            try:
                detection = compute_supply_detection(
                    results=results,
                    depth_image=self.resized_depth,
                    intrinsics=self.intrinsics,
                    proc_width=self.proc_width,
                    proc_height=self.proc_height,
                    header_stamp=compressed_image_msg.header.stamp,
                    tf_buffer=self.tf_buffer,
                    expected_class=class_id,
                )
            except SupplyTransformError as exc:
                self.get_logger().warn(f'TF transform failed for {class_name}: {exc}', throttle_duration_sec=5.0)
                continue

            if not detection.found or detection.position is None:
                continue

            self.draw_detection(annotated_image, detection, class_name)
            self.maybe_call_service(class_name, detection)

        self.publish_viz_image(annotated_image)

    def draw_detection(self, annotated_image: np.ndarray, detection: SupplyDetectionResult, class_name: str) -> None:
        """
        Draw bounding boxes and labels on the visualization image.

        Args:
            annotated_image (np.ndarray): The image to draw on.
            detection (SupplyDetectionResult): The detection result to draw.
            class_name (str): The name of the detected class.
        """
        if detection.original_bbox is None or detection.position is None:
            return

        x1, y1, x2, y2 = detection.original_bbox
        color = BUTTON_CLASS_COLORS.get(class_name, (255, 255, 255))
        label = (
            f'{class_name}: '
            f'x={detection.position[0]:.2f}m '
            f'y={detection.position[1]:.2f}m '
            f'z={detection.position[2]:.2f}m'
        )
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def publish_viz_image(self, image: np.ndarray) -> None:
        """
        Publish the annotated image for visualization.

        Args:
            image (np.ndarray): The image to publish.
        """
        msg = CompressedImage()
        msg.format = 'jpeg'
        msg.header.stamp = self.get_clock().now().to_msg()
        success, encoded_image = cv2.imencode('.jpg', image)
        if success:
            msg.data = encoded_image.tobytes()
            self.viz_pub.publish(msg)

    def maybe_call_service(self, class_name: str, detection: SupplyDetectionResult) -> None:
        """
        Check conditions and potentially call the PickPlace service.

        A service call is made if:
        - The detection's Z coordinate is within the configured range.
        - Enough time has passed since the last call for this button class.
        - A call for this class is not already in progress.

        Args:
            class_name (str): The name of the detected button class.
            detection (SupplyDetectionResult): The result of the detection.
        """
        if detection.position is None:
            return

        z_coord = detection.position[2]
        if not (self.service_z_min <= z_coord <= self.service_z_max):
            return

        now = self.get_clock().now()
        last_time = self.button_last_sent_time[class_name]
        if last_time is not None and (now - last_time) < self.button_send_interval:
            return

        if self.button_service_in_progress[class_name]:
            return

        request = PickPlace.Request(
            x=float(detection.position[0]),
            y=float(detection.position[1]),
            z=float(detection.position[2]),
        )

        self.button_service_in_progress[class_name] = True
        self.button_last_sent_time[class_name] = now
        future = self.pick_place_client.call_async(request)
        future.add_done_callback(partial(self.service_response_callback, class_name))

        self.command_pub.publish(String(data=f'service_call:{class_name}'))
        self.get_logger().info(
            f'Sent PickPlace request for {class_name} button '
            f'at ({request.x:.2f}, {request.y:.2f}, {request.z:.2f}).'
        )

    def service_response_callback(self, class_name: str, future) -> None:
        """
        Handle the response from the PickPlace service call.

        Logs the result and resets the 'in_progress' flag for that class.

        Args:
            class_name (str): The class name the call was for.
            future: The future object containing the service response.
        """
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'✅ PickPlace success for {class_name}: {response.message}')
            else:
                self.get_logger().warn(f'⚠️ PickPlace failed for {class_name}: {response.message}')
        except Exception as exc:  # pragma: no cover - defensive logging
            self.get_logger().error(f'PickPlace call for {class_name} raised: {exc}')
        finally:
            self.button_service_in_progress[class_name] = False


def main(args=None) -> None:
    """Entry point for the button_vision_node."""
    rclpy.init(args=args)
    node = ButtonVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
