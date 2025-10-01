#!/usr/bin/env python3
"""Button detection node using button_color.onnx to report distances via PickPlace service."""

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
    """Container describing a single detection and its 3D localisation."""

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
    """Return the highest confidence box for the desired class."""

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
    """Compute the 3D pose of the best detection for a given class."""

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
    """Detects colored buttons and sends their 3D positions via PickPlace service."""

    def __init__(self) -> None:
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
        self._is_shutting_down = True
        return super().destroy_node()

    # Camera info handling -------------------------------------------------
    def camera_info_callback(self, info_msg: CameraInfo) -> None:
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
        color_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5, slop=0.15)
        self.sync.registerCallback(self.realsense_callback)
        self.get_logger().info('✅ Button vision synchronizer initialised.')

    # Frame processing -----------------------------------------------------
    def realsense_callback(self, compressed_image_msg: CompressedImage, depth_msg: Image) -> None:
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

    # Drawing & publishing -------------------------------------------------
    def draw_detection(self, annotated_image: np.ndarray, detection: SupplyDetectionResult, class_name: str) -> None:
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
        msg = CompressedImage()
        msg.format = 'jpeg'
        msg.header.stamp = self.get_clock().now().to_msg()
        success, encoded_image = cv2.imencode('.jpg', image)
        if success:
            msg.data = encoded_image.tobytes()
            self.viz_pub.publish(msg)

    # Service handling -----------------------------------------------------
    def maybe_call_service(self, class_name: str, detection: SupplyDetectionResult) -> None:
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
