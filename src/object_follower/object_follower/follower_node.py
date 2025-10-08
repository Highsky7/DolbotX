#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node for Following a Target Object.

This node subscribes to a topic providing the 3D coordinates of a target object
and computes the required left and right wheel velocities to follow it. It
implements a proportional control strategy for both distance and heading,
with a speed scaling policy to reduce velocity during turns.
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point  # Subscribes to Point messages
from std_msgs.msg import Float32


def clamp(x, lo, hi):
    """
    Clamp a value to a specified range.

    Args:
        x: The value to clamp.
        lo: The lower bound of the range.
        hi: The upper bound of the range.

    Returns:
        The clamped value.
    """
    return lo if x < lo else hi if x > hi else x


class ObjectFollower(Node):
    """
    A ROS2 node that calculates wheel velocities to follow a target object.

    This node implements a control logic to maintain a specified distance
    from a target while aligning with its heading.

    Input:
      - `/target_xy` (geometry_msgs/Point): The 3D coordinates of the target.
        - Assumes RealSense optical frame (x: right+, y: down+, z: forward+)
          if `input_is_optical` is True.
        - Assumes a standard robot frame (x: forward+, y: left+) if False.

    Output:
      - `/cmd_vel_left`, `/cmd_vel_right` (std_msgs/Float32): Target velocities
        for the left and right wheels in [m/s].

    Velocity Policy:
      - Distance-based velocity: v_dist = k_dist * (r - follow_distance)
      - Cruise speed limit: v_cruise (default 0.6 m/s)
      - Speed reduction during turns: speed is scaled down as |theta| increases.
      - Final velocity: v = min(max(0, v_dist), v_cruise) * scale(theta)
      - Hard safety limit: |v| <= v_max_hw (never exceeds 0.6 m/s)
    """

    def __init__(self):
        """Initialize the ObjectFollower node, its parameters, and pub/sub."""
        super().__init__('object_follower')

        # ===== Topics =====
        self.declare_parameter('target_topic', '/target_xy')
        self.declare_parameter('left_topic', '/cmd_vel_left')
        self.declare_parameter('right_topic', '/cmd_vel_right')

        # ===== Input Coordinate System Flag =====
        # True  -> Input is RealSense optical (x: right, y: down, z: forward)
        # False -> Input is already in the robot's follow frame (x: forward, y: left)
        self.declare_parameter('input_is_optical', False)

        # ===== Kinematics / Control =====
        self.declare_parameter('wheelbase', 0.60)        # [m]
        self.declare_parameter('k_heading', 1.5)         # Proportional gain for heading [rad/rad]
        self.declare_parameter('k_dist', 1.5)            # Proportional gain for distance [m/s per m]
        self.declare_parameter('follow_distance', 0.3)   # [m]
        self.declare_parameter('reverse_ok', False)      # Whether reversing is allowed

        # ===== Velocity Policy (Important) =====
        self.declare_parameter('v_cruise', 0.6)          # Desired max speed when going straight
        self.declare_parameter('v_max_hw', 0.6)          # Hard speed limit for safety

        # Turn speed scaling parameters (linear interpolation)
        self.declare_parameter('turn_ang_straight', 0.05)  # [rad] Angle considered "almost straight"
        self.declare_parameter('turn_ang_full', 0.6)       # [rad] Angle for "full turn"
        self.declare_parameter('turn_scale_min', 0.35)     # Speed ratio during a full turn

        # ===== Camera Offset Correction =====
        self.declare_parameter('apply_camera_offset', True)
        self.declare_parameter('camera_offset_x', 0.27)   # Camera is 0.27m ahead of the base (+x_follow)
        self.declare_parameter('camera_offset_y', 0.0)

        # ===== Timeout =====
        self.declare_parameter('lost_timeout', 0.5)       # [s]

        # QoS Profile
        qos = QoSProfile(
            depth=10, history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers and Subscribers
        left_topic   = self.get_parameter('left_topic').value
        right_topic  = self.get_parameter('right_topic').value
        target_topic = self.get_parameter('target_topic').value

        self.pub_left  = self.create_publisher(Float32, left_topic, qos)
        self.pub_right = self.create_publisher(Float32, right_topic, qos)
        self.create_subscription(Point, target_topic, self.cb_target_point, qos)

        # Timer for periodic command publishing
        self.timer = self.create_timer(1.0 / 50.0, self.on_timer)  # 50 Hz

        # Internal state
        self.last_seen = -1.0
        self.v_cmd = 0.0
        self.w_cmd = 0.0

        self.get_logger().info(
            f"[object_follower] sub={target_topic} pub=({left_topic}, {right_topic}) "
            f"v_cruise={self.get_parameter('v_cruise').value:.2f} m/s"
        )

    def _turn_speed_scale(self, theta: float) -> float:
        """
        Calculate a speed scaling factor based on the turning angle.

        This function linearly interpolates the speed scale between 1.0 (for
        straight motion) and a minimum value (for sharp turns).

        Args:
            theta (float): The turning angle in radians.

        Returns:
            float: The speed scaling factor, between 0.0 and 1.0.
        """
        a = abs(theta)
        ang0 = float(self.get_parameter('turn_ang_straight').value)
        ang1 = float(self.get_parameter('turn_ang_full').value)
        smin = float(self.get_parameter('turn_scale_min').value)
        smin = clamp(smin, 0.0, 1.0)

        if a <= ang0:
            return 1.0
        if a >= ang1:
            return smin
        # Linear interpolation
        t = (a - ang0) / max(1e-9, (ang1 - ang0))
        return 1.0 + t * (smin - 1.0)

    def cb_target_point(self, msg: Point):
        """
        Callback for receiving the target's 3D coordinates.

        This function converts the input coordinates to the robot's follow
        frame, calculates the desired linear (v) and angular (w) velocities,
        and stores them in the node's state.

        Args:
            msg (Point): The message containing the target's coordinates.
        """
        # Input coordinates (RealSense optical: X=right+, Y=down+, Z=forward+), units in meters
        X_cam = float(msg.x)
        Y_cam = float(msg.y)
        Z_cam = float(msg.z)

        # Check if coordinate system transformation is needed
        optical = bool(self.get_parameter('input_is_optical').value)

        if optical:
            # Convert from optical frame to robot's follow frame (x: forward+, y: left+)
            x_follow = Z_cam
            y_follow = -X_cam
        else:
            # Assume input is already in the follow frame
            x_follow = X_cam
            y_follow = Y_cam

        # Correct for camera-to-base offset (translation)
        if bool(self.get_parameter('apply_camera_offset').value):
            x_follow += float(self.get_parameter('camera_offset_x').value)
            y_follow += float(self.get_parameter('camera_offset_y').value)

        # Convert to polar coordinates (r, theta)
        r = math.hypot(x_follow, y_follow)
        theta = math.atan2(y_follow, x_follow)  # left is positive

        # Angular velocity (P control)
        k_heading = float(self.get_parameter('k_heading').value)
        w = k_heading * theta

        # Distance-based linear velocity (P control) with policy
        follow_d = float(self.get_parameter('follow_distance').value)
        k_dist   = float(self.get_parameter('k_dist').value)
        v_cruise = float(self.get_parameter('v_cruise').value)
        v_max_hw = float(self.get_parameter('v_max_hw').value)

        v_dist = k_dist * (r - follow_d)  # Positive if too far, negative if too close
        if not bool(self.get_parameter('reverse_ok').value):
            v_dist = max(0.0, v_dist)     # Prohibit reversing

        # Apply turn-based speed scaling
        scale = self._turn_speed_scale(theta)
        v = min(max(0.0, v_dist), v_cruise) * scale

        # Apply hard safety limit
        v = clamp(v, -v_max_hw, v_max_hw)

        # Store the calculated commands in the node's state
        self.v_cmd = v
        self.w_cmd = w
        self.last_seen = time.time()

    def on_timer(self):
        """
        Periodically called to publish wheel velocity commands.

        This function checks for a target timeout. If the target is seen, it
        calculates and publishes the left and right wheel velocities based on
        the stored v_cmd and w_cmd. If the target is lost, it publishes zero
        velocities to stop the robot.
        """
        now = time.time()
        if self.last_seen < 0.0 or (now - self.last_seen) > float(self.get_parameter('lost_timeout').value):
            # Target lost -> stop
            self.pub_left.publish(Float32(data=0.0))
            self.pub_right.publish(Float32(data=0.0))
            return

        # Differential drive kinematics: calculate left/right wheel speeds
        L = float(self.get_parameter('wheelbase').value)
        v = self.v_cmd
        w = self.w_cmd
        v_l = v - (w * L / 2.0)
        v_r = v + (w * L / 2.0)

        # (Safety) Scale down wheel velocities if they exceed the hard limit
        v_max_hw = float(self.get_parameter('v_max_hw').value)
        peak = max(abs(v_l), abs(v_r), 1e-9)
        if peak > v_max_hw:
            s = v_max_hw / peak
            v_l *= s
            v_r *= s

        self.pub_left.publish(Float32(data=float(v_l)))
        self.pub_right.publish(Float32(data=float(v_r)))


def main():
    """The main entry point for the ROS2 node."""
    rclpy.init()
    node = ObjectFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
