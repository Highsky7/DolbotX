#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Node to Convert Steering Commands to Differential Drive Velocities.

This node subscribes to a steering angle topic and optionally a speed topic.
It calculates the required left and right wheel velocities for a differential
drive robot to achieve the desired motion, based on Ackermann steering geometry.
It also includes features like speed reduction on turns, command filtering,
and handling of external 'stop'/'go' commands.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32, Float64, String


def _isfinite(x: float) -> bool:
    """
    Check if a float is finite (not infinity or NaN).

    Args:
        x (float): The number to check.

    Returns:
        bool: True if the number is finite, False otherwise.
    """
    return math.isfinite(x)


class SteeringToDiff(Node):
    """
    Converts steering angle and speed commands to differential drive wheel speeds.
    """
    def __init__(self):
        """
        Initialize the SteeringToDiff node.

        Declares and loads all parameters, sets up publishers and subscribers,
        and starts the main control timer.
        """
        super().__init__('steering_to_diff')

        # ===== Basic Parameters =====
        self.declare_parameter('wheelbase', 0.60)              # L [m]
        self.declare_parameter('track_width', 0.516)           # W [m]
        self.declare_parameter('v_nominal', 0.50)              # Nominal speed [m/s] if use_speed_topic is false
        self.declare_parameter('max_speed', 0.6)               # Max linear speed [m/s]
        self.declare_parameter('kappa_gain', 3.0)              # Curvature gain
        self.declare_parameter('max_steering_angle', 1.5)      # [rad]
        self.declare_parameter('use_speed_topic', False)       # Use external speed topic or v_nominal
        self.declare_parameter('control_hz', 30.0)             # Control loop frequency

        # Safety Limits
        self.declare_parameter('speed_min', -6.0)              # [m/s]
        self.declare_parameter('speed_max',  6.0)              # [m/s] (clamp for topic input)
        self.declare_parameter('max_curvature', 0.0)           # [1/m], 0 means disabled (angle-based)

        # Topics
        self.declare_parameter('steering_angle_topic', '/steering_angle')
        self.declare_parameter('left_topic', '/cmd_vel_left')
        self.declare_parameter('right_topic', '/cmd_vel_right')
        self.declare_parameter('speed_topic', '/target_speed')
        self.declare_parameter('traffic_cmd_topic', '/traffic_command')   # stop/go command
        self.declare_parameter('supply_cmd_topic',  '/supply_command')    # stop/go command (same logic)

        # QoS
        self.declare_parameter('qos_history_depth', 10)
        self.declare_parameter('qos_reliability', 'RELIABLE')  # or BEST_EFFORT

        # Deadzones / Filters
        self.declare_parameter('v_deadzone', 1e-3)             # [m/s]
        self.declare_parameter('ang_deadzone', 1e-4)           # [rad]
        self.declare_parameter('cmd_vel_deadzone', 0.25)       # [m/s] Final output deadzone (added)
        self.declare_parameter('enable_filters', False)
        self.declare_parameter('angle_alpha', 0.2)
        self.declare_parameter('speed_alpha', 0.2)

        # === Speed Reduction on Turn ===
        self.declare_parameter('speed_reduce_enable', True)    # Enable/disable speed reduction on turn
        self.declare_parameter('speed_scale_min', 0.4)         # Minimum speed scale on full turn
        self.declare_parameter('speed_reduce_ang_max', 0.6)    # Angle at which minimum scale is applied

        # Internal state
        self._delta_raw = 0.0
        self._delta_filt = 0.0
        self._v_cmd_raw = 0.0
        self._v_cmd_filt = 0.0
        self._timer = None

        # Traffic (stop/go) state: 'go' or 'stop'
        self._traffic_mode = 'go'

        # Load initial parameters
        self._read_params()

        # Configure QoS
        qos = QoSProfile(depth=self.qos_history_depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (ReliabilityPolicy.RELIABLE
                           if self.qos_reliability == 'RELIABLE'
                           else ReliabilityPolicy.BEST_EFFORT)

        # Get topic names
        ang_topic = self.get_parameter('steering_angle_topic').value
        speed_topic = self.get_parameter('speed_topic').value
        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        traffic_topic = self.get_parameter('traffic_cmd_topic').value
        supply_topic  = self.get_parameter('supply_cmd_topic').value

        # Subscribers/Publishers
        self.create_subscription(Float64, ang_topic, self.cb_angle, qos)
        if self.use_speed_topic:
            self.create_subscription(Float64, speed_topic, self.cb_speed, qos)

        # Subscribe to stop/go commands (both topics use the same logic)
        self.create_subscription(String, traffic_topic, self.cb_traffic, qos)
        self.create_subscription(String, supply_topic,  self.cb_supply,  qos)

        self.pub_left = self.create_publisher(Float32, left_topic, qos)
        self.pub_right = self.create_publisher(Float32, right_topic, qos)

        # Initial speed state
        self._v_cmd_raw = self.v_nominal if not self.use_speed_topic else 0.0
        self._v_cmd_filt = self._v_cmd_raw

        # Start the timer
        self._restart_timer()

        # Dynamic parameter callback
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(
            f"[steering_to_diff] L={self.L:.3f} W={self.W:.3f} "
            f"v0={self.v_nominal:.3f} max_speed={self.max_speed:.3f} "
            f"use_speed_topic={self.use_speed_topic} hz={self.control_hz:.1f} "
            f"QoS={self.qos_reliability} depth={self.qos_history_depth} "
            f"filters={self.enable_filters} max_curv={self.max_curvature:.3f} "
            f"speed_reduce={self.speed_reduce_enable} "
            f"scale_min={self.speed_scale_min:.2f} ang_max={self.speed_reduce_ang_max:.3f} "
            f"cmd_vel_dz={self.cmd_vel_deadzone:.3f}"
        )

    def _read_params(self):
        """Read and validate all ROS parameters."""
        gp = self.get_parameter
        self.L = float(gp('wheelbase').value)
        self.W = float(gp('track_width').value)
        self.v_nominal = float(gp('v_nominal').value)
        self.max_speed = float(gp('max_speed').value)
        self.kappa_gain = float(gp('kappa_gain').value)
        self.max_steering_angle = float(gp('max_steering_angle').value)
        self.use_speed_topic = bool(gp('use_speed_topic').value)
        self.control_hz = float(gp('control_hz').value)

        self.speed_min = float(gp('speed_min').value)
        self.speed_max = float(gp('speed_max').value)
        if self.speed_min > self.speed_max:
            self.speed_min, self.speed_max = self.speed_max, self.speed_min

        self.max_curvature = float(gp('max_curvature').value)  # 0.0 => disabled

        self.qos_history_depth = int(gp('qos_history_depth').value)
        self.qos_reliability = str(gp('qos_reliability').value).upper()
        if self.qos_reliability not in ('RELIABLE', 'BEST_EFFORT'):
            self.get_logger().warn("qos_reliability must be RELIABLE or BEST_EFFORT; fallback RELIABLE")
            self.qos_reliability = 'RELIABLE'

        self.v_deadzone = float(gp('v_deadzone').value)
        self.ang_deadzone = float(gp('ang_deadzone').value)
        self.cmd_vel_deadzone = float(gp('cmd_vel_deadzone').value) # Added
        self.enable_filters = bool(gp('enable_filters').value)
        self.angle_alpha = float(max(1e-6, min(1.0, float(gp('angle_alpha').value))))
        self.speed_alpha = float(max(1e-6, min(1.0, float(gp('speed_alpha').value))))

        # Speed reduction parameters
        self.speed_reduce_enable = bool(gp('speed_reduce_enable').value)
        self.speed_scale_min = float(gp('speed_scale_min').value)
        self.speed_scale_min = max(0.0, min(1.0, self.speed_scale_min))  # Clamp to 0-1
        self.speed_reduce_ang_max = float(gp('speed_reduce_ang_max').value)
        self.speed_reduce_ang_max = max(1e-6, self.speed_reduce_ang_max)

        if self.L <= 0.0 or self.W <= 0.0 or self.max_speed <= 0.0:
            self.get_logger().error("Invalid params: L, W, max_speed must be > 0")

    def _restart_timer(self):
        """Cancel and restart the main control timer."""
        hz = max(self.control_hz, 1.0)
        if self._timer is not None:
            self._timer.cancel()
        self._timer = self.create_timer(1.0 / hz, self.on_timer)

    def _on_param_update(self, params):
        """
        Handle dynamic parameter updates.

        Args:
            params: A list of updated parameters.

        Returns:
            SetParametersResult: Indicates whether the update was successful.
        """
        names = [p.name for p in params]
        need_timer_restart = any(n in ('control_hz',) for n in names)
        prev_use_speed_topic = self.use_speed_topic

        self._read_params()

        if need_timer_restart:
            self._restart_timer()

        self.get_logger().info(f"Parameters updated: {sorted(names)}")
        if prev_use_speed_topic != self.use_speed_topic:
            self.get_logger().warn("use_speed_topic changed; restart node to re-create subscriptions if needed.")
        return SetParametersResult(successful=True)

    def _handle_stop_go(self, source: str, cmd: str):
        """
        Handle common 'stop' and 'go' commands from different topics.

        Args:
            source (str): The name of the source topic for logging.
            cmd (str): The command string, expected to be 'stop' or 'go'.
        """
        c = (cmd or '').strip().lower()
        if c == 'stop':
            self._traffic_mode = 'stop'
            self._publish_both(0.0, 0.0)  # Stop immediately
            self.get_logger().info(f"[{source}] STOP received -> Stopping.")
        elif c == 'go':
            self._traffic_mode = 'go'
            # Optionally, reset filter states here:
            # self._v_cmd_filt = self._v_cmd_raw
            # self._delta_filt = self._delta_raw
            self.get_logger().info(f"[{source}] GO received -> Resuming.")
        else:
            self.get_logger().warn(f"[{source}] Unknown command: '{cmd}' (available: 'stop'|'go')")

    def _compute_speed_scale(self, delta: float) -> float:
        """
        Calculate a speed scaling factor based on steering angle.
        Linearly decreases from 1.0 to speed_scale_min as |delta| increases.

        Args:
            delta (float): The current steering angle in radians.

        Returns:
            float: The speed scaling factor (between 0.0 and 1.0).
        """
        if not self.speed_reduce_enable:
            return 1.0
        a = abs(delta)
        if a <= self.ang_deadzone:
            return 1.0
        if a >= self.speed_reduce_ang_max:
            return self.speed_scale_min
        # Linear interpolation
        t = (a - self.ang_deadzone) / (self.speed_reduce_ang_max - self.ang_deadzone)
        scale = 1.0 + t * (self.speed_scale_min - 1.0)
        return max(self.speed_scale_min, min(1.0, scale))

    def cb_angle(self, msg: Float64):
        """
        Callback for the steering angle topic.

        Args:
            msg (Float64): The received steering angle message.
        """
        delta_in = float(msg.data)
        if not _isfinite(delta_in):
            self.get_logger().warn("Received non-finite steering angle; ignoring")
            return
        # Clamp angle to prevent extreme tan() values
        max_ang = max(1e-6, self.max_steering_angle)
        delta_in = max(-max_ang, min(max_ang, delta_in))
        self._delta_raw = delta_in
        if self.enable_filters:
            self._delta_filt = self.angle_alpha * self._delta_raw + (1.0 - self.angle_alpha) * self._delta_filt
        else:
            self._delta_filt = self._delta_raw

    def cb_speed(self, msg: Float64):
        """
        Callback for the target speed topic.

        Args:
            msg (Float64): The received speed message.
        """
        v_in = float(msg.data)
        if not _isfinite(v_in):
            self.get_logger().warn("Received non-finite speed; ignoring")
            return
        # Clamp the topic input explicitly for clarity
        v_in = max(self.speed_min, min(self.speed_max, v_in))
        self._v_cmd_raw = v_in
        if self.enable_filters:
            self._v_cmd_filt = self.speed_alpha * self._v_cmd_raw + (1.0 - self.speed_alpha) * self._v_cmd_filt
        else:
            self._v_cmd_filt = self._v_cmd_raw

    def cb_traffic(self, msg: String):
        """Callback for the traffic command topic."""
        self._handle_stop_go('traffic', msg.data)

    def cb_supply(self, msg: String):
        """Callback for the supply command topic."""
        self._handle_stop_go('supply', msg.data)

    def on_timer(self):
        """
        Main control loop, executed periodically by the timer.
        Calculates and publishes wheel velocities.
        """
        # If in 'stop' mode, publish zeros and return immediately
        if self._traffic_mode == 'stop':
            self._publish_both(0.0, 0.0)
            return

        # Select linear velocity v source
        if self.use_speed_topic:
            v = self._v_cmd_filt if self.enable_filters else self._v_cmd_raw
        else:
            v = self.v_nominal

        # Steering angle
        delta = self._delta_filt if self.enable_filters else self._delta_raw

        # Apply speed reduction scale based on angle (maintaining sign)
        scale = self._compute_speed_scale(delta)
        v = math.copysign(abs(v) * scale, v)

        # Low-speed deadzone
        if abs(v) < self.v_deadzone:
            self._publish_both(0.0, 0.0)
            return

        # Small-angle deadzone -> kappa=0
        if abs(delta) < self.ang_deadzone:
            kappa = 0.0
        else:
            # Angle-based curvature
            kappa = math.tan(delta) / max(self.L, 1e-6)

        # Apply gain and max curvature limit
        kappa *= self.kappa_gain
        if self.max_curvature > 0.0:
            lim = abs(self.max_curvature)
            if kappa > lim:
                kappa = lim
            elif kappa < -lim:
                kappa = -lim

        # Angular velocity
        omega = v * kappa

        # Left/right wheel velocities
        half_W = 0.5 * self.W
        v_left = v - omega * half_W
        v_right = v + omega * half_W

        # Normalizing saturation (maintaining ratio)
        peak = max(abs(v_left), abs(v_right), 1e-9)
        if peak > self.max_speed:
            scale2 = self.max_speed / peak
            v_left *= scale2
            v_right *= scale2

        # ==== MODIFIED: Apply final cmd_vel deadzone ====
        if abs(v_left) <= self.cmd_vel_deadzone:
            v_left = 0.0
        if abs(v_right) <= self.cmd_vel_deadzone:
            v_right = 0.0

        self._publish_both(v_left, v_right)

    def _publish_both(self, v_left: float, v_right: float):
        """Publish the left and right wheel velocities."""
        self.pub_left.publish(Float32(data=float(v_left)))
        self.pub_right.publish(Float32(data=float(v_right)))


def main():
    """Main entry point for the ROS2 node."""
    rclpy.init()
    node = SteeringToDiff()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()