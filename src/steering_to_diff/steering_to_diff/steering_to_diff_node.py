#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Float64

class SteeringToDiff(Node):
    """
    입력:  /steering_angle (Float64, rad)
    출력:  /cmd_vel_left  (Float32, m/s)
          /cmd_vel_right (Float32, m/s)
    선택:  /target_speed (Float64, m/s) -- use_speed_topic=true일 때
    모델:
      κ = tan(δ)/L,  ω = v*κ,
      v_L = v - ω*(W/2),  v_R = v + ω*(W/2)
    """
    def __init__(self):
        super().__init__('steering_to_diff')

        # 파라미터
        self.declare_parameter('wheelbase', 0.40)           # L [m]
        self.declare_parameter('track_width', 0.36)         # W [m]
        self.declare_parameter('v_nominal', 0.40)           # 기본 속도 [m/s]
        self.declare_parameter('max_speed', 1.20)           # 제한 [m/s]
        self.declare_parameter('kappa_gain', 1.0)           # 스키드 보정
        self.declare_parameter('max_steering_angle', 0.60)  # 입력 각 제한 [rad]
        self.declare_parameter('use_speed_topic', False)
        self.declare_parameter('control_hz', 50.0)

        self.declare_parameter('steering_angle_topic', '/steering_angle')
        self.declare_parameter('left_topic', '/cmd_vel_left')
        self.declare_parameter('right_topic', '/cmd_vel_right')
        self.declare_parameter('speed_topic', '/target_speed')

        self.delta = 0.0
        self.v_cmd = float(self.get_parameter('v_nominal').value)
        self.use_speed_topic = bool(self.get_parameter('use_speed_topic').value)

        qos = QoSProfile(depth=10)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = ReliabilityPolicy.RELIABLE

        ang_topic   = self.get_parameter('steering_angle_topic').value
        left_topic  = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        speed_topic = self.get_parameter('speed_topic').value

        self.create_subscription(Float64, ang_topic, self.cb_angle, qos)
        if self.use_speed_topic:
            self.create_subscription(Float64, speed_topic, self.cb_speed, qos)

        self.pub_left  = self.create_publisher(Float32, left_topic, qos)
        self.pub_right = self.create_publisher(Float32, right_topic, qos)

        hz = float(self.get_parameter('control_hz').value)
        self.timer = self.create_timer(1.0 / max(hz, 1.0), self.on_timer)

        self.get_logger().info(
            f"[steering_to_diff] L={self.get_parameter('wheelbase').value:.3f} "
            f"W={self.get_parameter('track_width').value:.3f} "
            f"v0={self.get_parameter('v_nominal').value:.3f} "
            f"use_speed_topic={self.use_speed_topic}"
        )

    def cb_angle(self, msg: Float64):
        max_ang = float(self.get_parameter('max_steering_angle').value)
        self.delta = max(-max_ang, min(max_ang, float(msg.data)))

    def cb_speed(self, msg: Float64):
        self.v_cmd = float(msg.data)

    def on_timer(self):
        L = float(self.get_parameter('wheelbase').value)
        W = float(self.get_parameter('track_width').value)
        max_speed = float(self.get_parameter('max_speed').value)
        kappa_gain = float(self.get_parameter('kappa_gain').value)
        v = float(self.v_cmd) if self.use_speed_topic else float(self.get_parameter('v_nominal').value)

        if abs(self.delta) < 1e-6:
            kappa = 0.0
        else:
            kappa = math.tan(self.delta) / max(L, 1e-6)
        kappa *= kappa_gain

        omega = v * kappa
        v_left  = v - omega * (W * 0.5)
        v_right = v + omega * (W * 0.5)

        v_left  = max(-max_speed, min(max_speed, v_left))
        v_right = max(-max_speed, min(max_speed, v_right))

        self.pub_left.publish(Float32(data=float(v_left)))
        self.pub_right.publish(Float32(data=float(v_right)))

def main():
    rclpy.init()
    node = SteeringToDiff()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
