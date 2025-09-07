#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32, Float64


class SteeringToDiff(Node):
    """
    입력:
      - /steering_angle (Float64, rad)
      - /target_speed  (Float64, m/s) ← use_speed_topic=True일 때

    출력:
      - /cmd_vel_left  (Float32, m/s)
      - /cmd_vel_right (Float32, m/s)

    모델:
      κ = tan(δ)/L,  ω = v*κ,
      v_L = v - ω*(W/2),  v_R = v + ω*(W/2)

    개선:
      - 좌/우 포화 시 정규화 스케일(비율 보존)
      - 저속/소각 데드존
      - 선택적 1차 IIR 필터
      - 동적 파라미터 업데이트 콜백
    """

    def __init__(self):
        super().__init__('steering_to_diff')

        # ===== 기본값(요청하신 값 반영) =====
        self.declare_parameter('wheelbase', 0.60)             # L [m]
        self.declare_parameter('track_width', 0.516)          # W [m]
        self.declare_parameter('v_nominal', 0.4)             # [m/s]
        self.declare_parameter('max_speed', 0.55)             # [m/s]
        self.declare_parameter('kappa_gain', 2.0)
        self.declare_parameter('max_steering_angle', 1.5)    # [rad]
        self.declare_parameter('use_speed_topic', False)
        self.declare_parameter('control_hz', 50.0)

        # 토픽
        self.declare_parameter('steering_angle_topic', '/steering_angle')
        self.declare_parameter('left_topic', '/cmd_vel_left')
        self.declare_parameter('right_topic', '/cmd_vel_right')
        self.declare_parameter('speed_topic', '/target_speed')

        # QoS
        self.declare_parameter('qos_history_depth', 10)
        self.declare_parameter('qos_reliability', 'RELIABLE')  # or BEST_EFFORT

        # 데드존/필터
        self.declare_parameter('v_deadzone', 1e-3)            # [m/s]
        self.declare_parameter('ang_deadzone', 1e-4)          # [rad]
        self.declare_parameter('enable_filters', False)
        self.declare_parameter('angle_alpha', 0.2)
        self.declare_parameter('speed_alpha', 0.2)

        # 파라미터 캐싱
        self._read_params()

        # 내부 상태
        self._delta_raw = 0.0
        self._delta_filt = 0.0
        self._v_cmd_raw = self.v_nominal
        self._v_cmd_filt = self.v_nominal

        # QoS 설정
        qos = QoSProfile(depth=self.qos_history_depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (ReliabilityPolicy.RELIABLE
                           if self.qos_reliability == 'RELIABLE'
                           else ReliabilityPolicy.BEST_EFFORT)

        # 토픽 이름
        ang_topic = self.get_parameter('steering_angle_topic').value
        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        speed_topic = self.get_parameter('speed_topic').value

        # 구독/발행자
        self.create_subscription(Float64, ang_topic, self.cb_angle, qos)
        if self.use_speed_topic:
            self.create_subscription(Float64, speed_topic, self.cb_speed, qos)

        self.pub_left = self.create_publisher(Float32, left_topic, qos)
        self.pub_right = self.create_publisher(Float32, right_topic, qos)

        # 타이머
        hz = max(self.control_hz, 1.0)
        self.timer = self.create_timer(1.0 / hz, self.on_timer)

        # 동적 파라미터 콜백
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(
            f"[steering_to_diff] L={self.L:.3f} W={self.W:.3f} "
            f"v0={self.v_nominal:.3f} max_speed={self.max_speed:.3f} "
            f"use_speed_topic={self.use_speed_topic} hz={hz:.1f} "
            f"QoS={self.qos_reliability} depth={self.qos_history_depth} "
            f"filters={self.enable_filters}"
        )

    # ---- 파라미터 로드/검증 ----
    def _read_params(self):
        self.L = float(self.get_parameter('wheelbase').value)
        self.W = float(self.get_parameter('track_width').value)
        self.v_nominal = float(self.get_parameter('v_nominal').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.kappa_gain = float(self.get_parameter('kappa_gain').value)
        self.max_steering_angle = float(self.get_parameter('max_steering_angle').value)
        self.use_speed_topic = bool(self.get_parameter('use_speed_topic').value)
        self.control_hz = float(self.get_parameter('control_hz').value)

        self.qos_history_depth = int(self.get_parameter('qos_history_depth').value)
        self.qos_reliability = str(self.get_parameter('qos_reliability').value).upper()

        self.v_deadzone = float(self.get_parameter('v_deadzone').value)
        self.ang_deadzone = float(self.get_parameter('ang_deadzone').value)
        self.enable_filters = bool(self.get_parameter('enable_filters').value)
        self.angle_alpha = float(self.get_parameter('angle_alpha').value)
        self.speed_alpha = float(self.get_parameter('speed_alpha').value)

        if self.L <= 0.0 or self.W <= 0.0 or self.max_speed <= 0.0:
            self.get_logger().error("Invalid params: L, W, max_speed must be > 0")
        if self.qos_reliability not in ('RELIABLE', 'BEST_EFFORT'):
            self.get_logger().warn("qos_reliability must be 'RELIABLE' or 'BEST_EFFORT'; using RELIABLE")
            self.qos_reliability = 'RELIABLE'
        self.angle_alpha = float(max(1e-6, min(1.0, self.angle_alpha)))
        self.speed_alpha = float(max(1e-6, min(1.0, self.speed_alpha)))

    def _on_param_update(self, params):
        changed = sorted([p.name for p in params])
        self._read_params()
        self.get_logger().info(f"Parameters updated: {changed}")
        return SetParametersResult(successful=True)

    # ---- 콜백 ----
    def cb_angle(self, msg: Float64):
        delta_in = float(msg.data)
        delta_in = max(-self.max_steering_angle, min(self.max_steering_angle, delta_in))
        self._delta_raw = delta_in
        if self.enable_filters:
            self._delta_filt = self.angle_alpha * self._delta_raw + (1.0 - self.angle_alpha) * self._delta_filt
        else:
            self._delta_filt = self._delta_raw

    def cb_speed(self, msg: Float64):
        self._v_cmd_raw = float(msg.data)
        if self.enable_filters:
            self._v_cmd_filt = self.speed_alpha * self._v_cmd_raw + (1.0 - self.speed_alpha) * self._v_cmd_filt
        else:
            self._v_cmd_filt = self._v_cmd_raw

    # ---- 제어 주기 ----
    def on_timer(self):
        # 선속도 v
        if self.use_speed_topic:
            v = self._v_cmd_filt if self.enable_filters else self._v_cmd_raw
        else:
            v = self.v_nominal

        # 저속 데드존
        if abs(v) < self.v_deadzone:
            self._publish_both(0.0, 0.0)
            return

        # 조향각
        delta = self._delta_filt if self.enable_filters else self._delta_raw

        # 소각 데드존 → κ=0
        if abs(delta) < self.ang_deadzone:
            kappa = 0.0
        else:
            kappa = math.tan(delta) / max(self.L, 1e-6)

        # 보정
        kappa *= self.kappa_gain

        # 각속도
        omega = v * kappa

        # 좌/우 속도
        v_left = v - omega * (self.W * 0.5)
        v_right = v + omega * (self.W * 0.5)

        # 정규화 포화(비율 유지)
        peak = max(abs(v_left), abs(v_right), 1e-6)
        if peak > self.max_speed:
            scale = self.max_speed / peak
            v_left *= scale
            v_right *= scale

        self._publish_both(v_left, v_right)

    def _publish_both(self, v_left: float, v_right: float):
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

