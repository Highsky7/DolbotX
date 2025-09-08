#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32, Float64, String


def _isfinite(x: float) -> bool:
    return math.isfinite(x)


class SteeringToDiff(Node):
    def __init__(self):
        super().__init__('steering_to_diff')

        # ===== 기본 파라미터 =====
        self.declare_parameter('wheelbase', 0.60)              # L [m]
        self.declare_parameter('track_width', 0.516)           # W [m]
        self.declare_parameter('v_nominal', 0.50)              # [m/s]
        self.declare_parameter('max_speed', 0.6)              # [m/s]
        self.declare_parameter('kappa_gain', 3.0)
        self.declare_parameter('max_steering_angle', 1.5)      # [rad]
        self.declare_parameter('use_speed_topic', False)
        self.declare_parameter('control_hz', 30.0)

        # 안전 상/하한
        self.declare_parameter('speed_min', -6.0)              # [m/s]
        self.declare_parameter('speed_max',  6.0)              # [m/s] (토픽 입력 클램프)
        self.declare_parameter('max_curvature', 0.0)           # [1/m], 0이면 비활성(=각도 기반)

        # 토픽
        self.declare_parameter('steering_angle_topic', '/steering_angle')
        self.declare_parameter('left_topic', '/cmd_vel_left')
        self.declare_parameter('right_topic', '/cmd_vel_right')
        self.declare_parameter('speed_topic', '/target_speed')
        self.declare_parameter('traffic_cmd_topic', '/traffic_command')   # 정지/재개 명령
        self.declare_parameter('supply_cmd_topic',  '/supply_command')    # 정지/재개 명령(동일 로직)

        # QoS
        self.declare_parameter('qos_history_depth', 10)
        self.declare_parameter('qos_reliability', 'RELIABLE')  # or BEST_EFFORT

        # 데드존/필터
        self.declare_parameter('v_deadzone', 1e-3)             # [m/s]
        self.declare_parameter('ang_deadzone', 1e-4)           # [rad]
        self.declare_parameter('enable_filters', False)
        self.declare_parameter('angle_alpha', 0.2)
        self.declare_parameter('speed_alpha', 0.2)

        # === 조향각 기반 감속 ===
        self.declare_parameter('speed_reduce_enable', True)    # 각도 기반 감속 On/Off
        self.declare_parameter('speed_scale_min', 0.4)         # 급회전 시 최소 스케일
        self.declare_parameter('speed_reduce_ang_max', 0.6)    # 이 각도 이상이면 최소 스케일

        # 내부 상태
        self._delta_raw = 0.0
        self._delta_filt = 0.0
        self._v_cmd_raw = 0.0
        self._v_cmd_filt = 0.0
        self._timer = None

        # 트래픽(정지/재개) 상태: 'go' or 'stop'
        self._traffic_mode = 'go'

        # 초기 파라미터 로드
        self._read_params()

        # QoS 구성
        qos = QoSProfile(depth=self.qos_history_depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (ReliabilityPolicy.RELIABLE
                           if self.qos_reliability == 'RELIABLE'
                           else ReliabilityPolicy.BEST_EFFORT)

        # 토픽 이름들
        ang_topic = self.get_parameter('steering_angle_topic').value
        speed_topic = self.get_parameter('speed_topic').value
        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        traffic_topic = self.get_parameter('traffic_cmd_topic').value
        supply_topic  = self.get_parameter('supply_cmd_topic').value

        # 구독자/퍼블리셔
        self.create_subscription(Float64, ang_topic, self.cb_angle, qos)
        if self.use_speed_topic:
            self.create_subscription(Float64, speed_topic, self.cb_speed, qos)

        # 정지/재개 명령 구독(두 토픽 동일 로직)
        self.create_subscription(String, traffic_topic, self.cb_traffic, qos)
        self.create_subscription(String, supply_topic,  self.cb_supply,  qos)

        self.pub_left = self.create_publisher(Float32, left_topic, qos)
        self.pub_right = self.create_publisher(Float32, right_topic, qos)

        # 시작 속도 상태
        self._v_cmd_raw = self.v_nominal if not self.use_speed_topic else 0.0
        self._v_cmd_filt = self._v_cmd_raw

        # 타이머 시작
        self._restart_timer()

        # 동적 파라미터 콜백
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(
            f"[steering_to_diff] L={self.L:.3f} W={self.W:.3f} "
            f"v0={self.v_nominal:.3f} max_speed={self.max_speed:.3f} "
            f"use_speed_topic={self.use_speed_topic} hz={self.control_hz:.1f} "
            f"QoS={self.qos_reliability} depth={self.qos_history_depth} "
            f"filters={self.enable_filters} max_curv={self.max_curvature:.3f} "
            f"speed_reduce={self.speed_reduce_enable} "
            f"scale_min={self.speed_scale_min:.2f} ang_max={self.speed_reduce_ang_max:.3f}"
        )

    # ---- 파라미터 로드/검증 ----
    def _read_params(self):
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

        self.max_curvature = float(gp('max_curvature').value)  # 0.0 => 비활성

        self.qos_history_depth = int(gp('qos_history_depth').value)
        self.qos_reliability = str(gp('qos_reliability').value).upper()
        if self.qos_reliability not in ('RELIABLE', 'BEST_EFFORT'):
            self.get_logger().warn("qos_reliability must be RELIABLE or BEST_EFFORT; fallback RELIABLE")
            self.qos_reliability = 'RELIABLE'

        self.v_deadzone = float(gp('v_deadzone').value)
        self.ang_deadzone = float(gp('ang_deadzone').value)
        self.enable_filters = bool(gp('enable_filters').value)
        self.angle_alpha = float(max(1e-6, min(1.0, float(gp('angle_alpha').value))))
        self.speed_alpha = float(max(1e-6, min(1.0, float(gp('speed_alpha').value))))

        # 감속 파라미터
        self.speed_reduce_enable = bool(gp('speed_reduce_enable').value)
        self.speed_scale_min = float(gp('speed_scale_min').value)
        self.speed_scale_min = max(0.0, min(1.0, self.speed_scale_min))  # 0~1 보정
        self.speed_reduce_ang_max = float(gp('speed_reduce_ang_max').value)
        self.speed_reduce_ang_max = max(1e-6, self.speed_reduce_ang_max)

        if self.L <= 0.0 or self.W <= 0.0 or self.max_speed <= 0.0:
            self.get_logger().error("Invalid params: L, W, max_speed must be > 0")

    def _restart_timer(self):
        hz = max(self.control_hz, 1.0)
        if self._timer is not None:
            self._timer.cancel()
        self._timer = self.create_timer(1.0 / hz, self.on_timer)

    def _on_param_update(self, params):
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

    # ---- 공통 stop/go 처리 ----
    def _handle_stop_go(self, source: str, cmd: str):
        c = (cmd or '').strip().lower()
        if c == 'stop':
            self._traffic_mode = 'stop'
            self._publish_both(0.0, 0.0)  # 즉시 정지
            self.get_logger().info(f"[{source}] STOP 수신 → 정지 모드")
        elif c == 'go':
            self._traffic_mode = 'go'
            # 필요 시 필터 상태 원복:
            # self._v_cmd_filt = self._v_cmd_raw
            # self._delta_filt = self._delta_raw
            self.get_logger().info(f"[{source}] GO 수신 → 주행 모드 복귀")
        else:
            self.get_logger().warn(f"[{source}] 알 수 없는 명령: '{cmd}' (사용 가능: 'stop'|'go')")

    # ---- 각도 기반 속도 스케일 계산 ----
    def _compute_speed_scale(self, delta: float) -> float:
        """|delta|가 커질수록 [1.0 -> speed_scale_min]로 선형 감소."""
        if not self.speed_reduce_enable:
            return 1.0
        a = abs(delta)
        if a <= self.ang_deadzone:
            return 1.0
        if a >= self.speed_reduce_ang_max:
            return self.speed_scale_min
        # 선형 보간
        t = (a - self.ang_deadzone) / (self.speed_reduce_ang_max - self.ang_deadzone)
        scale = 1.0 + t * (self.speed_scale_min - 1.0)
        return max(self.speed_scale_min, min(1.0, scale))

    # ---- 콜백 ----
    def cb_angle(self, msg: Float64):
        delta_in = float(msg.data)
        if not _isfinite(delta_in):
            self.get_logger().warn("Received non-finite steering angle; ignoring")
            return
        # 각도 클램프(과도한 tan 폭주 방지)
        max_ang = max(1e-6, self.max_steering_angle)
        delta_in = max(-max_ang, min(max_ang, delta_in))
        self._delta_raw = delta_in
        if self.enable_filters:
            self._delta_filt = self.angle_alpha * self._delta_raw + (1.0 - self.angle_alpha) * self._delta_filt
        else:
            self._delta_filt = self._delta_raw

    def cb_speed(self, msg: Float64):
        v_in = float(msg.data)
        if not _isfinite(v_in):
            self.get_logger().warn("Received non-finite speed; ignoring")
            return
        # 토픽 입력 자체를 명시적으로 클램프(로그 가독성↑)
        v_in = max(self.speed_min, min(self.speed_max, v_in))
        self._v_cmd_raw = v_in
        if self.enable_filters:
            self._v_cmd_filt = self.speed_alpha * self._v_cmd_raw + (1.0 - self.speed_alpha) * self._v_cmd_filt
        else:
            self._v_cmd_filt = self._v_cmd_raw

    def cb_traffic(self, msg: String):
        self._handle_stop_go('traffic', msg.data)

    def cb_supply(self, msg: String):
        self._handle_stop_go('supply', msg.data)

    # ---- 제어 주기 ----
    def on_timer(self):
        # 트래픽 정지 상태면 즉시 0 발행 후 리턴
        if self._traffic_mode == 'stop':
            self._publish_both(0.0, 0.0)
            return

        # 선속도 v (명령 소스 선택)
        if self.use_speed_topic:
            v = self._v_cmd_filt if self.enable_filters else self._v_cmd_raw
        else:
            v = self.v_nominal

        # 조향각
        delta = self._delta_filt if self.enable_filters else self._delta_raw

        # 각도 기반 감속 스케일 적용 (부호 유지)
        scale = self._compute_speed_scale(delta)
        v = math.copysign(abs(v) * scale, v)

        # 저속 데드존
        if abs(v) < self.v_deadzone:
            self._publish_both(0.0, 0.0)
            return

        # 소각 데드존 → κ=0
        if abs(delta) < self.ang_deadzone:
            kappa = 0.0
        else:
            # 각도 기반 곡률
            kappa = math.tan(delta) / max(self.L, 1e-6)

        # 게인 및 최대 곡률 적용
        kappa *= self.kappa_gain
        if self.max_curvature > 0.0:
            lim = abs(self.max_curvature)
            if kappa > lim:
                kappa = lim
            elif kappa < -lim:
                kappa = -lim

        # 각속도
        omega = v * kappa

        # 좌/우 속도
        half_W = 0.5 * self.W
        v_left = v - omega * half_W
        v_right = v + omega * half_W

        # 정규화 포화(비율 유지)
        peak = max(abs(v_left), abs(v_right), 1e-9)
        if peak > self.max_speed:
            scale2 = self.max_speed / peak
            v_left *= scale2
            v_right *= scale2

        # 아주 작은 잔여값 제거(수치 잡음)
        if abs(v_left) < 1e-6:
            v_left = 0.0
        if abs(v_right) < 1e-6:
            v_right = 0.0

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
