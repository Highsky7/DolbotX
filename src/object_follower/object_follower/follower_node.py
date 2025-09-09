#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point        # ← Point로 구독
from std_msgs.msg import Float32


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class ObjectFollower(Node):
    """
    입력: /target_xy (geometry_msgs/Point)
         - RealSense optical 프레임이라고 가정(x:우+, y:아래+, z:전+), 단위 m
         - input_is_optical=False면 이미 x=전방+, y=좌+ 좌표계라고 간주

    출력: /cmd_vel_left, /cmd_vel_right (Float32)   # [m/s]

    속도 정책:
      - 거리 오차 기반 속도 v_dist = k_dist * (r - follow_distance)
      - 크루즈 상한 v_cruise (기본 0.6 m/s)
      - 조향 각도 |theta|가 커질수록 scale(θ)로 감속
      - 최종 v = min(max(0, v_dist), v_cruise) * scale(θ)
      - 하드 상한 |v| ≤ v_max_hw (절대 0.6m/s 초과 금지)
    """

    def __init__(self):
        super().__init__('object_follower')

        # ===== 토픽 =====
        self.declare_parameter('target_topic', '/target_xy')
        self.declare_parameter('left_topic', '/cmd_vel_left')
        self.declare_parameter('right_topic', '/cmd_vel_right')

        # ===== 입력 좌표계 플래그 =====
        # True  → 입력이 RealSense optical (x:우, y:아래, z:전)
        # False → 입력이 이미 추종 프레임 (x:전방, y:좌)
        self.declare_parameter('input_is_optical', True)

        # ===== 기구/제어 =====
        self.declare_parameter('wheelbase', 0.60)        # [m]
        self.declare_parameter('k_heading', 1.0)         # [rad/rad]
        self.declare_parameter('k_dist', 0.8)            # [m/s per m]
        self.declare_parameter('follow_distance', 1.0)   # [m]
        self.declare_parameter('reverse_ok', False)      # 후진 허용 여부

        # ===== 속도 정책(중요) =====
        self.declare_parameter('v_cruise', 0.6)          # 직진시 유지하고 싶은 최고 속도
        self.declare_parameter('v_max_hw', 0.6)          # 하드 상한(안전상 절대 초과 금지)

        # 턴 감속 스케일 파라미터(선형 보간)
        self.declare_parameter('turn_ang_straight', 0.05)  # [rad] 거의 직진
        self.declare_parameter('turn_ang_full', 0.6)       # [rad] 급회전
        self.declare_parameter('turn_scale_min', 0.35)     # 급회전시 속도 비율

        # ===== 카메라 오프셋 보정 =====
        self.declare_parameter('apply_camera_offset', True)
        self.declare_parameter('camera_offset_x', 0.27)   # 카메라가 베이스보다 앞(+x_follow) 0.27 m
        self.declare_parameter('camera_offset_y', 0.0)

        # ===== 타임아웃 =====
        self.declare_parameter('lost_timeout', 0.5)       # [s]

        # QoS
        qos = QoSProfile(
            depth=10, history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Pub/Sub
        left_topic   = self.get_parameter('left_topic').value
        right_topic  = self.get_parameter('right_topic').value
        target_topic = self.get_parameter('target_topic').value

        self.pub_left  = self.create_publisher(Float32, left_topic, qos)
        self.pub_right = self.create_publisher(Float32, right_topic, qos)
        self.create_subscription(Point, target_topic, self.cb_target_point, qos)

        # Timer
        self.timer = self.create_timer(1.0 / 50.0, self.on_timer)  # 50 Hz

        # 내부 상태
        self.last_seen = -1.0
        self.v_cmd = 0.0
        self.w_cmd = 0.0

        self.get_logger().info(
            f"[object_follower] sub={target_topic} pub=({left_topic}, {right_topic}) "
            f"v_cruise={self.get_parameter('v_cruise').value:.2f} m/s"
        )

    # ---- 각도 기반 스케일 ----
    def _turn_speed_scale(self, theta: float) -> float:
        a = abs(theta)
        ang0 = float(self.get_parameter('turn_ang_straight').value)
        ang1 = float(self.get_parameter('turn_ang_full').value)
        smin = float(self.get_parameter('turn_scale_min').value)
        smin = clamp(smin, 0.0, 1.0)

        if a <= ang0:
            return 1.0
        if a >= ang1:
            return smin
        # 선형 보간
        t = (a - ang0) / max(1e-9, (ang1 - ang0))
        return 1.0 + t * (smin - 1.0)

    # ---- 입력: geometry_msgs/Point (optical or follow frame) ----
    def cb_target_point(self, msg: Point):
        # 입력 좌표 (RealSense optical: X=우+, Y=아래+, Z=전+), 단위 m
        X_cam = float(msg.x)
        Y_cam = float(msg.y)
        Z_cam = float(msg.z)

        # 좌표계 변환 여부
        optical = bool(self.get_parameter('input_is_optical').value)

        if optical:
            # optical → 추종 프레임(x:전방+, y:좌+)
            x_follow = Z_cam
            y_follow = -X_cam
        else:
            # 이미 추종 프레임으로 들어온다고 가정
            x_follow = X_cam
            y_follow = Y_cam

        # 카메라 → 베이스 오프셋 보정(평행이동)
        if bool(self.get_parameter('apply_camera_offset').value):
            x_follow += float(self.get_parameter('camera_offset_x').value)
            y_follow += float(self.get_parameter('camera_offset_y').value)

        # 극좌표 (r, theta)
        r = math.hypot(x_follow, y_follow)
        theta = math.atan2(y_follow, x_follow)  # 좌(+)

        # 각속도 (P 제어)
        k_heading = float(self.get_parameter('k_heading').value)
        w = k_heading * theta

        # 거리 기반 속도 (P) + 정책
        follow_d = float(self.get_parameter('follow_distance').value)
        k_dist   = float(self.get_parameter('k_dist').value)
        v_cruise = float(self.get_parameter('v_cruise').value)
        v_max_hw = float(self.get_parameter('v_max_hw').value)

        v_dist = k_dist * (r - follow_d)      # 목표보다 멀면 양(+), 가까우면 음(-)
        if not bool(self.get_parameter('reverse_ok').value):
            v_dist = max(0.0, v_dist)         # 후진 금지

        # 턴 감속 스케일
        scale = self._turn_speed_scale(theta)
        v = min(max(0.0, v_dist), v_cruise) * scale

        # 하드 상한(절대 초과 금지)
        v = clamp(v, -v_max_hw, v_max_hw)

        # 상태 저장
        self.v_cmd = v
        self.w_cmd = w
        self.last_seen = time.time()

    def on_timer(self):
        now = time.time()
        if self.last_seen < 0.0 or (now - self.last_seen) > float(self.get_parameter('lost_timeout').value):
            # 타깃 미탐지 → 정지
            self.pub_left.publish(Float32(data=0.0))
            self.pub_right.publish(Float32(data=0.0))
            return

        # 차체 기구학: 좌/우 바퀴 속도
        L = float(self.get_parameter('wheelbase').value)
        v = self.v_cmd
        w = self.w_cmd
        v_l = v - (w * L / 2.0)
        v_r = v + (w * L / 2.0)

        # (안전) 좌/우도 절대값 0.6을 넘지 않도록 스케일링
        v_max_hw = float(self.get_parameter('v_max_hw').value)
        peak = max(abs(v_l), abs(v_r), 1e-9)
        if peak > v_max_hw:
            s = v_max_hw / peak
            v_l *= s
            v_r *= s

        self.pub_left.publish(Float32(data=float(v_l)))
        self.pub_right.publish(Float32(data=float(v_r)))


def main():
    rclpy.init()
    node = ObjectFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
