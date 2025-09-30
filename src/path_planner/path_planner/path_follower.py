#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

import tf2_ros

from std_msgs.msg import Int8, Float32
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PathFollower(Node):
    def __init__(self) -> None:
        super().__init__('path_follower')

        # 파라미터
        self.plan_topic = self.declare_parameter('plan_topic', '/plan') \
            .get_parameter_value().string_value
        self.cmd_vel_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel') \
            .get_parameter_value().string_value
        self.turn_topic = self.declare_parameter('turn_topic', '/turn_dir') \
            .get_parameter_value().string_value
        self.global_frame = self.declare_parameter('frame_id', 'map') \
            .get_parameter_value().string_value
        self.base_frame = self.declare_parameter('base_frame', 'base_link') \
            .get_parameter_value().string_value

        # 헤딩 정렬 허용 오차(도)
        self.heading_tol_deg = float(self.declare_parameter('heading_tolerance_deg', 5.0) \
            .get_parameter_value().double_value)
        self.heading_tol_rad = math.radians(self.heading_tol_deg)
        self.goal_tol = float(self.declare_parameter('goal_tol', 0.2) \
            .get_parameter_value().double_value)  # 목표점 반경 [m]
        # 순수추종(Pure Pursuit) 룩어헤드 거리 [m]
        self.lookahead_dist = float(self.declare_parameter('lookahead_distance', 0.8) \
            .get_parameter_value().double_value)

        # ALIGN 모드 회전 속도(rad/s)
        self.angular_speed = float(self.declare_parameter('angular_speed', 3.0) \
            .get_parameter_value().double_value)
        # DRIVE 모드 고정 전진 속도(m/s)
        self.linear_speed = float(self.declare_parameter('linear_speed', 2.0) \
            .get_parameter_value().double_value)
        # 차량 가상 휠베이스 길이 [m] - motor_bridge의 L_virtual과 동일해야 함
        self.wheelbase_m = float(self.declare_parameter('wheelbase', 0.40) \
            .get_parameter_value().double_value)
        # 조향각 명령 제한(도)
        self.max_steer_deg = float(self.declare_parameter('max_steer_deg', 35.0) \
            .get_parameter_value().double_value)
        self.rate_hz = float(self.declare_parameter('rate_hz', 20.0) \
            .get_parameter_value().double_value)

        # 시각화
        self.viz_enable = bool(self.declare_parameter('viz_enable', True) \
            .get_parameter_value().bool_value)
        self.marker_topic = self.declare_parameter('marker_topic', '/path_follower_markers') \
            .get_parameter_value().string_value
        self.arrow_len = float(self.declare_parameter('arrow_length', 0.4) \
            .get_parameter_value().double_value)
        self.arrow_w = float(self.declare_parameter('arrow_width', 0.05) \
            .get_parameter_value().double_value)
        self.arrow_h = float(self.declare_parameter('arrow_height', 0.05) \
            .get_parameter_value().double_value)
        self.wp_radius = float(self.declare_parameter('waypoint_radius', 0.1) \
            .get_parameter_value().double_value)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 퍼블리셔/서브스크라이버
        self.turn_pub = self.create_publisher(Int8, self.turn_topic, 10)
        self.steer_pub = self.create_publisher(Float32, '/auto_steer', 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.plan_sub = self.create_subscription(Path, self.plan_topic, self._on_plan, 10)
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, 10)
        self.heading_err_pub = self.create_publisher(Float32, '/heading_error_deg', 10)
        self.path_err_pub = self.create_publisher(Float32, '/path_error_m', 10)

        # 상태
        self._plan: List[Tuple[float, float]] = []
        self._plan_stamp = Time()
        self._idx = 0
        # 2단계: ALIGN → DRIVE (히스테리시스로 ALIGN으로 되돌아가지 않음)
        self._mode = 'ALIGN'

        # 제어 타이머
        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self._on_timer)

        self.get_logger().info(
            f"PathFollower up: plan={self.plan_topic}, cmd={self.cmd_vel_topic}, turn={self.turn_topic}"
        )

    def _on_plan(self, msg: Path) -> None:
        # 플랜을 (x, y) 리스트로 저장; frame_id가 존재하면 갱신
        self.global_frame = msg.header.frame_id or self.global_frame
        self._plan = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._plan_stamp = self.get_clock().now()
        self._idx = 0
        self._mode = 'ALIGN'
        self.get_logger().info(f"Received plan with {len(self._plan)} poses in frame '{self.global_frame}'")

    def _lookup_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.3)
            )
        except Exception as e:
            self.get_logger().warn(f"TF {self.global_frame}->{self.base_frame} unavailable: {e}")
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y

        qx = tf.transform.rotation.x
        qy = tf.transform.rotation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w

        # 쿼터니언으로부터 yaw 계산
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (x, y, yaw)

    def _publish_turn(self, direction: int) -> None:
        msg = Int8()
        msg.data = int(max(-1, min(1, direction)))
        self.turn_pub.publish(msg)

    def _publish_steer(self, deg: float) -> None:
        msg = Float32()
        msg.data = float(deg)
        self.steer_pub.publish(msg)

    def _stop(self) -> None:
        twist = Twist()
        self.cmd_pub.publish(twist)
        self._publish_turn(0)
        self._publish_steer(0.0)

    def _on_timer(self) -> None:
        # 플랜이 없을 때
        if not self._plan:
            self._stop()
            return

        pose = self._lookup_pose()
        if pose is None:
            self._stop()
            return

        x, y, yaw = pose

        # 현재 위치에서 룩어헤드 목표점 계산
        (wx, wy), dist, gid = self._find_lookahead_target(x, y)
        # 인덱스 보수적으로 전진
        if gid > self._idx:
            self._idx = gid - 1 if gid - 1 >= 0 else 0

        # 목표 도달 판단
        # 마지막 점 근처이고 goal_tol 이내면 정지
        if gid >= len(self._plan) - 1 and dist <= self.goal_tol:
            self.get_logger().info_once("Goal reached. Stopping.")
            self._stop()
            # 정지 상태에서 마지막 마커 발행
            if self.viz_enable:
                self._publish_markers(x, y, yaw, yaw, (wx, wy))
            return

        # 헤딩 제어 (ALIGN → DRIVE 2단계)
        # 정렬 기준: 룩어헤드 목표점 방향을 향하도록
        yaw_to_target = math.atan2(wy - y, wx - x) if dist > 1e-6 else yaw
        yaw_err = normalize_angle(yaw_to_target - yaw)

        twist = Twist()
        if self._mode == 'ALIGN':
            if abs(yaw_err) > self.heading_tol_rad:
                # /turn_dir 만 사용하여 제자리 회전
                twist.linear.x = 0.0
                twist.angular.z = 0.0  # bridge가 /turn_dir로 회전 수행
                self._publish_turn(-1 if yaw_err > 0.0 else +1)
                self._publish_steer(0.0)
            else:
                # 정렬 완료 → DRIVE로 전환
                self._mode = 'DRIVE'
                self._publish_turn(0)
                # 같은 주기 내에서 DRIVE 동작으로 이어짐

        if self._mode == 'DRIVE':
            # 순수추종(Pure Pursuit) 조향각 계산
            delta_deg = self._pure_pursuit_steer_deg(x, y, yaw, wx, wy)
            # 명령 발행
            twist.linear.x = self.linear_speed
            twist.angular.z = 0.0
            self._publish_turn(0)
            self._publish_steer(delta_deg)

        self.cmd_pub.publish(twist)

        # 오차 토픽
        # 헤딩 오차(도)
        self.heading_err_pub.publish(Float32(data=math.degrees(yaw_err)))
        # 횡방향 오차(부호, m) - 차량 좌표계에서 룩어헤드의 y값
        dx = wx - x
        dy = wy - y
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        y_v = -sin_y * dx + cos_y * dy
        self.path_err_pub.publish(Float32(data=float(y_v)))

        # RViz 마커 발행
        if self.viz_enable:
            self._publish_markers(x, y, yaw, yaw_to_target, (wx, wy))

    def _publish_markers(self, x: float, y: float, yaw: float, target_yaw: float, waypoint: Tuple[float, float]) -> None:
        now = self.get_clock().now().to_msg()
        # 현재 헤딩 화살표 (id=0)
        arrow = Marker()
        arrow.header.frame_id = self.global_frame
        arrow.header.stamp = now
        arrow.ns = 'path_follower'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = x
        arrow.pose.position.y = y
        arrow.pose.position.z = 0.05
        # yaw로부터 방향 쿼터니언 구성
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        arrow.pose.orientation.z = qz
        arrow.pose.orientation.w = qw
        arrow.scale.x = self.arrow_len
        arrow.scale.y = self.arrow_w
        arrow.scale.z = self.arrow_h
        arrow.color.a = 0.8
        arrow.color.r = 0.2
        arrow.color.g = 0.5
        arrow.color.b = 1.0
        arrow.lifetime = Duration(seconds=0.5).to_msg()
        self.marker_pub.publish(arrow)

        # 룩어헤드 포인트 구체 (id=1)
        wx, wy = waypoint
        wp = Marker()
        wp.header.frame_id = self.global_frame
        wp.header.stamp = now
        wp.ns = 'path_follower'
        wp.id = 1
        wp.type = Marker.SPHERE
        wp.action = Marker.ADD
        wp.pose.position.x = wx
        wp.pose.position.y = wy
        wp.pose.position.z = 0.05
        wp.pose.orientation.w = 1.0
        wp.scale.x = self.wp_radius * 2.0
        wp.scale.y = self.wp_radius * 2.0
        wp.scale.z = self.wp_radius * 2.0
        wp.color.a = 0.8
        wp.color.r = 1.0
        wp.color.g = 0.2
        wp.color.b = 0.2
        wp.lifetime = Duration(seconds=0.5).to_msg()
        self.marker_pub.publish(wp)

        # 목표 헤딩 화살표 (id=2)
        targ = Marker()
        targ.header.frame_id = self.global_frame
        targ.header.stamp = now
        targ.ns = 'path_follower'
        targ.id = 2
        targ.type = Marker.ARROW
        targ.action = Marker.ADD
        targ.pose.position.x = x
        targ.pose.position.y = y
        targ.pose.position.z = 0.06
        qz2 = math.sin(target_yaw * 0.5)
        qw2 = math.cos(target_yaw * 0.5)
        targ.pose.orientation.z = qz2
        targ.pose.orientation.w = qw2
        targ.scale.x = self.arrow_len * 0.9
        targ.scale.y = self.arrow_w
        targ.scale.z = self.arrow_h
        targ.color.a = 0.9
        targ.color.r = 0.1
        targ.color.g = 1.0
        targ.color.b = 0.2
        targ.lifetime = Duration(seconds=0.5).to_msg()
        self.marker_pub.publish(targ)

        # 순수추종 원호 (id=3)
        # 차량 좌표계의 룩어헤드로부터 곡률 계산
        dx = wx - x
        dy = wy - y
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x_v =  cos_y * dx + sin_y * dy
        y_v = -sin_y * dx + cos_y * dy
        Ld = max(math.hypot(x_v, y_v), 1e-6)
        kappa = 2.0 * y_v / (Ld * Ld)  # 곡률

        arc = Marker()
        arc.header.frame_id = self.global_frame
        arc.header.stamp = now
        arc.ns = 'path_follower'
        arc.id = 3
        arc.type = Marker.LINE_STRIP
        arc.action = Marker.ADD
        arc.scale.x = 0.03
        arc.color.a = 0.9
        arc.color.r = 0.7
        arc.color.g = 0.2
        arc.color.b = 1.0
        arc.lifetime = Duration(seconds=0.5).to_msg()

        points: list[Point] = []
        arc_len = max(self.lookahead_dist * 2.0, 0.5)
        steps = 40
        if abs(kappa) < 1e-6:
            # 전방 직선
            for i in range(steps + 1):
                s = arc_len * (i / steps)
                xv = s
                yv = 0.0
                pw = Point()
                pw.x = x + cos_y * xv - sin_y * yv
                pw.y = y + sin_y * xv + cos_y * yv
                pw.z = 0.05
                points.append(pw)
        else:
            R = 1.0 / kappa
            for i in range(steps + 1):
                s = arc_len * (i / steps)
                xv = R * math.sin(s / R)
                yv = R * (1.0 - math.cos(s / R))
                pw = Point()
                pw.x = x + cos_y * xv - sin_y * yv
                pw.y = y + sin_y * xv + cos_y * yv
                pw.z = 0.05
                points.append(pw)
        arc.points = points
        self.marker_pub.publish(arc)

    def _find_lookahead_target(self, x: float, y: float) -> Tuple[Tuple[float, float], float, int]:
        # 룩어헤드 거리 이상인 첫 번째 점을 찾는다. 없다면 마지막 점을 사용함
        best_idx = len(self._plan) - 1
        best_pt = self._plan[best_idx]
        best_dist = math.hypot(best_pt[0] - x, best_pt[1] - y)

        start = max(self._idx, 0)
        Ld = max(self.lookahead_dist, 1e-3)
        for i in range(start, len(self._plan)):
            px, py = self._plan[i]
            d = math.hypot(px - x, py - y)
            if d >= Ld:
                return (px, py), d, i
        return best_pt, best_dist, best_idx

    def _pure_pursuit_steer_deg(self, x: float, y: float, yaw: float, gx: float, gy: float) -> float:
        # 목표점을 base_link 좌표계로 변환
        dx = gx - x
        dy = gy - y
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x_v =  cos_y * dx + sin_y * dy
        y_v = -sin_y * dx + cos_y * dy

        Ld = max(math.hypot(x_v, y_v), 1e-6)
        # 순수추종 공식: delta = atan2(2*L*y, Ld^2)
        delta_rad = math.atan2(2.0 * self.wheelbase_m * y_v, Ld * Ld)
        delta_deg = math.degrees(delta_rad)
        # 제한(clamp)
        if self.max_steer_deg > 0.0:
            delta_deg = max(-self.max_steer_deg, min(self.max_steer_deg, delta_deg))
        return delta_deg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
