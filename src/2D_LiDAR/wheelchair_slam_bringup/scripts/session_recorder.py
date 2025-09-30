#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SessionRecorder(Node):
    """맵 픽셀과 SLAM 궤적을 주기적으로 디스크에 저장한다."""

    def __init__(self) -> None:
        super().__init__('session_recorder')

        self.session_dir = Path(self.declare_parameter('session_dir').value).expanduser().resolve()
        self.map_topic = self.declare_parameter('map_topic', '/map').value
        self.pose_topic = self.declare_parameter('pose_topic', '/slam_toolbox/pose').value
        self.save_period = float(self.declare_parameter('save_period', 5.0).value)
        self.min_pose_spacing = float(self.declare_parameter('min_pose_spacing', 0.05).value)
        self.min_time_spacing = float(self.declare_parameter('min_time_spacing', 0.2).value)

        if self.save_period <= 0.0:
            self.save_period = 5.0

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.map_dir = (self.session_dir / 'map').resolve()
        self.traj_dir = (self.session_dir / 'trajectory').resolve()
        self.map_dir.mkdir(parents=True, exist_ok=True)
        self.traj_dir.mkdir(parents=True, exist_ok=True)

        self._map_path = self.map_dir / 'slam_map'
        self._traj_path = self.traj_dir / 'slam_path.yaml'

        self._map_msg: Optional[OccupancyGrid] = None
        self._poses: List[PoseStamped] = []
        self._last_pose: Optional[PoseStamped] = None
        self._last_pose_time: Optional[float] = None

        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, 10)
        self.create_subscription(PoseStamped, self.pose_topic, self._on_pose, 50)
        self.create_timer(self.save_period, self._periodic_save)

        self.get_logger().info(
            f"SessionRecorder writing maps to {self._map_path.with_suffix('.pgm')} "
            f"and trajectories to {self._traj_path}"
        )

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._map_msg = msg

    def _on_pose(self, msg: PoseStamped) -> None:
        if self._last_pose is None:
            self._store_pose(msg)
            return

        last_time = self._last_pose_time or _stamp_to_sec(self._last_pose.header.stamp)
        current_time = _stamp_to_sec(msg.header.stamp)
        if current_time - last_time < self.min_time_spacing:
            return

        if self._pose_distance(self._last_pose, msg) < self.min_pose_spacing:
            return

        self._store_pose(msg)

    def _pose_distance(self, a: PoseStamped, b: PoseStamped) -> float:
        dx = a.pose.position.x - b.pose.position.x
        dy = a.pose.position.y - b.pose.position.y
        return math.hypot(dx, dy)

    def _store_pose(self, msg: PoseStamped) -> None:
        copy_msg = PoseStamped()
        copy_msg.header = msg.header
        copy_msg.pose = msg.pose
        self._poses.append(copy_msg)
        self._last_pose = copy_msg
        self._last_pose_time = _stamp_to_sec(copy_msg.header.stamp)

    def _periodic_save(self) -> None:
        if self._map_msg is not None:
            self._write_map(self._map_msg)
        if self._poses:
            self._write_path(self._poses)

    def destroy_node(self) -> bool:
        self._periodic_save()
        return super().destroy_node()

    def _write_map(self, msg: OccupancyGrid) -> None:
        pgm_path = self._map_path.with_suffix('.pgm')
        yaml_path = self._map_path.with_suffix('.yaml')

        width = msg.info.width
        height = msg.info.height
        data = msg.data
        image = bytearray(width * height)
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                value = data[idx]
                target_idx = (height - 1 - y) * width + x
                if value == 0:
                    image[target_idx] = 254
                elif value > 0:
                    image[target_idx] = 0
                else:
                    image[target_idx] = 205

        pgm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pgm_path, 'wb') as f:
            f.write(f"P5\n{width} {height}\n255\n".encode('ascii'))
            f.write(image)

        origin = msg.info.origin
        resolution = msg.info.resolution
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(
                "image: " + pgm_path.name + "\n"
                + f"resolution: {resolution}\n"
                + f"origin: [{origin.position.x}, {origin.position.y}, 0.0]\n"
                + "negate: 0\n"
                + "occupied_thresh: 0.65\n"
                + "free_thresh: 0.196\n"
            )

    def _write_path(self, poses: List[PoseStamped]) -> None:
        self._traj_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["poses:"]
        for pose in poses:
            xyz = pose.pose.position
            quat = pose.pose.orientation
            stamp = pose.header.stamp
            lines.append(
                f"  - time: {_stamp_to_sec(stamp):.6f}\n"
                f"    position: [{xyz.x:.6f}, {xyz.y:.6f}, {xyz.z:.6f}]\n"
                f"    orientation: [{quat.x:.6f}, {quat.y:.6f}, {quat.z:.6f}, {quat.w:.6f}]"
            )
        with open(self._traj_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SessionRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
