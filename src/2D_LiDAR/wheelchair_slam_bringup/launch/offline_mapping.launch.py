#!/usr/bin/env python3

import os
from datetime import datetime
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    with_rf2o = LaunchConfiguration('with_rf2o').perform(context).lower() == 'true'
    with_ekf = LaunchConfiguration('with_ekf').perform(context).lower() == 'true'
    session_root = LaunchConfiguration('session_root').perform(context)
    session_name = LaunchConfiguration('session_name').perform(context)
    save_period = LaunchConfiguration('save_period').perform(context)

    pkg_share = get_package_share_directory('wheelchair_slam_bringup')
    default_urdf = os.path.join(pkg_share, 'urdf', 'rplidar_myahrs.urdf')
    default_slam = os.path.join(pkg_share, 'config', 'slam.yaml')
    default_ekf = os.path.join(pkg_share, 'config', 'ekf_odom.yaml')

    urdf_file = LaunchConfiguration('robot_description_file').perform(context) or default_urdf
    slam_params_file = LaunchConfiguration('slam_params_file').perform(context) or default_slam
    ekf_params_file = LaunchConfiguration('ekf_params_file').perform(context) or default_ekf

    # Resolve to absolute paths and read URDF
    if not os.path.isabs(urdf_file):
        urdf_file = os.path.join(pkg_share, urdf_file)
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    if not os.path.isabs(slam_params_file):
        slam_params_file = os.path.join(pkg_share, slam_params_file)
    if not os.path.isabs(ekf_params_file):
        ekf_params_file = os.path.join(pkg_share, ekf_params_file)

    if not session_root:
        session_root = str(Path.home() / 'voice_based_wheelchair' / 'recordings')

    session_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not session_name:
        session_name = session_stamp

    root_path = Path(session_root).expanduser()
    session_dir = (root_path / session_name).resolve()
    map_dir = session_dir / 'map'
    traj_dir = session_dir / 'trajectory'
    for directory in (map_dir, traj_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        save_period_value = float(save_period)
    except (TypeError, ValueError):
        save_period_value = 5.0

    nodes = []

    # Static TFs from URDF
    nodes.append(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    ))

    # LiDAR odometry (rf2o) to provide /odom
    if with_rf2o:
        nodes.append(Node(
            package='rf2o_laser_odometry',
            executable='rf2o_laser_odometry',
            name='rf2o_laser_odometry',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ))

    # EKF fusion (optional)
    if with_ekf:
        nodes.append(Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_odom',
            output='screen',
            parameters=[ekf_params_file, {'use_sim_time': use_sim_time}]
        ))

    # SLAM Toolbox (async) with serialization overrides
    slam_overrides = {
        'serialization_file': str(traj_dir / 'pose_graph.cbor'),
        'serialization_format': 'cbor',
        'deserialize_pose_graph': False,
        'map_file_name': str(map_dir / 'slam_map'),
        'mode': 'mapping',
        'enable_interactive_mode': False,
        'use_sim_time': use_sim_time,
    }
    nodes.append(Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_file, slam_overrides]
    ))

    # Session recorder node to persist map & trajectory samples
    recorder_params = {
        'session_dir': str(session_dir),
        'map_topic': '/map',
        'pose_topic': '/slam_toolbox/pose',
        'save_period': save_period_value,
        'min_pose_spacing': 0.05,
        'min_time_spacing': 0.2,
        'use_sim_time': use_sim_time,
    }
    nodes.append(Node(
        package='wheelchair_slam_bringup',
        executable='session_recorder',
        name='session_recorder',
        output='screen',
        parameters=[recorder_params]
    ))

    return nodes


def generate_launch_description():
    pkg_share = get_package_share_directory('wheelchair_slam_bringup')
    default_urdf = os.path.join(pkg_share, 'urdf', 'rplidar_myahrs.urdf')
    default_slam = os.path.join(pkg_share, 'config', 'slam.yaml')
    default_ekf = os.path.join(pkg_share, 'config', 'ekf_odom.yaml')

    session_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulated time (rosbag --clock)'),
        DeclareLaunchArgument('with_rf2o', default_value='true', description='Run rf2o laser odometry'),
        DeclareLaunchArgument('with_ekf', default_value='false', description='Run robot_localization EKF fusion'),
        DeclareLaunchArgument('robot_description_file', default_value=default_urdf, description='URDF path'),
        DeclareLaunchArgument('slam_params_file', default_value=default_slam, description='slam_toolbox params YAML'),
        DeclareLaunchArgument('ekf_params_file', default_value=default_ekf, description='robot_localization EKF params YAML'),
        DeclareLaunchArgument('session_root', default_value=str(Path.home() / 'voice_based_wheelchair' / 'recordings'),
                              description='Directory where map/trajectory recordings are stored'),
        DeclareLaunchArgument('session_name', default_value=session_stamp,
                              description='Recording session name (default: timestamp)'),
        DeclareLaunchArgument('save_period', default_value='5.0',
                              description='Periodic save interval (seconds)'),
        OpaqueFunction(function=_launch_setup),
    ])
