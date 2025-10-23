# twist_to_diff_cpp/launch/teleop_serial.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_t2d = get_package_share_directory('twist_to_diff_cpp')
    teleop_cfg = os.path.join(pkg_t2d, 'config', 'teleop_twist_joy.yaml')  # 당신 yaml(ROS2형식) 저장 위치
    joy_cfg    = os.path.join(pkg_t2d, 'config', 'joy.yaml')               # deadzone 등

    pkg_bridge = get_package_share_directory('wheel_serial_bridge')
    bridge_cfg = os.path.join(pkg_bridge, 'config', 'params.yaml')

    return LaunchDescription([
        Node(package='joy', executable='joy_node', name='joy_node', parameters=[joy_cfg]),
        
        Node(package='teleop_twist_joy', executable='teleop_node',
             name='teleop_twist_joy_node', parameters=[teleop_cfg]),
             
        Node(package='twist_to_diff_cpp', executable='twist_to_diff',
             name='twist_to_diff', parameters=[{'wheel_base': 0.5}]),
             
        Node(package='wheel_serial_bridge', executable='bridge_unified',
             name='wheel_serial_bridge_unified', parameters=[bridge_cfg]),
    ])

