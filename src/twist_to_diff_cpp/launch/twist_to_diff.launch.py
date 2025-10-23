from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('twist_to_diff_cpp')
    joy_cfg   = os.path.join(pkg, 'config', 'joy.yaml')
    teleop_cfg= os.path.join(pkg, 'config', 'teleop_twist_joy.yaml')

    return LaunchDescription([
        Node(package='joy', executable='joy_node', name='joy_node',
             parameters=[joy_cfg]),
        Node(package='teleop_twist_joy', executable='teleop_node',
             name='teleop_twist_joy_node', parameters=[teleop_cfg]),
        Node(package='twist_to_diff_cpp', executable='twist_to_diff',
             name='twist_to_diff', parameters=[{'wheel_base': 0.5}])
    ])

