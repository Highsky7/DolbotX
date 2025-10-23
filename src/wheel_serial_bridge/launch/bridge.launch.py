from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('wheel_serial_bridge')
    cfg = os.path.join(pkg, 'config', 'params.yaml')
    return LaunchDescription([
        Node(package='wheel_serial_bridge', executable='bridge_unified',
             name='wheel_serial_bridge_unified', parameters=[cfg])
    ])

