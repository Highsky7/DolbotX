from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('steering_to_diff')
    params = os.path.join(pkg_share, 'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='steering_to_diff',
            executable='steering_to_diff',
            name='steering_to_diff',
            output='screen',
            parameters=[params],
        )
    ])
