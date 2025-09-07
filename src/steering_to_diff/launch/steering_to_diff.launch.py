from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='steering_to_diff',
            executable='steering_to_diff',
            name='steering_to_diff',
            output='screen',
            # parameters 인자를 제거
        )
    ])

