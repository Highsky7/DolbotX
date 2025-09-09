from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_follower',
            executable='follower',
            name='object_follower',
            output='screen',
            parameters=[os.path.join(
                os.getenv('COLCON_PREFIX_PATH').split(':')[0],
                'share/object_follower/config/follower.yaml')]
        )
    ])
