from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node for Camera 1
        Node(
            package='robot_vision',
            executable='distributed_realsense',
            name='realsense_processor_1',
            output='screen',
            parameters=[{'camera_namespace': '/camera/cam_1/'}]
        ),
        Node(
            package='image_transport',
            executable='republish',
            name='republish_cam_1',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', '/camera/cam_1/viz/compressed'),
                ('out', '/camera/cam_1/viz/decoded')
            ],
            output='screen'
        ),
        
        # Node for Camera 2
        Node(
            package='robot_vision',
            executable='distributed_realsense',
            name='realsense_processor_2',
            output='screen',
            parameters=[{'camera_namespace': '/camera/cam_2/'}]
        ),
        Node(
            package='image_transport',
            executable='republish',
            name='republish_cam_2',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', '/camera/cam_2/viz/compressed'),
                ('out', '/camera/cam_2/viz/decoded')
            ],
            output='screen'
        ),
        
        # Node for Camera 3
        Node(
            package='robot_vision',
            executable='distributed_realsense',
            name='realsense_processor_3',
            output='screen',
            parameters=[{'camera_namespace': '/camera/cam_3/'}]
        ),
        Node(
            package='image_transport',
            executable='republish',
            name='republish_cam_3',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', '/camera/cam_3/viz/compressed'),
                ('out', '/camera/cam_3/viz/decoded')
            ],
            output='screen'
        )
    ])
