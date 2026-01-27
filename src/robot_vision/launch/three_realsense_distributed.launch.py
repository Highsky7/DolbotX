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
        
        # Node for Camera 2
        Node(
            package='robot_vision',
            executable='distributed_realsense',
            name='realsense_processor_2',
            output='screen',
            parameters=[{'camera_namespace': '/camera/cam_2/'}]
        ),
        
        # Node for Camera 3
        Node(
            package='robot_vision',
            executable='distributed_realsense',
            name='realsense_processor_3',
            output='screen',
            parameters=[{'camera_namespace': '/camera/cam_3/'}]
        )
    ])
