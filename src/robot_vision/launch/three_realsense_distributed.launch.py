from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='./tracking.onnx',
        description='Path to the YOLO model file (ABS or relative to CWD)'
    )
    
    model_path = LaunchConfiguration('model_path')

    return LaunchDescription([
        model_path_arg,
        
        # Node for Camera 1
        Node(
            package='robot_vision',
            executable='distributed_realsense',
            name='realsense_processor_1',
            output='screen',
            parameters=[
                {'camera_namespace': '/camera/cam_1/'},
                {'model_path': model_path}
            ]
        ),
        
        # Node for Camera 2
        Node(
            package='robot_vision',
            executable='distributed_realsense',
            name='realsense_processor_2',
            output='screen',
            parameters=[
                {'camera_namespace': '/camera/cam_2/'},
                {'model_path': model_path}
            ]
        ),
        
        # Node for Camera 3
        Node(
            package='robot_vision',
            executable='distributed_realsense',
            name='realsense_processor_3',
            output='screen',
            parameters=[
                {'camera_namespace': '/camera/cam_3/'},
                {'model_path': model_path}
            ]
        )
    ])
