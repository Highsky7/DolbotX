from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch argument for config file selection
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='imu_bias.yaml',  # Use pre-calibrated bias by default
        description='Config file name (default.yaml for runtime calibration, imu_bias.yaml for pre-calibrated)'
    )
    
    # Get config file path based on argument
    config_file = PathJoinSubstitution([
        FindPackageShare('imu_preprocess'), 'config', LaunchConfiguration('config')
    ])

    return LaunchDescription([
        config_arg,
        Node(
            package   = 'imu_preprocess',
            executable= 'imu_preprocess_node',
            name      = 'imu_preprocess_node',
            parameters= [config_file],
            output    = 'screen',                       # ← 로그를 콘솔로
            remappings= [('/imu/data', '/imu/data')]    # 필요 시 변경
        )
    ])
