from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('steering_to_diff')

    # 기본 파라미터 파일 경로 (패키지 내부 config/steering_to_diff.yaml)
    default_params = PathJoinSubstitution([pkg_share, 'config', 'params.yaml'])

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='YAML parameter file to load'
    )

    node = Node(
        package='steering_to_diff',
        executable='steering_to_diff',
        name='steering_to_diff',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    return LaunchDescription([
        params_file_arg,
        node
    ])
