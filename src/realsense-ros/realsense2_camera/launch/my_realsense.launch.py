import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # ===================================================================================
    # 1. realsense2_camera의 launch 파일 포함
    # ===================================================================================
    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    # ROS 1의 <arg>들은 launch_arguments의 딕셔너리로 전달됩니다.
    # ROS 2 Humble 버전에 맞는 인자 이름으로 수정하고, 충돌을 막기 위한 설정을 추가합니다.
    realsense_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            # =======================================================================
            # 핵심 수정 사항: 충돌 방지를 위해 realsense 노드의 자체 URDF 및 TF 발행 기능 비활성화
            # =======================================================================
            'publish_robot_description': 'false', # Realsense 노드가 자체 URDF를 발행하지 않도록 설정
            
            # ROS 1의 align_depth와 동일한 기능
            'align_depth.enable': 'true',
            
            # 포인트 클라우드 활성화
            'pointcloud.enable': 'true',
            
            # ROS 1의 enable_sync와 동일한 기능
            'enable_sync': 'true',

            # IMU 센서 활성화 (기존 파일에도 있었으므로 유지)
            'enable_gyro': 'true',
            'enable_accel': 'true',
            
            # =======================================================================
            # 수정 사항: ROS 2에 맞는 프로파일 인자 이름으로 변경
            # =======================================================================
            # depth_module.profile -> depth_profile
            'depth_module.profile': '640x480x30',
            
            # rgb_camera.profile -> rgb_camera_profile
            'rgb_camera.profile': '640x480x30',
        }.items()
    )

    # ===================================================================================
    # 2. robot_description 파라미터 생성 (기존과 동일)
    # ===================================================================================
    urdf_file_path = os.path.join(
        get_package_share_directory('realsense2_description'),
        'urdf',
        'test_d435i_camera.urdf.xacro'
    )
    
    robot_description_content = xacro.process_file(
        urdf_file_path,
        mappings={'use_nominal_extrinsics': 'false'}
    ).toxml()

    # ===================================================================================
    # 3. robot_state_publisher 노드 실행 (기존과 동일)
    # ===================================================================================
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content
        }]
    )

    # 최종 LaunchDescription을 반환합니다.
    return LaunchDescription([
        realsense_camera,
        robot_state_publisher_node
    ])