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

    # 파이썬 노드(yolo_traffic.py, hsv_traffic.py)에서 필요한 토픽만 발행하도록
    # launch_arguments를 최적화합니다.
    realsense_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            # =======================================================================
            # [핵심 수정 1] TF 충돌 방지: 올바른 파라미터 이름으로 수정
            # robot_state_publisher가 URDF 기반의 TF를 발행하므로, realsense 노드의
            # 자체 TF 발행 기능은 비활성화하여 충돌을 막습니다.
            # 'publish_robot_description'은 유효하지 않은 파라미터이며, 'publish_tf'가 올바른 파라미터입니다.
            # =======================================================================
            'publish_tf': 'false',

            # =======================================================================
            # [필수] 코드에서 사용하는 기능들은 'true'로 유지합니다.
            # =======================================================================
            # '/camera/aligned_depth_to_color/image_raw' 토픽을 위해 필수
            'align_depth.enable': 'true',
            
            # 컬러, 뎁스, 카메라 정보 토픽의 타임스탬프 동기화를 위해 필수
            'enable_sync': 'true',

            # =======================================================================
            # [핵심 수정 2] 리소스 최적화: 불필요한 기능 비활성화
            # 현재 코드에서 사용하지 않는 포인트 클라우드와 IMU 데이터 발행을 중지하여
            # 시스템 부하를 줄입니다.
            # =======================================================================
            'pointcloud.enable': 'false', # 포인트 클라우드 비활성화
            'enable_gyro': 'false',       # IMU (Gyro) 비활성화
            'enable_accel': 'false',      # IMU (Accel) 비활성화
            
            # =======================================================================
            # [유지] 해상도 및 프레임 설정은 기존과 동일하게 유지합니다.
            # rs_launch.py에 정의된 파라미터 이름에 맞게 정확히 설정되어 있습니다.
            # =======================================================================
            'depth_module.profile': '640x480x30',
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