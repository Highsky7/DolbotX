import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robot_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', 'ament_index', 'resource_index', 'packages'),
            [os.path.join('resource', package_name)]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='YOLO and HSV based vision and path planning package for competition robot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'onnx_path_planner_pp = robot_vision.onnx_path_planning_pp:main',
            'onnx_traffic = robot_vision.onnx_traffic:main',
            'onnx_multi_traffic = robot_vision.onnx_multi_traffic:main',
            'onnx_traffic_qos = robot_vision.onnx_traffic_qos:main',
            'onnx_multi_traffic_qos = robot_vision.onnx_multi_traffic_qos:main',
            'yolotl_path_planner_pp = robot_vision.yolotl_path_planning_pp:main',
            'yolo_traffic_optimized = robot_vision.yolo_traffic_optimized:main',
            'yolo_traffic_qos_optimized = robot_vision.yolo_traffic_qos_optimized:main',
            'bev_utilis = robot_vision.utils.bev_utils_y_auto:main',
            'hsv_picker = robot_vision.utils.hsv_picker:main',
            'fire_detect = robot_vision.fire_detector:main',
            'pick_place_server = robot_vision.pick_place_server:main',
        ],
    },
)