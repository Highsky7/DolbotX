from setuptools import find_packages, setup

package_name = 'robot_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='YOLO and HSV based vision and path planning package for competition robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hsv_path_planner = robot_vision.hsv_path_planning:main',
            'hsv_path_planner_pp = robot_vision.hsv_path_planning_pp:main',
            'hsv_traffic = robot_vision.hsv_traffic:main',
            'yolo_path_planner = robot_vision.yolo_path_planning:main',
            'yolo_path_planner_pp = robot_vision.yolo_path_planning_pp:main',
            'yolo_traffic = robot_vision.yolo_traffic:main',
            'bev_utilis = robot_vision.utils.bev_utils_y_auto:main',
            'hsv_picker = robot_vision.utils.hsv_picker:main'
        ],
    },
)