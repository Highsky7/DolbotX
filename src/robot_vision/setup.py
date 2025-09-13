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
            'springfall_drive = robot_vision.springfall_drive:main',
            'nobev_springfall_drive = robot_vision.nobev_springfall_drive:main',
            'bezier_springfall_drive = robot_vision.bezier_springfall_drive:main',
            'summer_drive = robot_vision.summer_drive:main',
            'nobev_summer_drive = robot_vision.nobev_summer_drive:main',
            'bezier_summer_drive = robot_vision.bezier_summer_drive:main',
            'winter_drive = robot_vision.winter_drive:main',
            'nobev_winter_drive = robot_vision.nobev_winter_drive:main',
            'bezier_winter_drive = robot_vision.bezier_winter_drive:main',
            'unitree_tracker = robot_vision.unitree_tracker:main',
            'summer_vision = robot_vision.summer_vision:main',
            'summer_vision_manual = robot_vision.summer_vision_manual:main',
            'springfall_vision = robot_vision.springfall_vision:main',
            'spring_vision = robot_vision.spring_vision:main',
            'fall_vision = robot_vision.fall_vision:main',
            'vision_nofilter = robot_vision.vision_nofilter:main',
            'bev_utilis_auto = robot_vision.utils.bev_utils_y_auto:main',
            'bev_utilis = robot_vision.utils.bev_utils:main',
            'hsv_picker = robot_vision.utils.hsv_picker:main',
            'unified_recorder = robot_vision.utils.unified_recorder:main',
            'bev_recorder = robot_vision.utils.bev_recorder:main',
            'realsense_recorder = robot_vision.utils.realsense_recorder:main',
            'camera1_recorder = robot_vision.utils.camera1_recorder:main',
            'camera2_recorder = robot_vision.utils.camera2_recorder:main',
            'fire_detect = robot_vision.fire_detector:main',
            'pick_place_server = robot_vision.pick_place_server:main',
        ],
    },
)