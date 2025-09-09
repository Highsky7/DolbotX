from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'object_follower'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test', 'tests']),
    data_files=[
        # ament index 등록
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),

        # package.xml 설치
        ('share/' + package_name, ['package.xml']),

        # (있다면) launch/ 설치
        ('share/' + package_name + '/launch', glob('launch/*.py')),

        # (있다면) config/ 설치
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=[
        'setuptools',   # 런타임 파이썬 의존성은 보통 package.xml의 exec_depend로 관리
    ],
    extras_require={
        # 테스트 의존성은 tests_require 대신 여기로
        'test': [
            'pytest',
            'pytest-cov',
        ]
    },
    zip_safe=True,
    maintainer='junseong',
    maintainer_email='you@example.com',
    description='Follow a target in XY and output wheel velocities.',
    license='TODO: License declaration',

    entry_points={
        'console_scripts': [
            # ros2 run object_follower follower
            'follower = object_follower.follower_node:main',
        ],
    },
)
