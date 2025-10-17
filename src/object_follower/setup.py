from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'object_follower'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test', 'tests']),
    data_files=[
        # Register with the ament index
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),

        # Install package.xml
        ('share/' + package_name, ['package.xml']),

        # Install launch files if present
        ('share/' + package_name + '/launch', glob('launch/*.py')),

        # Install config files if present
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=[
        'setuptools',   # Runtime Python deps are usually tracked in package.xml exec_depend
    ],
    extras_require={
        # Place test dependencies here instead of tests_require
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
