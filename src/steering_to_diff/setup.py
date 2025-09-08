from setuptools import setup, find_packages

package_name = 'steering_to_diff'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/steering_to_diff.launch.py']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Steering angle to left/right wheel speeds (ROS 2 Humble).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'steering_to_diff = steering_to_diff.steering_to_diff_node:main',
        ],
    },
)

