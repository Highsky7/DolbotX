# DolbotX

## Overview

DolbotX is a versatile autonomous robot platform developed using ROS2. The robot is designed to operate in various simulated "seasonal" environments (Spring, Summer, Fall, Winter) and robot tracking mission(Unitree GO2 tracking), each with unique perception and navigation challenges. It also features a mode for following a target robot.

The system is built to run on a distributed computing system between local host laptop and jetson orin nano, interfacing with multiple cameras (including a RealSense depth camera) and Arduino-based controllers for wheel motion and LED indicators.

## Features

- **Seasonal Vision Algorithms:**
    - **Spring:** Identifies friendly vs. enemy units using a model trained on synthetic data of North and South Korean military uniforms.
    - **Summer:** Recognizes and responds to traffic lights (stopping for red, proceeding for green).
    - **Fall:** Detects and follows visual markers.
    - **Winter:** Navigates snowy environments by identifying the drivable path.
- **Object Following:** Tracks and follows a Unitree robot using a learned model of its rear profile, maintaining a set distance using depth data.
- **Drivable Area Segmentation:** Utilizes BEV (Bird's-Eye View) perception to identify and navigate drivable surfaces (e.g., mint-colored tracks, sand, gravel) by generating and following Bezier curve paths.
    - **Spring, Fall** Use mint-colored tracks for drivable surfaces
    - **Summer** Use mint-colored drivable tracks, sand, stone for drivable surfaces
    -   **Winter** Use mint-colored drivable tracks, snow for drivable surfaces
- **Multi-Camera System:** Integrates several USB cameras and a RealSense depth camera for comprehensive environmental perception.
- **Hardware Integration:** Communicates with Arduino controllers for low-level wheel and LED control.

## Workspace Structure

This ROS2 workspace is organized into the following key packages:

- `robot_vision/`: Contains all the ROS2 nodes related to computer vision tasks for each seasonal mode and for drivable area segmentation.
- `object_follower/`: Implements the robot-following functionality.
- `steering_to_diff/`: A utility package to convert steering commands into differential drive commands for the wheels.
- `led_serial_bridge/`: A bridge to communicate with an Arduino for controlling LEDs.
- `usb_cam/`: Package for managing standard USB webcams.
- `mtc_interfaces/`: Defines custom ROS2 messages and services.
- `Arduino/`: Contains the Arduino sketches for the microcontroller-based components.

### In-Depth: The `robot_vision` Package

The `robot_vision` package is the core of the robot's autonomous capabilities. It is designed for high performance in a real-time, distributed computing environment.

#### Architectural Highlights:

- **ONNX Model Inference:** To achieve high-speed and efficient inference, the package leverages models converted to the ONNX (Open Neural Network Exchange) format. This allows for hardware-accelerated computation on the NVIDIA Jetson platform, significantly reducing processing latency compared to running models in their native frameworks.

- **Multithreaded Architecture:** The ROS2 nodes within this package are designed with a multithreaded architecture. The main thread handles the ROS2 message passing (subscriptions and publications), while a separate worker thread is dedicated to the computationally intensive tasks of model inference and image processing. This separation ensures that the ROS2 communication remains responsive and is not blocked by heavy computation, which is critical for real-time performance.

- **Seasonal Nodes:** The package contains distinct nodes for each seasonal mission, allowing for modularity and clarity:
    - `spring_vision.py`: Handles the friend-or-foe identification.
    - `summer_vision.py`: Manages traffic light detection.
    - `fall_vision.py`: Focuses on visual marker detection.
    - `*_drive.py`: These nodes handle the drivable area segmentation and path planning for each season, using the output from the vision nodes.

## Getting Started

### Prerequisites

**Hardware:**
- NVIDIA Jetson Orin Nano
- Multiple Cameras (Logitech C922 pro, Abko APC 850, Topsync TS-B7WQ3O)
- Intel RealSense d435i Depth Camera
- Arduino Mega for drive and Arduino Uno for led control

**Software:**
- ROS2 (Humble recommended)
- Python 3.10+
- `colcon` for building the workspace
- 'colcon build --symlink-install ' for better usage with modifying python files easily

### Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```
This will install:
- `ultralytics==8.3.172`
- `numpy==1.26.4`

### Installation

1.  Clone this repository.
2.  Source your ROS2 installation.
    ```bash
    source /opt/ros/humble/setup.bash
    ```
3.  Build the workspace using `colcon`.
    ```bash
    colcon build
    ```
3-1. Build the workspace with active development
    ```bash
    colcon build --symlink-install
    ```

## Usage

The robot's operation is modular and requires running several nodes in separate terminals.

### 1. Launch Cameras

**On the Jetson device:**

- **Trigger udev:**
  ```bash
  sudo udevadm trigger
  ```
- **Driving Camera (Logitech):**
  ```bash
  ros2 run usb_cam usb_cam_node_exe --ros-args --remap __ns:=/camera3 --params-file /home/nvidia/DolbotX/src/usb_cam/config/params_3.yaml
  ```
- **Left Camera (Abko):**
  ```bash
  ros2 run usb_cam usb_cam_node_exe --ros-args --remap __ns:=/camera1 --params-file /home/nvidia/DolbotX/src/usb_cam/config/params_1.yaml
  ```
- **Right Camera (Topsync):**
  ```bash
  ros2 run usb_cam usb_cam_node_exe --ros-args --remap __ns:=/camera2 --params-file /home/nvidia/DolbotX/src/usb_cam/config/params_2.yaml
  ```
- **RealSense Depth Camera:**
  ```bash
  ros2 launch realsense2_camera rs_launch.py
  ```

### 2. Launch Robot Base and Controllers

- **Steering to Differential Drive:**
  ```bash
  ros2 launch steering_to_diff stering_to_diff.launch.py
  ```
- **Wheel Serial Bridge (on Jetson):**
  ```bash
  ros2 launch wheel_serial_bridge bridge.launch.py
  ```
- **LED Control (on Jetson):**
  ```bash
  ros2 run led_serial_bridge led_serial_bridge
  ```

### 3. Launch a Mission

Choose one of the following missions.

#### Spring Mission
- **Vision Node:**
  ```bash
  ros2 run robot_vision spring_vision
  ```
- **Drive Node:**
  ```bash
  ros2 run robot_vision bezier_springfall_drive
  ```

#### Summer Mission
- **Vision Node:**
  ```bash
  ros2 run robot_vision summer_vision
  ```
- **Drive Node:**
  ```bash
  ros2 run robot_vision bezier_summer_drive
  ```

#### Fall Mission
- **Vision Node:**
  ```bash
  ros2 run robot_vision fall_vision
  ```
- **Drive Node:**
  ```bash
  ros2 run robot_vision bezier_springfall_drive
  ```

#### Winter Mission
- **Drive Node:**
  ```bash
  ros2 run robot_vision bezier_winter_drive
  ```

#### Object Follower Mission
- **Tracker Node:**
  ```bash
  ros2 run robot_vision unitree_tracker
  ```
- **Follower Launch:**
  ```bash
  ros2 launch object_follower object_follower.launch.py
  ```
