# DolbotX

## Overview

DolbotX is a versatile autonomous robot platform developed using ROS2. The robot is designed to operate in various simulated "seasonal" environments (Spring, Summer, Fall, Winter) and a robot tracking mission, each with unique perception and navigation challenges. It also features a mode for following a target robot (Unitree Go2).

The system is built to run on a distributed computing system, typically between a host laptop and an NVIDIA Jetson, interfacing with multiple cameras (including a RealSense depth camera) and Arduino-based controllers for wheel motion and LED indicators.

## Features

- **Seasonal Vision Algorithms:**
    - **Spring:** Identifies friendly ("ROKA") vs. enemy units using a model trained on military uniform data and controls an LED indicator accordingly.
    - **Summer:** Concurrently recognizes traffic lights to issue 'stop'/'go' commands and detects supply boxes to trigger actions.
    - **Fall:** Detects and follows a series of visual markers (`A`, `E`, `Heart`, etc.).
    - **Winter:** Navigates snowy environments by identifying the drivable path from a fusion of standard track and snow segmentation.
- **Object Following:** Tracks and follows a Unitree Go2 robot using a learned model of its rear profile, maintaining a set distance using depth data.
- **Drivable Area Segmentation:** Utilizes both Bird's-Eye-View (BEV) and camera-space perception to identify and navigate drivable surfaces. It supports various terrains like colored tracks, sand, gravel, and snow by generating and following paths, even when multiple drivable regions coexist in the frame (see the demonstration gallery below, including a YouTube Short for multi-area driving).
- **Multi-Camera System:** Integrates several USB cameras and an Intel RealSense depth camera for comprehensive environmental perception.
- **Hardware Integration:** Communicates with Arduino controllers for low-level wheel and LED control.

## Workspace Structure

This ROS2 workspace is organized into several key packages:

- `robot_vision/`: Contains all ROS2 nodes for computer vision tasks, including seasonal missions, drivable area segmentation, and object tracking.
- `object_follower/`: Implements the robot-following functionality, calculating wheel velocities based on the target's position.
- `steering_to_diff/`: A utility package that converts high-level steering angle commands into differential drive velocities for the left and right wheels.
- `led_serial_bridge/`: A robust serial communication bridge to send commands (`roka`, `enemy`, `none`) to an Arduino for controlling LED indicators.
- `usb_cam/`: A flexible and efficient package for capturing and publishing images from standard V4L2-compatible USB webcams.
- `mtc_interfaces/`: Defines custom ROS2 messages and services used across the workspace.
- `Arduino/`: Contains the Arduino sketches for the microcontroller-based components.

### In-Depth: The `robot_vision` Package

The `robot_vision` package is the core of the robot's autonomous capabilities. It is designed for high performance in a real-time, distributed computing environment.

#### Demonstration Gallery

<p align="center">
  <img src="competition_result.gif" alt="Competition run visualization" height="360">
</p>
<p align="center"><em>Competition run: end-to-end perception and planning performance from the `robot_vision/bezier_*_drive.py` pipeline.</em></p>

<p align="center">
  <img src="single_area.gif" alt="Single-area drive visualization" height="360">
</p>
<p align="center"><em>Single-area drive: a `robot_vision/(seasonal)_drive.py` node maintaining a smooth path through a single drivable corridor.</em></p>

<p align="center">
  <a href="https://youtube.com/shorts/69BXtWKU-2o?si=P_e7tiKzJTd6eniV" target="_blank" rel="noopener">
    <img src="https://img.youtube.com/vi/69BXtWKU-2o/hqdefault.jpg" alt="Multi-area drive YouTube Short" height="360">
  </a>
</p>
<p align="center"><em>Multi-area drive: watch the same planning stack adapt across multiple drivable regions via YouTube Shorts.</em></p>

#### Architectural Highlights:

- **ONNX Model Inference:** Leverages models converted to the ONNX (Open Neural Network Exchange) format for high-speed, hardware-accelerated inference on the NVIDIA Jetson platform.
- **Multithreaded Architecture:** ROS2 nodes use a multi-threaded design. The main thread handles ROS2 message passing, while separate worker threads perform computationally intensive tasks like model inference and image processing. This ensures that the ROS2 communication layer remains responsive.
- **Seasonal Nodes:** The package contains distinct nodes for each mission:
    - `spring_vision.py`: Identifies friend-or-foe and publishes to `/led_control`.
    - `summer_vision.py`: Concurrently manages traffic light detection (publishes to `/traffic_command`) and supply box detection (triggers a `PickPlace` service call).
    - `fall_vision.py`: Detects a variety of visual markers and publishes visualizations.
    - `unitree_tracker.py`: Detects a Unitree Go2 robot and publishes its 3D coordinates to `/target_xy`.
- **Path Planning Nodes:**
    - `bezier_*_drive.py`: A series of nodes that perform drivable area segmentation in a Bird's-Eye-View (BEV) and generate a smooth path using Bézier curves.
    - `nobev_*_drive.py`: Alternative, simplified path planning nodes that operate directly in the camera's image space without a BEV transformation, using a proportional controller.

### In-Depth: The `Arduino` Sketches

- `DolbotX_Wheel_Control/`: The main sketch for the Arduino Mega. It receives target left and right wheel velocities via serial, uses two motor encoders and a PID loop for closed-loop speed control, and drives all six motors.
- `LED_Control/`: A simple sketch for an Arduino Uno/Nano. It listens for serial commands ("roka", "enemy", "none") and controls a red and green LED to indicate status.
- `LEFT_MOTOR_FINAL/` & `RIGHT_MOTOR_FINAL/`: Development sketches used for testing the motors on each side of the robot independently.

## Getting Started

### Prerequisites

**Hardware:**
- NVIDIA Jetson Orin Nano
- Multiple Cameras (Logitech C922 Pro, Abko APC 850, Topsync TS-B7WQ3O)
- Intel RealSense D435i Depth Camera
- Arduino Mega (for wheel control) & Arduino Uno (for LED control)

**Software:**
- ROS2 Humble
- Python 3.10+
- `colcon` for building the workspace
- **Required Ubuntu Packages:**
  ```bash
  sudo apt-get update
  sudo apt-get install ros-humble-realsense2-camera ros-humble-tf2-geometry-msgs python3-serial
  ```

### Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```
This will install `ultralytics` with specific versions of `numpy` and `pyrealsense2`.

### Installation

1.  Clone this repository.
2.  Source your ROS2 installation:
    ```bash
    source /opt/ros/humble/setup.bash
    ```
3.  Build the workspace. For active development on Python nodes, use the `--symlink-install` flag to avoid rebuilding after every change.
    ```bash
    # Standard build
    colcon build

    # Development build
    colcon build --symlink-install
    ```

### Data Collection Workflow

DolbotX records synchronized RGB, depth, and pose streams. Use the following three-terminal routine to create a dataset ready for post-processing.

1. **Terminal 1 – Launch the recorder node**
   ```bash
   ros2 run robot_vision unified_recorder.py
   ```
   Captures metadata and timing signals so all topics remain aligned.

2. **Terminal 2 – Start ROS bag recording**
   ```bash
   ros2 bag record \
   /camera3/image_raw/compressed \
   /camera/color/camera_info \
   /camera/color/image_raw/compressed \
   /camera/aligned_depth_to_color/image_raw \
   /camera1/image_raw/compressed \
   /camera2/image_raw/compressed \
   /tf \
   /tf_static \
   -o my_robot_run_$(date +%Y-%m-%d-%H-%M-%S) \
   -s mcap \
   --compression-mode file \
   --compression-format zstd
   ```
   MCAP with Zstd compression balances storage requirements and fast playback.

3. **Terminal 3 – Run the H.264 converter**
   ```bash
   cd src/robot_vision/robot_vision/utils
   python h264_converter.py
   ```
   Transcodes the recorded image topics to an efficient H.264 format for labelling and sharing.

## Usage

The robot's operation is modular and requires running several nodes in separate terminals.

### 1. Launch Cameras

**On the Jetson device:**

- **Trigger udev to recognize devices:**
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

- **Steering to Differential Drive Converter:**
  ```bash
  ros2 launch steering_to_diff steering_to_diff.launch.py
  ```
- **Wheel Serial Bridge (for communication with Arduino Mega):**
  ```bash
  ros2 launch wheel_serial_bridge bridge.launch.py
  ```
- **LED Control Bridge (for communication with Arduino Uno):**
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
- **Vision Node (handles both traffic lights and supply boxes):**
  ```bash
  ros2 run robot_vision summer_vision
  ```
- **Drive Node (navigates standard track, sand, and stone):**
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
- **Drive Node (navigates standard track and snow):**
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
