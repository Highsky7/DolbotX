cmake_minimum_required(VERSION 3.5)
project(wheelchair_slam_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  config
  urdf
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/session_recorder.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
