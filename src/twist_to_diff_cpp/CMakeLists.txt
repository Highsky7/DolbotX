cmake_minimum_required(VERSION 3.8)
project(twist_to_diff_cpp)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(twist_to_diff src/twist_to_diff.cpp)
ament_target_dependencies(twist_to_diff rclcpp geometry_msgs std_msgs)

install(TARGETS twist_to_diff
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_package()

