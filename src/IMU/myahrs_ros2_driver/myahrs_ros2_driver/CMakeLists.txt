cmake_minimum_required(VERSION 3.5)
project(myahrs_ros2_driver)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)

include_directories(
  include
)

set(EXECUTABLE_NAME "myahrs_ros2_driver")

set(SOURCES
  src/main.cpp
  src/myahrs_ros2_driver.cpp
)

add_executable(${EXECUTABLE_NAME} ${SOURCES})

set(dependencies
  "rclcpp"
  "std_msgs"
  "sensor_msgs"
  "geometry_msgs"
  "tf2"
  "tf2_ros"
  "tf2_msgs"
)

ament_target_dependencies(${EXECUTABLE_NAME} ${dependencies})

install(TARGETS ${EXECUTABLE_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY rviz DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
