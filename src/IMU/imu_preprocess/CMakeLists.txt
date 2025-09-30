cmake_minimum_required(VERSION 3.8)
project(imu_preprocess)

##########  의존 패키지 찾기  ##########
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(ament_index_cpp REQUIRED)

##########  헤더 포함 경로  ##########
include_directories(include)

##########  실행파일  ##########
add_executable(imu_preprocess_node src/imu_preprocess_node.cpp)
ament_target_dependencies(
  imu_preprocess_node
  rclcpp
  sensor_msgs
  nlohmann_json
  ament_index_cpp
)

##########  설치 규칙  ##########
install(TARGETS imu_preprocess_node
  DESTINATION lib/${PROJECT_NAME})

# 파라미터 YAML·Launch 파일도 함께 설치
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

##########  패키지 설정  ##########
ament_package()
