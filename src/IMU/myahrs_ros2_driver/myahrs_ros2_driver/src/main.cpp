// Copyright 2021 CLOBOT Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <myahrs_ros2_driver/myahrs_ros2_driver.hpp>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  std::cout << "argc : " << argc << std::endl;
  std::cout << "argv[0] : " << argv[0] << std::endl;
  std::cout << "argv[1] : " << argv[1] << std::endl;
  std::cout << "argv[2] : " << argv[2] << std::endl;
  std::cout << "argv[3] : " << argv[3] << std::endl;
  std::cout << "argv[4] : " << argv[4] << std::endl;
  std::cout << "argv[5] : " << argv[5] << std::endl;

  if (argc < 6) {
    std::cout << "please check argument!!! use like..." << std::endl;
    std::cout << "arguments = ['/dev/ttyACM1','115200']" << std::endl;
    exit(0);
  }

  std::string port = argv[1];
  int baud_rate = std::stoi(argv[2]);

  rclcpp::spin(
    std::make_shared<WithRobot::MyAhrsDriverForROS>(port, baud_rate));
  rclcpp::shutdown();

  return 0;
}
