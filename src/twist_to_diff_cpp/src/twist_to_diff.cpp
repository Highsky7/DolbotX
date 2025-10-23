#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

class TwistToDiff : public rclcpp::Node {
public:
  TwistToDiff() : rclcpp::Node("twist_to_diff_cpp") {
    wheel_base_ = this->declare_parameter<double>("wheel_base", 0.5); // m

    left_pub_  = this->create_publisher<std_msgs::msg::Float32>("cmd_vel_left", 10);
    right_pub_ = this->create_publisher<std_msgs::msg::Float32>("cmd_vel_right", 10);

    sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", rclcpp::QoS(10),
      std::bind(&TwistToDiff::onTwist, this, std::placeholders::_1));

    param_cb_handle_ = this->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto & p : params) {
          if (p.get_name() == "wheel_base") {
            wheel_base_ = p.as_double();
          }
        }
        return result;
      });
  }

private:
  void onTwist(const geometry_msgs::msg::Twist::SharedPtr msg) {
    const double v = msg->linear.x;
    const double w = msg->angular.z;
    const double v_left  = v - (w * wheel_base_ / 2.0);
    const double v_right = v + (w * wheel_base_ / 2.0);

    std_msgs::msg::Float32 l; l.data = static_cast<float>(v_left);
    std_msgs::msg::Float32 r; r.data = static_cast<float>(v_right);
    left_pub_->publish(l);
    right_pub_->publish(r);
  }

  double wheel_base_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr left_pub_, right_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TwistToDiff>());
  rclcpp::shutdown();
  return 0;
}

