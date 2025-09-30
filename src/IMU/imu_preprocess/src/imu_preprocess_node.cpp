#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <array>
#include <cmath>

class ImuPreprocessNode : public rclcpp::Node
{
public:
  ImuPreprocessNode() : Node("imu_preprocess_node")
  {
    /* ---------- 파라미터 ---------- */
    use_precalibrated_bias_ = declare_parameter<bool>("use_precalibrated_bias", false);
    lpf_cutoff_ = declare_parameter<double>("lpf_cutoff", 15.0);  // 1차 IIR LPF 컷오프 [Hz]
    
    // Pre-calibrated bias parameters
    if (use_precalibrated_bias_) {
      bias_acc_[0] = declare_parameter<double>("bias_acc_x", 0.0);
      bias_acc_[1] = declare_parameter<double>("bias_acc_y", 0.0);
      bias_acc_[2] = declare_parameter<double>("bias_acc_z", 0.0);
      bias_gyro_[0] = declare_parameter<double>("bias_gyro_x", 0.0);
      bias_gyro_[1] = declare_parameter<double>("bias_gyro_y", 0.0);
      bias_gyro_[2] = declare_parameter<double>("bias_gyro_z", 0.0);
      
      calibrated_ = true;  // Skip calibration phase
      
      RCLCPP_INFO(get_logger(),
        "Using pre-calibrated IMU bias:\n"
        "  acc  = [%.4f, %.4f, %.4f] m/s²\n"
        "  gyro = [%.4f, %.4f, %.4f] rad/s",
        bias_acc_[0], bias_acc_[1], bias_acc_[2],
        bias_gyro_[0], bias_gyro_[1], bias_gyro_[2]);
    } else {
      // Only declare calib_duration if we're doing runtime calibration
      calib_duration_ = declare_parameter<double>("calib_duration", 20.0);
    }

    /* ---------- 토픽 I/O ---------- */
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu/data", rclcpp::SensorDataQoS(),
      //"/ouster/imu", rclcpp::SensorDataQoS(),
      std::bind(&ImuPreprocessNode::imuCallback, this, std::placeholders::_1));

    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(
      "/imu/processed", rclcpp::SensorDataQoS());

    start_time_ = now();
  }

private:
  /* ===== IMU 콜백 ===== */
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const double dt =
      (last_time_.nanoseconds() == 0) ? 0.01
      : (now() - last_time_).seconds();
    last_time_ = now();

    /* 1) 정적 캘리브레이션 단계 ------------------------- */
    if (!calibrated_) {
      // This should only happen if use_precalibrated_bias_ is false
      accumulateBias(*msg);
      if ((now() - start_time_).seconds() >= calib_duration_) {
        finalizeBias();     // 편향 확정 + 로그 1회 출력
        calibrated_ = true;
      }
      return;               // 아직 퍼블리시하지 않음
    }

    /* 2) 편향 제거 -------------------------------------- */
    sensor_msgs::msg::Imu out = *msg;
    out.linear_acceleration.x -= bias_acc_[0];
    out.linear_acceleration.y -= bias_acc_[1];
    out.linear_acceleration.z -= bias_acc_[2];
    out.angular_velocity.x    -= bias_gyro_[0];
    out.angular_velocity.y    -= bias_gyro_[1];
    out.angular_velocity.z    -= bias_gyro_[2];

    /* 3) 1차 IIR 저역통과필터 --------------------------- */
    if (lpf_cutoff_ > 0.0) {
      const double tau   = 1.0 / (2.0 * M_PI * lpf_cutoff_);
      const double alpha = dt / (tau + dt);

      auto lpf = [alpha](double x, double &prev) {
        double y = alpha * x + (1.0 - alpha) * prev;
        prev = y;
        return y;
      };

      out.linear_acceleration.x = lpf(out.linear_acceleration.x, acc_prev_[0]);
      out.linear_acceleration.y = lpf(out.linear_acceleration.y, acc_prev_[1]);
      out.linear_acceleration.z = lpf(out.linear_acceleration.z, acc_prev_[2]);
      out.angular_velocity.x    = lpf(out.angular_velocity.x,    gyro_prev_[0]);
      out.angular_velocity.y    = lpf(out.angular_velocity.y,    gyro_prev_[1]);
      out.angular_velocity.z    = lpf(out.angular_velocity.z,    gyro_prev_[2]);
    }

    imu_pub_->publish(out);
  }

  /* ===== 편향 누적 ===== */
  void accumulateBias(const sensor_msgs::msg::Imu &m)
  {
    sum_acc_[0]  += m.linear_acceleration.x;
    sum_acc_[1]  += m.linear_acceleration.y;
    sum_acc_[2]  += m.linear_acceleration.z - 9.81;         // 중력 제거(Z)
    sum_gyro_[0] += m.angular_velocity.x;
    sum_gyro_[1] += m.angular_velocity.y;
    sum_gyro_[2] += m.angular_velocity.z;
    ++sample_cnt_;
  }

  /* ===== 편향 확정 + 단 1회 로그 ===== */
  void finalizeBias()
  {
    for (int i = 0; i < 3; ++i) {
      bias_acc_[i]  = sum_acc_[i]  / sample_cnt_;
      bias_gyro_[i] = sum_gyro_[i] / sample_cnt_;
    }
    RCLCPP_INFO(get_logger(),
      "IMU bias calibrated:\n"
      "  acc  = [%.4f, %.4f, %.4f] m/s²\n"
      "  gyro = [%.4f, %.4f, %.4f] rad/s",
      bias_acc_[0],  bias_acc_[1],  bias_acc_[2],
      bias_gyro_[0], bias_gyro_[1], bias_gyro_[2]);
  }

  /* ===== 멤버 ===== */
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr     imu_pub_;
  rclcpp::Time start_time_, last_time_;

  bool use_precalibrated_bias_;
  double calib_duration_;
  double lpf_cutoff_;
  bool calibrated_{false};

  std::array<double,3> sum_acc_{}, sum_gyro_{};
  std::array<double,3> bias_acc_{}, bias_gyro_{};
  std::array<double,3> acc_prev_{}, gyro_prev_{};
  size_t sample_cnt_{0};
};

/* ===== 메인 ===== */
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuPreprocessNode>());
  rclcpp::shutdown();
  return 0;
}
