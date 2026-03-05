#ifndef ARDUINOBOT_INTERFACE_H
#define ARDUINOBOT_INTERFACE_H

#include <rclcpp/rclcpp.hpp>
#include <hardware_interface/system_interface.hpp>
#include <libserial/SerialPort.h>
#include <rclcpp_lifecycle/state.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
#include <std_msgs/msg/string.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>

#include <vector>
#include <string>
#include <thread>
#include <memory>
#include <atomic>
#include <cstdint>


namespace arduinobot_controller
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class ArduinobotInterface : public hardware_interface::SystemInterface
{
public:
  ArduinobotInterface();
  virtual ~ArduinobotInterface();

  // Implementing rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface
  virtual CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;
  virtual CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

  // Implementing hardware_interface::SystemInterface
  virtual CallbackReturn on_init(const hardware_interface::HardwareInfo &hardware_info) override;
  virtual std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  virtual std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  virtual hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  virtual hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  LibSerial::SerialPort arduino_;
  std::string port_;
  std::vector<double> position_commands_;
  std::vector<double> prev_position_commands_;
  std::vector<double> position_states_;

  // ROS2 publishing for MLX sensor
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr mlx_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr mlx_ambient_publisher_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pickup_command_subscription_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread executor_thread_;
  bool executor_running_ = false;
  std::atomic<bool> pickup_requested_{false};
  std::atomic<uint64_t> pickup_lockout_until_ms_{0};
  static constexpr uint64_t pickup_lockout_duration_ms_ = 7000;

  // MLX sensor readings
  double mlx_x_;
  double mlx_y_;
  double mlx_z_;
  double mlx_ambient_x_;
  double mlx_ambient_y_;
  double mlx_ambient_z_;

  // Helper method
  bool parseMLXData(const std::string& line);
  bool sendSerialMessage(const std::string& msg);
  void pickupCommandCallback(const std_msgs::msg::String::SharedPtr msg);
};
}  // namespace arduinobot_controller


#endif  // ARDUINOBOT_INTERFACE_H
