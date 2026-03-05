#include "arduinobot_controller/arduinobot_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <thread>
#include <chrono>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <functional>
#include <cmath>


namespace arduinobot_controller
{

namespace
{

uint64_t monotonicMs()
{
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::string lowercaseCopy(std::string input)
{
  std::transform(input.begin(), input.end(), input.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return input;
}

}  // namespace

std::string compensateZeros(const int value)
{
  std::string compensate_zeros = "";
  if(value < 10){
    compensate_zeros = "00";
  } else if(value < 100){
    compensate_zeros = "0";
  } else {
    compensate_zeros = "";
  }
  return compensate_zeros;
}
  
ArduinobotInterface::ArduinobotInterface()
{
}


ArduinobotInterface::~ArduinobotInterface()
{
  // Stop executor thread
  executor_running_ = false;
  if (executor_thread_.joinable())
  {
    executor_thread_.join();
  }

  if (arduino_.IsOpen())
  {
    try
    {
      arduino_.Close();
    }
    catch (...)
    {
      RCLCPP_FATAL_STREAM(rclcpp::get_logger("ArduinobotInterface"),
                          "Something went wrong while closing connection with port " << port_);
    }
  }
}


CallbackReturn ArduinobotInterface::on_init(const hardware_interface::HardwareInfo &hardware_info)
{
  CallbackReturn result = hardware_interface::SystemInterface::on_init(hardware_info);
  if (result != CallbackReturn::SUCCESS)
  {
    return result;
  }

  try
  {
    port_ = info_.hardware_parameters.at("port");
  }
  catch (const std::out_of_range &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("ArduinobotInterface"), "No Serial Port provided! Aborting");
    return CallbackReturn::FAILURE;
  }

  // Manual initial positions - using "idle" state from SRDF
  // Order: joint_1, joint_2, joint_3, joint_4, gripper_left_finger_joint
  std::vector<double> initial_positions = {0.0, 0.2495, -2.1817, -0.9, 0.0};

  position_commands_.resize(info_.joints.size());
  position_states_.resize(info_.joints.size());
  prev_position_commands_.resize(info_.joints.size());

  // Initialize with manual values
  for (size_t i = 0; i < info_.joints.size() && i < initial_positions.size(); i++)
  {
    position_commands_[i] = initial_positions[i];
    position_states_[i] = initial_positions[i];
    prev_position_commands_[i] = initial_positions[i];

    RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"),
               "Joint %s initial position: %f rad",
               info_.joints[i].name.c_str(), initial_positions[i]);
  }

  // Initialize MLX sensor values
  mlx_x_ = 0.0;
  mlx_y_ = 0.0;
  mlx_z_ = 0.0;
  mlx_ambient_x_ = 0.0;
  mlx_ambient_y_ = 0.0;
  mlx_ambient_z_ = 0.0;

  // Create ROS2 node and publisher for MLX sensor
  try
  {
    node_ = rclcpp::Node::make_shared("arduinobot_mlx_publisher");
    mlx_publisher_ = node_->create_publisher<sensor_msgs::msg::MagneticField>(
        "/mlx", 10);
    mlx_ambient_publisher_ = node_->create_publisher<sensor_msgs::msg::MagneticField>(
        "/mlx_ambient", 10);
    pickup_command_subscription_ = node_->create_subscription<std_msgs::msg::String>(
        "/manual_arm_control/command",
        10,
        std::bind(&ArduinobotInterface::pickupCommandCallback, this, std::placeholders::_1));

    // Create executor and start spinning thread
    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);
    executor_running_ = true;

    executor_thread_ = std::thread([this]() {
      while (executor_running_)
      {
        executor_->spin_some(std::chrono::milliseconds(10));
      }
    });

    RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"),
                "MLX publishers ready on /mlx + /mlx_ambient; pickup command subscription on /manual_arm_control/command");
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("ArduinobotInterface"),
                        "Failed to initialize MLX publisher: " << e.what());
    return CallbackReturn::FAILURE;
  }

  return CallbackReturn::SUCCESS;
}


std::vector<hardware_interface::StateInterface> ArduinobotInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  // Provide only a position Interafce
  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_states_[i]));
  }

  return state_interfaces;
}


std::vector<hardware_interface::CommandInterface> ArduinobotInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  // Provide only a position Interafce
  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_commands_[i]));
  }

  return command_interfaces;
}


CallbackReturn ArduinobotInterface::on_activate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"), "Starting robot hardware ...");

  try
  {
    arduino_.Open(port_);
    arduino_.SetBaudRate(LibSerial::BaudRate::BAUD_115200);
  }
  catch (...)
  {
    RCLCPP_FATAL_STREAM(rclcpp::get_logger("ArduinobotInterface"),
                        "Something went wrong while interacting with port " << port_);
    return CallbackReturn::FAILURE;
  }

  // Use the manual initial positions set in on_init() instead of reading from Arduino
  RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"), 
              "Using manual initial positions - not reading current servo positions from Arduino");

  // Initialize previous commands to match current commands
  prev_position_commands_ = position_commands_;

  RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"),
              "Hardware started, ready to take commands");
  return CallbackReturn::SUCCESS;
}


CallbackReturn ArduinobotInterface::on_deactivate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"), "Stopping robot hardware ...");

  // Stop executor
  executor_running_ = false;
  if (executor_)
  {
    executor_->cancel();
  }

  if (executor_thread_.joinable())
  {
    executor_thread_.join();
  }

  if (arduino_.IsOpen())
  {
    try
    {
      arduino_.Close();
    }
    catch (...)
    {
      RCLCPP_FATAL_STREAM(rclcpp::get_logger("ArduinobotInterface"),
                          "Something went wrong while closing connection with port " << port_);
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"), "Hardware stopped");
  return CallbackReturn::SUCCESS;
}


hardware_interface::return_type ArduinobotInterface::read(const rclcpp::Time &time,
                                                          const rclcpp::Duration &period)
{
  // Open Loop Control - assuming the robot is always where we command to be
  position_states_ = position_commands_;

  // Read available data from serial port (non-blocking)
  if (arduino_.IsOpen() && arduino_.IsDataAvailable())
  {
    try
    {
      std::string line;
      arduino_.ReadLine(line, '\n', 10); // 10ms timeout

      // Try to parse as MLX sensor data
      if (parseMLXData(line))
      {
        if (mlx_publisher_)
        {
          auto msg = sensor_msgs::msg::MagneticField();
          msg.header.stamp = time;
          msg.header.frame_id = "end_effector_frame";
          msg.magnetic_field.x = mlx_x_;
          msg.magnetic_field.y = mlx_y_;
          msg.magnetic_field.z = mlx_z_;
          mlx_publisher_->publish(msg);
        }

        if (mlx_ambient_publisher_)
        {
          auto ambient_msg = sensor_msgs::msg::MagneticField();
          ambient_msg.header.stamp = time;
          ambient_msg.header.frame_id = "end_effector_frame";
          ambient_msg.magnetic_field.x = mlx_ambient_x_;
          ambient_msg.magnetic_field.y = mlx_ambient_y_;
          ambient_msg.magnetic_field.z = mlx_ambient_z_;
          mlx_ambient_publisher_->publish(ambient_msg);
        }
      }
    }
    catch (const std::exception& e)
    {
      // Non-critical: just skip this read if parsing fails
      // Don't spam logs - sensor reads happen at 100Hz control rate
    }
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type ArduinobotInterface::write(const rclcpp::Time &time,
                                                           const rclcpp::Duration &period)
{
  // Dispatch one-shot pickup command before joint streaming.
  if (pickup_requested_.exchange(false))
  {
    pickup_lockout_until_ms_.store(monotonicMs() + pickup_lockout_duration_ms_);
    if (!sendSerialMessage("p001,"))
    {
      return hardware_interface::return_type::ERROR;
    }
    return hardware_interface::return_type::OK;
  }

  // Hold off direct joint commands while Arduino runs pickup sequence.
  if (monotonicMs() < pickup_lockout_until_ms_.load())
  {
    return hardware_interface::return_type::OK;
  }

  if (position_commands_ == prev_position_commands_)
  {
    // Nothing changed, do not send any command
    return hardware_interface::return_type::OK;
  }

  // Debug: Log the command positions
  //RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"), 
  //            "Received command: j1=%.3f, j2=%.3f, j3=%.3f, j4=%.3f, gripper=%.3f",
  //            position_commands_[0], position_commands_[1], position_commands_[2], 
  //            position_commands_[3], position_commands_[4]);

  std::string msg;
  // New formula: degrees = (radians + 3π/4) × (180/π)
  // Maps [-3π/4, 3π/4] rad to [0°, 270°]
  int base = static_cast<int>((position_commands_.at(0) + (0.75 * M_PI)) * 180.0 / M_PI);
  msg.append("b");
  msg.append(compensateZeros(base));
  msg.append(std::to_string(base));
  msg.append(",");
  int shoulder = static_cast<int>((position_commands_.at(1) + (0.75 * M_PI)) * 180.0 / M_PI);
  msg.append("s");
  msg.append(compensateZeros(shoulder));
  msg.append(std::to_string(shoulder));
  msg.append(",");
  int elbow = static_cast<int>((position_commands_.at(2) + (0.75 * M_PI)) * 180.0 / M_PI);
  msg.append("e");
  msg.append(compensateZeros(elbow));
  msg.append(std::to_string(elbow));
  msg.append(",");
  int wrist = static_cast<int>((position_commands_.at(3) + (0.75 * M_PI)) * 180.0 / M_PI);
  msg.append("w");
  msg.append(compensateZeros(wrist));
  msg.append(std::to_string(wrist));
  msg.append(",");
  // Gripper: maps [0, 0.05] rad to [0°, 220°]
  int gripper = static_cast<int>(position_commands_.at(4) * 4400.0); // convert 0-0.05 m to 0-220 deg
  msg.append("g");
  msg.append(compensateZeros(gripper));
  msg.append(std::to_string(gripper));
  msg.append(",");

  if (!sendSerialMessage(msg))
  {
    return hardware_interface::return_type::ERROR;
  }

  prev_position_commands_ = position_commands_;

  return hardware_interface::return_type::OK;
}

bool ArduinobotInterface::parseMLXData(const std::string& line)
{
  // MLX sensor data format: "x,y,z,ax,ay,az\n" (6 comma-separated values)
  // Position response format: "base,shoulder,elbow,wrist,gripper\n" (5 values)

  std::istringstream iss(line);
  std::string token;
  std::vector<double> values;

  // Parse comma-separated values
  while (std::getline(iss, token, ','))
  {
    try
    {
      values.push_back(std::stod(token));
    }
    catch (...)
    {
      return false; // Invalid numeric value
    }
  }

  // MLX data has exactly 6 values (primary + ambient)
  if (values.size() == 6)
  {
    mlx_x_ = values[0];
    mlx_y_ = values[1];
    mlx_z_ = values[2];
    mlx_ambient_x_ = values[3];
    mlx_ambient_y_ = values[4];
    mlx_ambient_z_ = values[5];

    // DEBUG: Log parsed values
    // static int log_count = 0;
    // if (++log_count % 10 == 0)  // Log every 10 readings to avoid spam
    // {
    //   RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"),
    //               "MLX parsed: x=%.2f, y=%.2f, z=%.2f, ax=%.2f, ay=%.2f, az=%.2f",
    //               mlx_x_, mlx_y_, mlx_z_, mlx_ambient_x_, mlx_ambient_y_, mlx_ambient_z_);
    // }

    return true;
  }

  return false; // Not MLX data (could be position response with 5 values)
}

bool ArduinobotInterface::sendSerialMessage(const std::string& msg)
{
  if (!arduino_.IsOpen())
  {
    RCLCPP_ERROR(rclcpp::get_logger("ArduinobotInterface"),
                 "Serial port is not open; cannot send command");
    return false;
  }

  try
  {
    RCLCPP_INFO_STREAM(rclcpp::get_logger("ArduinobotInterface"), "Sending command " << msg);
    arduino_.Write(msg);
    return true;
  }
  catch (...)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("ArduinobotInterface"),
                        "Something went wrong while sending the message "
                            << msg << " to the port " << port_);
    return false;
  }
}

void ArduinobotInterface::pickupCommandCallback(const std_msgs::msg::String::SharedPtr msg)
{
  if (!msg)
  {
    return;
  }

  const std::string command = lowercaseCopy(msg->data);
  if (command == "pickup")
  {
    pickup_requested_.store(true);
    pickup_lockout_until_ms_.store(monotonicMs() + pickup_lockout_duration_ms_);
    RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"),
                "Received pickup request; dispatching p001 to Arduino");
  }
}
}  // namespace arduinobot_controller

PLUGINLIB_EXPORT_CLASS(arduinobot_controller::ArduinobotInterface, hardware_interface::SystemInterface)
