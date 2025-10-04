#include "arduinobot_controller/arduinobot_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <thread>
#include <chrono>
#include <sstream>


namespace arduinobot_controller
{

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
  std::vector<double> initial_positions = {0.0, 0.2495, -2.3562, -0.6098, 0.0};

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
  
  // Debug: Log the current state being reported
  static int log_counter = 0;
  if (++log_counter % 100 == 0) { // Log every 100 reads to avoid spam
    RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"), 
                "Current state: j1=%.3f, j2=%.3f, j3=%.3f, j4=%.3f, gripper=%.3f",
                position_states_[0], position_states_[1], position_states_[2], 
                position_states_[3], position_states_[4]);
  }
  
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type ArduinobotInterface::write(const rclcpp::Time &time,
                                                           const rclcpp::Duration &period)
{
  if (position_commands_ == prev_position_commands_)
  {
    // Nothing changed, do not send any command
    return hardware_interface::return_type::OK;
  }

  // Debug: Log the command positions
  RCLCPP_INFO(rclcpp::get_logger("ArduinobotInterface"), 
              "Received command: j1=%.3f, j2=%.3f, j3=%.3f, j4=%.3f, gripper=%.3f",
              position_commands_[0], position_commands_[1], position_commands_[2], 
              position_commands_[3], position_commands_[4]);

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

  try
  {
    RCLCPP_INFO_STREAM(rclcpp::get_logger("ArduinobotInterface"), "Sending new command " << msg);
    arduino_.Write(msg);
  }
  catch (...)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("ArduinobotInterface"),
                        "Something went wrong while sending the message "
                            << msg << " to the port " << port_);
    return hardware_interface::return_type::ERROR;
  }

  prev_position_commands_ = position_commands_;

  return hardware_interface::return_type::OK;
}
}  // namespace arduinobot_controller

PLUGINLIB_EXPORT_CLASS(arduinobot_controller::ArduinobotInterface, hardware_interface::SystemInterface)
