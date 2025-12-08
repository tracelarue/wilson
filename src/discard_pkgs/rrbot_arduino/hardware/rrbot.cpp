// Copyright 2020 ros2_control Development Team
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

#include "ros2_control_demo_example_1/rrbot.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace ros2_control_demo_example_1
{
hardware_interface::CallbackReturn RRBotSystemPositionOnlyHardware::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }
  logger_ = std::make_shared<rclcpp::Logger>(
    rclcpp::get_logger("controller_manager.resource_manager.hardware_component.system.RRBot"));
  clock_ = std::make_shared<rclcpp::Clock>(rclcpp::Clock());

  // BEGIN: This part here is for exemplary purposes - Please do not copy to your production code
  hw_start_sec_ = stod(info_.hardware_parameters["example_param_hw_start_duration_sec"]);
  hw_stop_sec_ = stod(info_.hardware_parameters["example_param_hw_stop_duration_sec"]);
  hw_slowdown_ = stod(info_.hardware_parameters["example_param_hw_slowdown"]);
  // END: This part here is for exemplary purposes - Please do not copy to your production code

  // Configure Arduino communication parameters - same pattern as diffdrive_arduino
  cfg_.device = info_.hardware_parameters["device"];
  cfg_.baud_rate = std::stoi(info_.hardware_parameters["baud_rate"]);
  cfg_.timeout_ms = std::stoi(info_.hardware_parameters["timeout_ms"]);
  hw_states_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
  hw_commands_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());

  for (const hardware_interface::ComponentInfo & joint : info_.joints)
  {
    // RRBotSystemPositionOnly has exactly one state and command interface on each joint
    if (joint.command_interfaces.size() != 1)
    {
      RCLCPP_FATAL(
        get_logger(), "Joint '%s' has %zu command interfaces found. 1 expected.",
        joint.name.c_str(), joint.command_interfaces.size());
      return hardware_interface::CallbackReturn::ERROR;
    }

    if (joint.command_interfaces[0].name != hardware_interface::HW_IF_POSITION)
    {
      RCLCPP_FATAL(
        get_logger(), "Joint '%s' have %s command interfaces found. '%s' expected.",
        joint.name.c_str(), joint.command_interfaces[0].name.c_str(),
        hardware_interface::HW_IF_POSITION);
      return hardware_interface::CallbackReturn::ERROR;
    }

    if (joint.state_interfaces.size() != 1)
    {
      RCLCPP_FATAL(
        get_logger(), "Joint '%s' has %zu state interface. 1 expected.", joint.name.c_str(),
        joint.state_interfaces.size());
      return hardware_interface::CallbackReturn::ERROR;
    }

    if (joint.state_interfaces[0].name != hardware_interface::HW_IF_POSITION)
    {
      RCLCPP_FATAL(
        get_logger(), "Joint '%s' have %s state interface. '%s' expected.", joint.name.c_str(),
        joint.state_interfaces[0].name.c_str(), hardware_interface::HW_IF_POSITION);
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn RRBotSystemPositionOnlyHardware::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "Configuring Arduino communication...");
  
  // Disconnect if already connected
  if (comms_.connected())
  {
    comms_.disconnect();
  }
  
  // Connect to Arduino
  try 
  {
    comms_.connect(cfg_.device, cfg_.baud_rate, cfg_.timeout_ms);
    RCLCPP_INFO(get_logger(), "Successfully connected to Arduino on %s", cfg_.device.c_str());
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(get_logger(), "Failed to connect to Arduino: %s", e.what());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // reset values always when configuring hardware
  for (uint i = 0; i < hw_states_.size(); i++)
  {
    hw_states_[i] = 0;
    hw_commands_[i] = 0;
  }
  RCLCPP_INFO(get_logger(), "Successfully configured!");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn RRBotSystemPositionOnlyHardware::on_cleanup(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "Cleaning up Arduino connection...");
  
  if (comms_.connected())
  {
    comms_.disconnect();
    RCLCPP_INFO(get_logger(), "Disconnected from Arduino");
  }
  
  RCLCPP_INFO(get_logger(), "Successfully cleaned up!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
RRBotSystemPositionOnlyHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (uint i = 0; i < info_.joints.size(); i++)
  {
    state_interfaces.emplace_back(
      hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_states_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
RRBotSystemPositionOnlyHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (uint i = 0; i < info_.joints.size(); i++)
  {
    command_interfaces.emplace_back(
      hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]));
  }

  return command_interfaces;
}

hardware_interface::CallbackReturn RRBotSystemPositionOnlyHardware::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "Activating ...please wait...");
  
  // Verify Arduino connection
  if (!comms_.connected())
  {
    RCLCPP_ERROR(get_logger(), "Arduino not connected! Cannot activate.");
    return hardware_interface::CallbackReturn::ERROR;
  }

  // command and state should be equal when starting
  for (uint i = 0; i < hw_states_.size(); i++)
  {
    hw_commands_[i] = hw_states_[i];
  }

  RCLCPP_INFO(get_logger(), "Successfully activated!");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn RRBotSystemPositionOnlyHardware::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "Deactivating ...");
  
  // Optionally send stop commands to Arduino before deactivating
  if (comms_.connected())
  {
    // Send servos to neutral position (90 degrees)
    try 
    {
      for (size_t i = 0; i < hw_commands_.size(); i++)
      {
        // Set to neutral position (90 degrees = 0 radians)
        comms_.set_servo_position(static_cast<int>(i), 90);
      }
    }
    catch (const std::exception& e)
    {
      RCLCPP_WARN(get_logger(), "Failed to send servo stop commands: %s", e.what());
    }
  }

  RCLCPP_INFO(get_logger(), "Successfully deactivated!");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type RRBotSystemPositionOnlyHardware::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (!comms_.connected())
  {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Arduino not connected!");
    return hardware_interface::return_type::ERROR;
  }

  try
  {
    // Read servo positions from Arduino using ros_arduino_bridge protocol
    std::vector<int> servo_positions = comms_.read_servo_positions();
    
    for (size_t i = 0; i < hw_states_.size() && i < info_.joints.size() && i < servo_positions.size(); i++)
    {
      // Convert servo angle (0-180 degrees) to radians
      hw_states_[i] = (servo_positions[i] * M_PI) / 180.0;
    }

    // Debug output
    std::stringstream ss;
    ss << "Joint positions:";
    for (uint i = 0; i < hw_states_.size(); i++)
    {
      ss << std::fixed << std::setprecision(2) << " " << info_.joints[i].name 
         << ":" << hw_states_[i];
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "%s", ss.str().c_str());
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Read failed: %s", e.what());
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type RRBotSystemPositionOnlyHardware::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (!comms_.connected())
  {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Arduino not connected!");
    return hardware_interface::return_type::ERROR;
  }

  try
  {
    // Send servo position commands to Arduino using ros_arduino_bridge protocol
    // Convert radians to degrees (0-180) and send to each servo
    for (size_t i = 0; i < hw_commands_.size() && i < info_.joints.size(); i++)
    {
      // Convert from radians to degrees (0-180)
      double angle_rad = hw_commands_[i];
      // Clamp to servo range and convert to degrees
      double angle_deg = std::max(0.0, std::min(180.0, ((angle_rad + M_PI/2) * 180.0) / M_PI));
      int servo_angle = static_cast<int>(angle_deg);
      
      // Send command to specific servo using ros_arduino_bridge 's' command
      comms_.set_servo_position(static_cast<int>(i), servo_angle);
    }
    
    // Debug output
    std::stringstream ss;
    ss << "Servo commands:";
    for (uint i = 0; i < hw_commands_.size(); i++)
    {
      double angle_deg = std::max(0.0, std::min(180.0, ((hw_commands_[i] + M_PI/2) * 180.0) / M_PI));
      ss << " " << info_.joints[i].name << ":" << std::fixed << std::setprecision(0) << angle_deg << "°";
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "%s", ss.str().c_str());
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Write failed: %s", e.what());
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

}  // namespace ros2_control_demo_example_1

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  ros2_control_demo_example_1::RRBotSystemPositionOnlyHardware, hardware_interface::SystemInterface)
