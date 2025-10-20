cmake_minimum_required(VERSION 3.8)
project(pick_place_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate action messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PickObject.action"
  DEPENDENCIES geometry_msgs
)

# Build the action server executable
add_executable(pick_object_action_server src/pick_object_action_server.cpp)

# Specify dependencies for the executable
ament_target_dependencies(pick_object_action_server
  rclcpp
  rclcpp_action
  moveit_task_constructor_core
  moveit_ros_planning_interface
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  shape_msgs
)

# Link the generated action interface
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(pick_object_action_server "${cpp_typesupport_target}")

target_include_directories(pick_object_action_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(pick_object_action_server PUBLIC c_std_99 cxx_std_17)

# Install targets
install(TARGETS pick_object_action_server
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
