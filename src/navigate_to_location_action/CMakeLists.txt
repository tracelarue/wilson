cmake_minimum_required(VERSION 3.8)
project(navigate_to_location_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate action messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/NavigateToLocation.action"
  DEPENDENCIES geometry_msgs std_msgs action_msgs
)

# Install Python modules directory
install(DIRECTORY navigate_to_location_action
  DESTINATION lib/python3.10/site-packages
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

# Install Python scripts
install(PROGRAMS
  navigate_to_location_action/navigate_to_location_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME navigate_to_location_server
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files if needed
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
