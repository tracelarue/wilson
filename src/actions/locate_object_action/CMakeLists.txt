cmake_minimum_required(VERSION 3.8)
project(locate_object_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate action messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/LocateObject.action"
  DEPENDENCIES geometry_msgs
)

# Install Python modules directory
install(DIRECTORY locate_object_action
  DESTINATION lib/python3.10/site-packages
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

# Install Python scripts
install(PROGRAMS
  locate_object_action/locate_object_action_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME locate_object_action_server
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
