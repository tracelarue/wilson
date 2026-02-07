cmake_minimum_required(VERSION 3.8)
project(ir_signal_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/IrSignal.action"
  DEPENDENCIES std_msgs action_msgs
)

install(DIRECTORY ir_signal_action
  DESTINATION lib/python3.10/site-packages
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

install(PROGRAMS
  ir_signal_action/ir_signal_action_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ir_signal_action_server
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
