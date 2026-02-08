cmake_minimum_required(VERSION 3.8)
project(network_actions)

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
  "action/MiniFridge.action"
  DEPENDENCIES std_msgs action_msgs
)

install(DIRECTORY network_actions
  DESTINATION lib/python3.10/site-packages
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

install(PROGRAMS
  network_actions/mini_fridge_action_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mini_fridge_action_server
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
