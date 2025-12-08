# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About Wilson

Wilson is an autonomous differential-drive robot with a 4-DOF manipulator arm, designed for beverage retrieval tasks. It uses ROS2 Humble and integrates Google's Gemini Live API for conversational AI control. The robot features navigation (Nav2), manipulation (MoveIt2), and multimodal AI perception.

## Build and Run Commands

### Build System
```bash
# Full build with symlink install (recommended for development)
colcon build --symlink-install

# Build specific package
colcon build --packages-select <package_name>

# Source the workspace
source install/setup.bash
```

**Important:** Always source `install/setup.bash` after building before running any ROS2 commands.

### Launch Simulation
```bash
# Complete Wilson simulation (Gazebo + RViz + Nav2 + MoveIt)
colcon build --symlink-install && \
source install/setup.bash && \
ros2 launch wilson wilson_sim.launch.py
```

This launches in the following sequence:
1. Gazebo simulation with robot spawned
2. Nav2 navigation stack (3s delay)
3. AMCL localization (5s delay)
4. MoveIt move_group (8s delay)
5. GrabDrinkActionServer (8s delay)
6. Initial pose publisher (15s delay)

### Launch Real Robot
```bash
ros2 launch wilson wilson_real.launch.py
```

### Testing Individual Components
```bash
# Test navigation only
ros2 launch wilson navigation_launch.py

# Test manipulation only
ros2 launch wilson move_group.launch.py

# Test grab drink action server
ros2 launch grab_drink_action grab_drink_server.launch.py

# Test Gemini AI node (requires GOOGLE_API_KEY in .env)
ros2 run gemini gemini_node
```

### Useful Debug Commands
```bash
# Check running nodes
ros2 node list

# Check available actions
ros2 action list

# Monitor transforms
ros2 run tf2_ros tf2_echo base_link end_effector_frame

# View joint states
ros2 topic echo /joint_states --once

# Send navigation goal (map frame)
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: 'map'}, pose: {position: {x: -2.0, y: -2.5, z: 0.0}}}}" --feedback

# Send grab drink goal (depth camera frame)
ros2 action send_goal /grab_drink grab_drink_action/action/GrabDrink "{target_position: {x: 0.45, y: -0.07, z: 0.1}, target_frame: 'depth_camera_link_optical'}" --feedback
```

## High-Level Architecture

### Core Subsystems

**Hardware Abstraction Layer (ROS2 Control)**
- Controllers: `diff_drive_controller`, `arm_controller`, `gripper_controller`, `joint_state_broadcaster`
- Hardware interfaces: Gazebo simulation (fake hardware) or real Arduino-based controllers
- All controllers run at 100Hz update rate

**Navigation Stack (Nav2)**
- Global planner: Navfn
- Local planner: DWB with velocity smoother
- Localization: AMCL particle filter (500-1000 particles)
- Map frame → odom frame → base_link frame transform chain
- Named locations: Mini fridge (0.5, 0), Living room (-2, -2.5), Kitchen (-0.5, 3)

**Manipulation Stack (MoveIt2)**
- Planning group: `arm` (joint_1 through joint_4, end_effector_roll/yaw)
- Gripper group: `gripper` (left/right finger joints)
- IK solver: KDL plugin
- End effector frame: `end_effector_frame`
- MoveIt Task Constructor (MTC) used for complex multi-stage tasks

**AI Integration (Gemini Live)**
- Multimodal streaming: Gemini 2.5 Flash Live API
- Tools: `navigate_to_location`, `image_input`, `get_distance`
- Audio I/O: 16kHz microphone input, 24kHz speaker output (resampled from hardware rates)
- Vision: RGB camera + depth camera for 3D object localization
- Action client for Nav2 navigate_to_pose

### Package Responsibilities

- **wilson**: Main robot description (URDF), launch files, maps, configurations
- **wilson_moveit_config**: MoveIt configuration (kinematics, joint limits, controllers)
- **grab_drink_action**: MTC-based grasp action server for drink manipulation
- **gemini**: Gemini Live API integration node with ROS2 action clients
- **depth_cam**: Arducam ToF camera driver
- **diffdrive_arduino**: Differential drive base hardware interface
- **ldlidar_stl_ros**: 2D LiDAR driver (LD-19)

### Key Integration Points

**Coordinate Frame Hierarchy:**
```
map (from map server)
└── odom (from AMCL)
    └── base_link (from diff_drive odometry)
        └── ... (robot kinematic chain from URDF)
            └── depth_camera_link_optical (camera frame)
```

**Action Interfaces:**
- `/navigate_to_pose`: Nav2 navigation action (map frame)
- `/grab_drink`: Custom grasp action (any frame, typically depth_camera_link_optical)

**Topic Flow:**
- Sensors → `/scan`, `/camera/image_raw`, `/camera/depth/image_raw`
- Joint states → `/joint_states` (from joint_state_broadcaster)
- Odometry → `/odom` (from diff_drive_controller)
- Velocity commands → `/cmd_vel` (to diff_drive_controller)

### Critical Architecture Notes

1. **Timing Dependencies**: The launch file uses staged delays to prevent race conditions. Do not remove timers without understanding service availability dependencies.

2. **Transform Management**: All coordinate transformations go through TF2. Never bypass TF for coordinate conversions. The grab_drink action automatically handles frame transformations.

3. **Controller Namespaces**:
   - Simulation uses parameters from `gazebo_controller_manager.yaml`
   - Real robot uses `robot_controller_manager.yaml`
   - Launch argument `use_fake_hardware` switches between them

4. **MoveIt Integration**:
   - The `grab_drink_action` requires MoveIt's `move_group` node to be running
   - Planning scene is dynamically updated with detected objects
   - Collision checking can be selectively disabled (e.g., gripper-object collision during grasp)

5. **Gemini AI Tools**:
   - Function declarations define available robot capabilities
   - Tool responses are streamed back to Gemini for reasoning
   - Image analysis uses synchronous Gemini 2.0 Flash (not Live API)
   - Depth-based 3D localization requires camera FOV calibration

## Development Workflow

### Adding a New Manipulation Task

1. Create action definition in `<package>/action/<ActionName>.action`
2. Implement action server inheriting from `rclcpp_action::Server` (C++) or `ActionServer` (Python)
3. Use MoveIt Task Constructor for multi-stage manipulation:
   ```cpp
   mtc::Task task;
   task.stages()->addChild(std::make_unique<mtc::stages::CurrentState>("current"));
   task.stages()->addChild(std::make_unique<mtc::stages::MoveTo>("move", planner));
   // ... add more stages
   task.plan();
   ```
4. Update CMakeLists.txt/package.xml with action dependencies
5. Create launch file to start the action server

### Adding a Gemini AI Tool

1. Define function declaration in `gemini/gemini/gemini_node.py`:
   ```python
   tools.FunctionDeclaration(
       name="your_tool_name",
       description="What this tool does",
       parameters=content.Schema(
           type=content.Type.OBJECT,
           properties={
               "param1": content.Schema(type=content.Type.STRING),
           }
       )
   )
   ```
2. Implement handler in `handle_function_call` method
3. Return `FunctionResponse` with result

### Modifying Robot Configuration

- URDF changes: Edit `src/wilson/description/wilson.urdf.xacro` or sub-files
- Controller parameters: Edit `src/wilson/config/*_controller_manager.yaml`
- MoveIt configuration: Use MoveIt Setup Assistant or manually edit `src/wilson_moveit_config/config/*`
- Navigation parameters: Edit `src/wilson/config/nav2_params.yaml`

## Environment Setup

### Required Environment Variables
```bash
# In Docker container or host system
export ROS_DOMAIN_ID=7
export DISPLAY=:0  # For GUI applications
```

### API Keys
Create `.env` file in workspace root:
```
GOOGLE_API_KEY="your_api_key_here"
```

### Docker Development
```bash
# Build image
docker build -t wilson_image .

# Run container (from wilson directory)
docker run -it --user ros --network=host --ipc=host \
  -v $PWD:/wilson \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --env=DISPLAY=:0 \
  --env=QT_X11_NO_MITSHM=1 \
  -v /dev:/dev \
  --privileged \
  --name wilson \
  wilson_image

# Access running container
docker exec -it wilson /bin/bash
```

## Common Issues and Solutions

**"No planning solutions found" (MoveIt)**
- Check if target pose is within workspace limits
- Verify joint limits in `wilson_moveit_config/config/joint_limits.yaml`
- Increase planning time or use different planner

**Navigation not starting**
- Ensure initial pose is published (happens at T=15s in sim)
- Check AMCL particle cloud is converged in RViz
- Verify map server loaded correctly: `ros2 topic echo /map --once`

**Transform lookup failures**
- Check all required frames exist: `ros2 run tf2_tools view_frames`
- Ensure `use_sim_time` parameter matches simulation state
- Verify controller managers are running: `ros2 control list_controllers`

**Gemini not responding**
- Check API key is valid and in `.env` file
- Verify microphone permissions: `arecord -d 5 test.wav && aplay test.wav`
- Ensure rosbridge is NOT running (Gemini node uses direct ROS2, not rosbridge)

**Controllers fail to spawn**
- Check URDF loads without errors: `ros2 launch wilson rsp.launch.py`
- Verify `ros2_control` tags exist in URDF
- Ensure controller manager is running before spawning controllers

## Package Dependencies

### C++ Packages
- rclcpp, rclcpp_action
- moveit_task_constructor_core
- tf2_ros, tf2_geometry_msgs
- geometry_msgs, sensor_msgs, std_msgs

### Python Packages
- google-genai (Gemini API)
- pyaudio (audio I/O)
- scipy (signal resampling)
- opencv-python (image processing)
- ArducamDepthCamera (depth camera driver)

### ROS2 Packages
- ros-humble-ros2-control
- ros-humble-ros2-controllers
- ros-humble-navigation2
- ros-humble-nav2-bringup
- ros-humble-moveit
- ros-humble-gazebo-ros-pkgs

## Code Conventions

- Joint naming: `joint_1` through `joint_4`, `end_effector_roll_joint`, `end_effector_yaw_joint`, `gripper_left_finger_joint`, `gripper_right_finger_joint`
- Planning groups: `arm`, `gripper`
- Frame naming: Use TF2 standard naming (`base_link`, `base_footprint`, `end_effector_frame`)
- Launch files: Place in `launch/sim/`, `launch/real/`, or `launch/base/` depending on context
- Configuration files: Place in `config/` directory within package
