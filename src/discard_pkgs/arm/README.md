# 4-DOF Robotic Arm Package

This ROS2 Humble package provides a 4-DOF robotic arm simulation with ros2_control and joint trajectory controller support.

## Package Structure

```
arm/
├── bringup/           # Test scripts and utilities
├── config/            # Configuration files
├── description/       # Robot description files (URDF/XACRO)
├── launch/           # Launch files
├── CMakeLists.txt
├── package.xml
└── README.md
```

## Features

- 4-DOF robotic arm with revolute joints
- Full ros2_control integration
- Joint trajectory controller for position control
- Gazebo simulation support
- RViz visualization
- Test scripts for controller validation

## Joint Configuration

The arm has 4 revolute joints:

1. **joint_1** - Base rotation (±180°)
2. **joint_2** - Shoulder (±90°)
3. **joint_3** - Elbow (±180°)
4. **joint_4** - Wrist roll (±180°)

## Quick Start

### 1. Build the package

```bash
cd /home/trace/robot
colcon build --packages-select arm
source install/setup.bash
```

### 2. Launch the arm with fake hardware

```bash
ros2 launch arm arm.launch.py
```

### 3. Launch the arm in Gazebo simulation

```bash
ros2 launch arm arm_gazebo.launch.py
```

### 4. Test the controller

In a new terminal:

```bash
source /home/trace/robot/install/setup.bash
python3 /home/trace/robot/src/arm/bringup/test_arm_controller.py
```

## Usage

### Controlling the Arm

The arm is controlled via the joint trajectory controller. You can send commands to:

```
/arm_controller/joint_trajectory
```

### Monitoring Joint States

Joint states are published on:

```
/joint_states
```

### Available Services

- `/controller_manager/list_controllers` - List active controllers
- `/controller_manager/switch_controller` - Switch controllers
- `/arm_controller/query_state` - Query controller state

### Example Commands

#### Send a simple trajectory:

```bash
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "
header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: ''
joint_names: ['joint_1', 'joint_2', 'joint_3', 'joint_4']
points:
- positions: [0.5, 0.3, -0.5, 0.2]
  velocities: [0.0, 0.0, 0.0, 0.0]
  time_from_start:
    sec: 3
    nanosec: 0
"
```

#### List active controllers:

```bash
ros2 control list_controllers
```

#### Get joint states:

```bash
ros2 topic echo /joint_states
```

## Configuration Files

### Controllers (`config/arm_controllers.yaml`)

Contains configuration for:
- Joint state broadcaster
- Joint trajectory controller
- Control parameters and constraints

### RViz (`config/arm.rviz`)

Pre-configured RViz setup with:
- Robot model visualization
- TF frames
- Joint state monitoring

## Hardware Integration

To use with real hardware, modify the `<hardware>` section in `description/urdf/arm_ros2_control.urdf.xacro`:

```xml
<hardware>
  <plugin>your_hardware_interface/YourHardwareInterface</plugin>
  <!-- Add your hardware-specific parameters here -->
</hardware>
```

## Troubleshooting

### Controller not starting

1. Check if ros2_control_node is running:
   ```bash
   ros2 node list | grep ros2_control
   ```

2. Verify controller configuration:
   ```bash
   ros2 control list_controllers
   ```

### Gazebo simulation issues

1. Make sure Gazebo is properly installed:
   ```bash
   sudo apt install gazebo ros-humble-gazebo-ros-pkgs
   ```

2. Check if the robot spawns correctly:
   ```bash
   ros2 topic list | grep robot_description
   ```

### Joint limits

The arm respects joint limits defined in the URDF. Check the limits in `description/urdf/arm.urdf.xacro` if movements are restricted.

## Dependencies

- ros2_control
- ros2_controllers
- joint_trajectory_controller
- gazebo_ros2_control
- robot_state_publisher
- xacro
- rviz2

Install missing dependencies with:

```bash
sudo apt update
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-joint-trajectory-controller ros-humble-gazebo-ros2-control
```
