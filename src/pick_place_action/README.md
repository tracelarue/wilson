# Pick and Place Action Server

A ROS2 action server for pick and place operations using MoveIt Task Constructor.

## Overview

This package provides an action server that accepts pick requests with object coordinates in any frame (default: `depth_camera_optical`) and executes a pick sequence using MoveIt Task Constructor.

## Action Interface

### Action Name
`pick_object`

### Goal Message
```
geometry_msgs/Point target_position      # Object position
string target_frame                      # Frame of target_position (e.g., "depth_camera_optical")
float32 cylinder_height                  # Height of cylindrical object (default: 0.122)
float32 cylinder_radius                  # Radius of cylindrical object (default: 0.033)
```

### Feedback Message
```
string current_stage                     # Current execution stage
float32 progress_percentage              # Progress (0-100%)
```

### Result Message
```
bool success                             # Whether the pick succeeded
string message                           # Status/error message
geometry_msgs/Pose final_object_pose     # Final object pose in world frame
```

## Building

```bash
cd ~/wilson
colcon build --packages-select pick_place_action
source install/setup.bash
```

## Usage

### Launch the Action Server

```bash
ros2 launch pick_place_action pick_action_server.launch.py
```

### Send a Goal via Command Line

```bash
# Basic example - pick object at x=0.45, y=-0.07, z=0.1 in depth_camera_optical frame
ros2 action send_goal /pick_object pick_place_action/action/PickObject "{
  target_position: {x: 0, y: -0.1, z: 0.45},
  target_frame: 'depth_camera_optical',
  cylinder_height: 0.122,
  cylinder_radius: 0.033
}" --feedback
```

### Send a Goal from Python

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from pick_place_action.action import PickObject
from geometry_msgs.msg import Point

class PickObjectClient(Node):
    def __init__(self):
        super().__init__('pick_object_client')
        self._action_client = ActionClient(self, PickObject, 'pick_object')

    def send_goal(self, x, y, z, frame='depth_camera_optical'):
        goal_msg = PickObject.Goal()
        goal_msg.target_position = Point(x=x, y=y, z=z)
        goal_msg.target_frame = frame
        goal_msg.cylinder_height = 0.122
        goal_msg.cylinder_radius = 0.033

        self._action_client.wait_for_server()

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Stage: {feedback.current_stage}, Progress: {feedback.progress_percentage}%')

    def result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info(f'Success! {result.message}')
        else:
            self.get_logger().error(f'Failed: {result.message}')

def main(args=None):
    rclpy.init(args=args)
    client = PickObjectClient()

    # Example: Pick object at position relative to depth camera
    client.send_goal(x=0.45, y=-0.07, z=0.1, frame='depth_camera_optical')

    rclpy.spin(client)

if __name__ == '__main__':
    main()
```

## Features

- **Dynamic Frame Transformation**: Automatically transforms target coordinates from the specified frame to the world frame
- **Configurable Object Dimensions**: Supports custom cylinder dimensions for different objects
- **Progress Feedback**: Provides real-time feedback on execution progress
- **Dynamic Approach Vector**: Computes approach direction based on robot base position
- **Full Pick Sequence**: Executes complete pick sequence including:
  - Open gripper
  - Move to pre-grasp position
  - Approach object
  - Generate grasp pose
  - Close gripper
  - Attach object
  - Lift object
  - Return home

## Configuration

The action server uses the following MoveIt group names (defined in the source code):
- **arm_group**: `"arm"`
- **hand_group**: `"gripper"`
- **hand_frame**: `"end_effector_frame"`

To change these, modify the constants in [src/pick_object_action_server.cpp](src/pick_object_action_server.cpp:268-270).

## Troubleshooting

### Action server not available
Make sure the action server is running and MoveIt is properly configured:
```bash
ros2 action list
# Should show: /pick_object
```

### Transform errors
Ensure all required TF frames are being published:
```bash
ros2 run tf2_ros tf2_echo world depth_camera_optical
```

### Planning failures
Check that:
1. The target position is reachable
2. Object dimensions are reasonable
3. MoveIt is properly configured for your robot
4. Collision checking is not too restrictive

## Dependencies

- `rclcpp`
- `rclcpp_action`
- `moveit_task_constructor_core`
- `tf2_ros`
- `geometry_msgs`
- `shape_msgs`

## License

TODO: Add license information
