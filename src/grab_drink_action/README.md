# Grab Drink Action Server

A ROS2 action server for grabbing drinks using MoveIt Task Constructor.

## Overview

This package provides an action server that accepts grab requests with drink coordinates in any frame (default: `depth_camera_link_optical`) and executes a grab sequence using MoveIt Task Constructor.

## Action Interface

### Action Name
`grab_drink`

### Goal Message
```
geometry_msgs/Point target_position      # Drink position
string target_frame                      # Frame of target_position (e.g., "depth_camera_link_optical")
```

**Note:** Drink dimensions are hardcoded (height: 0.122m, radius: 0.033m for standard can size).

### Feedback Message
```
string current_stage                     # Current execution stage
float32 progress_percentage              # Progress (0-100%)
```

### Result Message
```
bool success                             # Whether the grab succeeded
string message                           # Status/error message
geometry_msgs/Pose final_drink_pose      # Final drink pose in world frame
```

## Building

```bash
cd ~/wilson
colcon build --packages-select grab_drink_action
source install/setup.bash
```

## Usage

### Launch the Action Server

```bash
ros2 launch grab_drink_action grab_drink_server.launch.py
```

### Send a Goal via Command Line

```bash
# Basic example - grab drink at x=0, y=0.1, z=0.4 in depth_camera_link_optical frame
ros2 action send_goal /grab_drink grab_drink_action/action/GrabDrink "{
  target_position: {x: 0, y: 0.1, z: 0.4},
  target_frame: 'depth_camera_link_optical'
}" --feedback
```

### Send a Goal from Python

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from grab_drink_action.action import GrabDrink
from geometry_msgs.msg import Point

class GrabDrinkClient(Node):
    def __init__(self):
        super().__init__('grab_drink_client')
        self._action_client = ActionClient(self, GrabDrink, 'grab_drink')

    def send_goal(self, x, y, z, frame='depth_camera_link_optical'):
        goal_msg = GrabDrink.Goal()
        goal_msg.target_position = Point(x=x, y=y, z=z)
        goal_msg.target_frame = frame

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
    client = GrabDrinkClient()

    # Example: Grab drink at position relative to depth camera
    client.send_goal(x=0.45, y=-0.07, z=0.1, frame='depth_camera_link_optical')

    rclpy.spin(client)

if __name__ == '__main__':
    main()
```

## Features

- **Dynamic Frame Transformation**: Automatically transforms target coordinates from the specified frame to the world frame
- **Standard Drink Dimensions**: Uses hardcoded dimensions optimized for standard can size (height: 0.122m, radius: 0.033m)
- **Progress Feedback**: Provides real-time feedback on execution progress
- **Dynamic Approach Vector**: Computes approach direction based on robot base position
- **Full Grab Sequence**: Executes complete grab sequence including:
  - Open gripper
  - Move to pre-grasp position
  - Approach drink
  - Generate grasp pose
  - Close gripper
  - Attach drink
  - Lift drink
  - Return home

## Configuration

The action server uses the following MoveIt group names (defined in the source code):
- **arm_group**: `"arm"`
- **hand_group**: `"gripper"`
- **hand_frame**: `"end_effector_frame"`

To change these, modify the constants in [src/grab_drink_action_server.cpp](src/grab_drink_action_server.cpp:334-336).

## Troubleshooting

### Action server not available
Make sure the action server is running and MoveIt is properly configured:
```bash
ros2 action list
# Should show: /grab_drink
```

### Transform errors
Ensure all required TF frames are being published:
```bash
ros2 run tf2_ros tf2_echo base_link depth_camera_link_optical
```

### Planning failures
Check that:
1. The target position is reachable
2. MoveIt is properly configured for your robot
3. Collision checking is not too restrictive

## Dependencies

- `rclcpp`
- `rclcpp_action`
- `moveit_task_constructor_core`
- `tf2_ros`
- `geometry_msgs`
- `shape_msgs`

## License

TODO: Add license information
