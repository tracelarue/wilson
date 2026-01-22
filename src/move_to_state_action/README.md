# move_to_state_action

ROS2 Humble action server for moving robot groups to named states using MoveIt. This package provides a simple, reusable interface for commanding the robot to move to predefined poses defined in the SRDF.

## Overview

The `move_to_state_action` package provides an action server that:
1. Accepts a named state as input (e.g., "ready", "zero", "open")
2. Automatically detects which MoveIt planning group the state belongs to
3. Plans and executes motion to reach the target state
4. Provides real-time feedback during planning and execution

**Key Feature**: Field names have **no underscores** (`statename` instead of `state_name`) for full compatibility with MCP (Model Context Protocol) servers.

## Features

- **Auto-Detection**: Automatically determines which group (arm/gripper) a state belongs to by querying the SRDF
- **Simple Interface**: Single string input - just provide the state name
- **MCP Compatible**: Field names without underscores work seamlessly with MCP servers
- **Safe Execution**: Uses MoveIt's collision checking and motion planning
- **Real-time Feedback**: Progress updates during planning and execution
- **Error Handling**: Graceful handling of invalid states, planning failures, and execution errors

## Prerequisites

### System Requirements
- ROS2 Humble
- MoveIt 2
- Robot with SRDF containing named group states

### Available States (Wilson Robot)

Based on [wilson.srdf](../wilson_moveit_config/config/wilson.srdf):

**Arm States:**
- `ready` - Default ready position
- `idle` - Idle/rest position
- `zero` - All joints at zero
- `ready_to_grab` - Pre-grasp position
- `ground_grab` - Low grabbing position
- `air_grab` - Mid-air grabbing position

**Gripper States:**
- `open` - Gripper fully open
- `close` - Gripper fully closed
- `can` - Gripper sized for a can

## Installation

1. Package is located in your ROS2 workspace:
   ```bash
   cd ~/wilson/src/move_to_state_action
   ```

2. Build the package:
   ```bash
   cd ~/wilson
   colcon build --packages-select move_to_state_action --symlink-install
   source install/setup.bash
   ```

## Usage

### Launch the Action Server

```bash
# Source your workspace
source ~/wilson/install/setup.bash

# Launch the action server
ros2 launch move_to_state_action move_to_state_server.launch.py

# For simulation (with sim time)
ros2 launch move_to_state_action move_to_state_server.launch.py use_sim_time:=true
```

### Send Action Goals

**Command Line:**
```bash
# Move arm to ready position
ros2 action send_goal /move_to_state move_to_state_action/action/MoveToState \
  "{statename: 'ready'}" --feedback

# Move arm to zero position
ros2 action send_goal /move_to_state move_to_state_action/action/MoveToState \
  "{statename: 'zero'}" --feedback

# Open gripper
ros2 action send_goal /move_to_state move_to_state_action/action/MoveToState \
  "{statename: 'open'}" --feedback

# Close gripper to can size
ros2 action send_goal /move_to_state move_to_state_action/action/MoveToState \
  "{statename: 'can'}" --feedback
```

**Python Client:**
```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from move_to_state_action.action import MoveToState

class MoveToStateClient(Node):
    def __init__(self):
        super().__init__('move_to_state_client')
        self._action_client = ActionClient(self, MoveToState, 'move_to_state')

    def send_goal(self, state_name):
        goal_msg = MoveToState.Goal()
        goal_msg.statename = state_name  # Note: no underscore (MCP compatible!)

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'{feedback.currentstatus} ({feedback.progresspercentage:.1f}%)')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info(f'Success! {result.message}')
        else:
            self.get_logger().error(f'Failed: {result.message}')

def main(args=None):
    rclpy.init(args=args)
    client = MoveToStateClient()
    client.send_goal('ready')
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**C++ Client:**
```cpp
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "move_to_state_action/action/move_to_state.hpp"

using MoveToState = move_to_state_action::action::MoveToState;
using GoalHandleMoveToState = rclcpp_action::ClientGoalHandle<MoveToState>;

class MoveToStateClient : public rclcpp::Node {
public:
    MoveToStateClient() : Node("move_to_state_client") {
        client_ = rclcpp_action::create_client<MoveToState>(this, "move_to_state");
    }

    void send_goal(const std::string& state_name) {
        if (!client_->wait_for_action_server(std::chrono::seconds(10))) {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
            return;
        }

        auto goal_msg = MoveToState::Goal();
        goal_msg.statename = state_name;  // Note: no underscore (MCP compatible!)

        auto send_goal_options = rclcpp_action::Client<MoveToState>::SendGoalOptions();
        send_goal_options.feedback_callback =
            [this](auto, const auto feedback) {
                RCLCPP_INFO(this->get_logger(), "%s (%.1f%%)",
                    feedback->currentstatus.c_str(),
                    feedback->progresspercentage);
            };

        client_->async_send_goal(goal_msg, send_goal_options);
    }

private:
    rclcpp_action::Client<MoveToState>::SharedPtr client_;
};
```

## Action Interface

### Goal
```
string statename    # Name of the state (e.g., "ready", "zero", "open")
```

### Result
```
bool success        # True if motion completed successfully
string message      # Status message (success or error description)
```

### Feedback
```
string currentstatus         # Current operation status
float32 progresspercentage   # Progress from 0.0 to 100.0
```

## MCP Server Compatibility

This action is designed for full compatibility with MCP (Model Context Protocol) servers, which have issues with underscores in field names.

**Field Names (No Underscores):**
- ✅ `statename` (not `state_name`)
- ✅ `currentstatus` (not `current_status`)
- ✅ `progresspercentage` (not `progress_percentage`)

This allows seamless integration with Gemini AI tools and other MCP-based systems.

## Topics and Services

### Action Servers
- `/move_to_state` (move_to_state_action/action/MoveToState): Main action interface

## Architecture

### State Detection Flow
1. **Load Robot Model**: Query robot_description to access SRDF
2. **Scan Groups**: Check each MoveIt planning group (arm, gripper, arm_gripper)
3. **Find State**: Search for the requested state name in group's default states
4. **Return Group**: Return the first group containing the state

### Execution Flow
1. **Detect Group**: Automatically determine which group the state belongs to
2. **Initialize MoveGroup**: Create MoveGroupInterface for the detected group
3. **Set Target**: Use `setNamedTarget(statename)`
4. **Plan**: Generate collision-free trajectory
5. **Execute**: Execute the planned trajectory
6. **Verify**: Return success/failure result

## Error Handling

The action server handles various failure scenarios:

- **State Not Found**: Returns error if state doesn't exist in any group
- **Planning Failure**: Returns error if no valid path can be found
- **Execution Failure**: Returns error if trajectory execution fails
- **Cancellation**: Supports goal cancellation before and during execution

## Common Issues and Solutions

### "State 'xyz' not found in SRDF"
- Check that the state is defined in your SRDF file
- Verify state name spelling matches exactly (case-sensitive)
- Ensure SRDF is loaded correctly in robot_description

### "Motion planning failed"
- Check if target state is kinematically reachable
- Verify joint limits in MoveIt configuration
- Increase planning time if needed
- Check for collision constraints

### "Action server not available"
- Ensure action server is launched: `ros2 node list | grep move_to_state`
- Check MoveIt move_group is running
- Verify robot_description is published

## Integration with Wilson

This package integrates with the Wilson robot system:

1. **Add to Main Launch** (optional - can be launched separately):
   ```python
   from launch.actions import IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource

   move_to_state_server = IncludeLaunchDescription(
       PythonLaunchDescriptionSource([
           os.path.join(get_package_share_directory('move_to_state_action'),
                       'launch', 'move_to_state_server.launch.py')
       ]),
       launch_arguments={'use_sim_time': use_sim_time}.items()
   )
   ```

2. **Call from Gemini AI** (MCP compatible):
   ```python
   def move_to_state_tool(self, statename: str):
       """Tool for Gemini to move robot to named states."""
       goal = MoveToState.Goal()
       goal.statename = statename  # MCP-compatible field name!

       future = self.move_to_state_client.send_goal_async(goal)
       # Handle result...
   ```

## Development

### Adding New States

1. Edit your SRDF file:
   ```xml
   <group_state name="my_new_state" group="arm">
       <joint name="joint_1" value="0.5"/>
       <joint name="joint_2" value="1.0"/>
       <!-- ... -->
   </group_state>
   ```

2. Reload robot description (restart MoveIt)

3. Test the new state:
   ```bash
   ros2 action send_goal /move_to_state move_to_state_action/action/MoveToState \
     "{statename: 'my_new_state'}" --feedback
   ```

### Debugging

Enable detailed logging:
```bash
ros2 run move_to_state_action move_to_state_action_server --ros-args \
  --log-level move_to_state_action_server:=debug
```

Monitor action feedback:
```bash
ros2 action send_goal /move_to_state move_to_state_action/action/MoveToState \
  "{statename: 'ready'}" --feedback
```

## Comparison with Alternatives

### vs Direct MoveGroupInterface
- ✅ **move_to_state_action**: Provides action interface with feedback, cancellation, async execution
- ❌ **Direct MoveGroup**: Requires manual node creation, no standardized interface

### vs MoveIt Task Constructor
- ✅ **move_to_state_action**: Simple, fast for basic moves to named states
- ❌ **MTC**: Overkill for simple named state moves, but better for complex multi-stage tasks

### vs Service Interface
- ✅ **move_to_state_action**: Supports feedback, cancellation, progress updates
- ❌ **Service**: No feedback during execution, no cancellation support

## License

MIT License

## Author

Created for the Wilson autonomous robot project.

## References

- [ROS2 Actions Documentation](https://docs.ros.org/en/humble/Tutorials/Intermediate/Writing-an-Action-Server-Client/Cpp.html)
- [MoveIt 2 Documentation](https://moveit.picknik.ai/main/index.html)
- [Wilson Robot Documentation](../README.md)
- [MCP Protocol](https://modelcontextprotocol.io/)
