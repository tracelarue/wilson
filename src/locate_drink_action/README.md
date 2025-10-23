# locate_drink_action

ROS2 Humble action server package for robotic drink localization using Gemini AI computer vision. This package enables a differential-drive robot to detect a specified drink using camera input and position itself optimally relative to the drink.

## Overview

The `locate_drink_action` package provides an action server that:
1. Uses Gemini AI to detect and localize a drink by name in camera images
2. Calculates the 3D position of the drink relative to the camera frame
3. Controls the robot's differential drive base to achieve optimal positioning
4. Provides real-time feedback on detection status and positioning errors

## Features

- **AI-Powered Detection**: Uses Google's Gemini 2.0 Flash model for robust drink detection
- **3D Localization**: Combines RGB and depth camera data for accurate positioning
- **Adaptive Control**: Proportional controller for smooth robot movement
- **Configurable Parameters**: All control gains, tolerances, and timeouts adjustable via YAML
- **Comprehensive Feedback**: Real-time status updates during positioning
- **Error Handling**: Robust error handling for API failures, timeouts, and detection failures

## Prerequisites

### System Requirements
- ROS2 Humble
- Python 3.10+
- RGB and Depth camera (e.g., Intel RealSense, Arducam ToF)
- Differential drive robot base

### Python Dependencies
Install the required Python packages:
```bash
pip3 install google-genai opencv-python pillow python-dotenv numpy
```

### API Key
You need a Google API key for Gemini AI. Create a `.env` file in your workspace root:
```bash
# /home/trace/wilson/.env
GOOGLE_API_KEY="your_api_key_here"
```

Get your API key from: https://aistudio.google.com/apikey

## Installation

1. Clone or copy this package to your ROS2 workspace:
   ```bash
   cd ~/wilson/src
   # Package should be in: src/locate_drink_action/
   ```

2. Build the package:
   ```bash
   cd ~/wilson
   colcon build --packages-select locate_drink_action --symlink-install
   source install/setup.bash
   ```

## Usage

### Launch the Action Server

```bash
# Source your workspace
source ~/wilson/install/setup.bash

# Launch the action server
ros2 launch locate_drink_action locate_drink_server.launch.py
```

### Send an Action Goal

Using the command line:
```bash
ros2 action send_goal /locate_drink locate_drink_action/action/LocateDrink \
  "{drink_name: 'Coca Cola'}" --feedback
```

Using Python:
```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from locate_drink_action.action import LocateDrink

class LocateDrinkClient(Node):
    def __init__(self):
        super().__init__('locate_drink_client')
        self._action_client = ActionClient(self, LocateDrink, 'locate_drink')

    def send_goal(self, drink_name):
        goal_msg = LocateDrink.Goal()
        goal_msg.drink_name = drink_name

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Status: {feedback.current_status}')
        self.get_logger().info(f'Position: x={feedback.current_position.x:.3f}, z={feedback.current_position.z:.3f}')
        self.get_logger().info(f'Errors: x={feedback.error_x:.3f}, z={feedback.error_z:.3f}')

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
            self.get_logger().info(f'Final position: x={result.final_position.x:.3f}, z={result.final_position.z:.3f}')
        else:
            self.get_logger().error(f'Failed: {result.message}')

def main(args=None):
    rclpy.init(args=args)
    client = LocateDrinkClient()
    client.send_goal('Sprite')
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Configuration

All parameters are configurable via [config/locate_drink_params.yaml](config/locate_drink_params.yaml):

### Target Position Parameters
- `target_x` (default: 0.0): Target horizontal position in meters (centered in camera view)
- `target_z` (default: 0.4): Target distance from camera in meters
- `position_tolerance` (default: 0.03): Tolerance in meters for successful positioning

### Control Parameters
- `k_linear` (default: 0.5): Linear velocity proportional gain
- `k_angular` (default: 1.0): Angular velocity proportional gain
- `max_linear_vel` (default: 0.3): Maximum linear velocity in m/s
- `max_angular_vel` (default: 0.5): Maximum angular velocity in rad/s
- `min_movement_threshold` (default: 0.005): Minimum velocity to prevent oscillations

### Camera Parameters
- `h_fov` (default: 1.089): Horizontal field of view in radians
- `aspect_ratio` (default: 1.333): Image aspect ratio (width/height)
- `camera_frame` (default: 'depth_camera_link_optical'): TF frame for camera
- `image_width` (default: 640): Image width in pixels
- `image_height` (default: 480): Image height in pixels

### Operation Parameters
- `max_detection_attempts` (default: 5): Maximum detection retries before failure
- `control_loop_rate` (default: 2.0): Control loop frequency in Hz
- `overall_timeout` (default: 60.0): Overall operation timeout in seconds
- `api_timeout` (default: 10.0): Gemini API timeout in seconds
- `movement_settle_time` (default: 0.5): Wait time after velocity commands in seconds

## Action Interface

### Goal
```
string drink_name  # Name of the drink to locate (e.g., "Coca Cola", "Sprite", "Water bottle")
```

### Result
```
bool success                      # True if positioning successful
string message                    # Status message
geometry_msgs/Point final_position  # Final 3D position (x, y, z) in meters
```

### Feedback
```
string current_status              # Current operation status
geometry_msgs/Point current_position  # Current drink position (x, y, z) in meters
float32 error_x                    # Horizontal error from target in meters
float32 error_z                    # Distance error from target in meters
uint8 detection_attempts           # Number of detection attempts made
```

## Coordinate System

The action server operates in the `depth_camera_link_optical` frame:
- **X-axis**: Horizontal (positive = right from camera's view)
- **Y-axis**: Vertical (positive = down from camera's view) - *not controlled*
- **Z-axis**: Depth/distance from camera (positive = forward)

**Target Position** (configurable):
- `x = 0.0`: Drink centered horizontally in camera view
- `z = 0.4`: Drink is 0.4 meters from the camera

The robot moves via differential drive (`/cmd_vel`) to minimize the error between the current and target positions.

## Topics

### Subscribed Topics
- `/camera/image_raw` (sensor_msgs/Image): RGB camera feed
- `/camera/depth/image_raw` (sensor_msgs/Image): Depth camera feed

### Published Topics
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands for differential drive base

### Action Servers
- `/locate_drink` (locate_drink_action/action/LocateDrink): Main action interface

## Architecture

### Detection Pipeline
1. **Image Acquisition**: Capture synchronized RGB and depth images
2. **AI Detection**: Send RGB image to Gemini AI with drink name
3. **Bounding Box Extraction**: Parse JSON response for drink location
4. **Depth Lookup**: Extract depth value at bounding box center
5. **3D Calculation**: Project pixel coordinates to 3D using camera FOV and depth

### Control Loop
1. **Detect & Localize**: Get current drink position (x, z)
2. **Calculate Error**: `error_x = current_x - target_x`, `error_z = current_z - target_z`
3. **Proportional Control**:
   - Linear velocity: `v = k_linear * error_z` (move forward/back)
   - Angular velocity: `ω = -k_angular * error_x` (rotate left/right)
4. **Send Commands**: Publish to `/cmd_vel` with velocity limits
5. **Check Convergence**: Success if `|error_x| < tolerance` AND `|error_z| < tolerance`
6. **Repeat**: Continue until converged or timeout

## Error Handling

The action server handles various failure scenarios:

- **Drink Not Detected**: Retries up to `max_detection_attempts` times
- **API Timeout**: Returns failure with clear error message
- **Camera Unavailable**: Waits briefly for camera data, fails if unavailable
- **Overall Timeout**: Aborts operation after `overall_timeout` seconds
- **Goal Cancellation**: Stops robot and returns cancelled status

## Tuning Guide

### Improving Detection Accuracy
1. Ensure good lighting conditions
2. Use specific drink names (e.g., "red Coca Cola can" vs "drink")
3. Adjust camera position for clear view
4. Increase `max_detection_attempts` if environment is cluttered

### Improving Positioning Performance
1. **Faster Convergence**: Increase `k_linear` and `k_angular` gains
2. **Smoother Motion**: Decrease gains, reduce `max_linear_vel` and `max_angular_vel`
3. **Reduce Oscillations**: Increase `min_movement_threshold`
4. **Tighter Tolerance**: Decrease `position_tolerance` (requires more precise control)

### Reducing API Costs
1. Decrease `control_loop_rate` (fewer API calls per second)
2. Increase `movement_settle_time` (allow more time between detections)
3. Implement caching or tracking (future enhancement)

## Troubleshooting

### "Camera data not available"
- Check camera is publishing: `ros2 topic list | grep camera`
- Verify topic names match parameters
- Check camera node is running

### "Failed to detect drink"
- Ensure drink is visible in camera view
- Try more specific drink descriptions
- Check lighting conditions
- Verify GOOGLE_API_KEY is set correctly

### Robot moves erratically
- Reduce control gains (`k_linear`, `k_angular`)
- Decrease velocity limits
- Increase `movement_settle_time`
- Check for controller conflicts (other nodes publishing to `/cmd_vel`)

### "Timeout after X seconds"
- Increase `overall_timeout`
- Increase `control_loop_rate` for faster convergence
- Check if robot is physically able to reach target position

## Integration with Wilson

This package is designed to work with the Wilson robot system. To integrate:

1. Add to Wilson's main launch file:
   ```python
   locate_drink_server = IncludeLaunchDescription(
       PythonLaunchDescriptionSource([
           os.path.join(get_package_share_directory('locate_drink_action'), 'launch', 'locate_drink_server.launch.py')
       ]),
       launch_arguments={'use_sim_time': use_sim_time}.items()
   )
   ```

2. Call from Gemini AI tools:
   ```python
   def locate_drink_tool(self, drink_name):
       """Tool for Gemini to locate drinks."""
       goal = LocateDrink.Goal()
       goal.drink_name = drink_name

       future = self.locate_drink_client.send_goal_async(goal)
       # Handle result...
   ```

## Future Enhancements

Potential improvements:
- [ ] Object tracking to reduce API calls
- [ ] Multiple drink detection and selection
- [ ] Integration with MoveIt for arm reaching
- [ ] Visual servoing for finer positioning
- [ ] Obstacle avoidance during positioning
- [ ] Dynamic reconfiguration of parameters
- [ ] Support for different camera types/calibrations

## License

MIT License

## Author

Created for the Wilson autonomous robot project.

## References

- [ROS2 Actions Documentation](https://docs.ros.org/en/humble/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html)
- [Google Gemini API](https://ai.google.dev/docs)
- [Wilson Robot Documentation](../README.md)
