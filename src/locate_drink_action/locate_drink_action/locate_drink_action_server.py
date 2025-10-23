#!/usr/bin/env python3

"""
Locate Drink Action Server

Uses Gemini AI for drink detection and positions the robot optimally
relative to the detected drink using differential drive control.
"""

import os
import io
import json
import math
import time
import threading
import cv2
import PIL.Image
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from locate_drink_action.action import LocateDrink
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Point
from cv_bridge import CvBridge


class LocateDrinkActionServer(Node):
    """
    Action server that locates a specified drink using Gemini AI and
    positions the robot optimally using differential drive control.
    """

    def __init__(self):
        super().__init__('locate_drink_action_server')

        # Load environment variables
        env_path = '/home/trace/wilson/.env'
        load_dotenv(dotenv_path=env_path)
        self.api_key = os.getenv('GOOGLE_API_KEY')

        if not self.api_key:
            self.get_logger().error('GOOGLE_API_KEY not found in environment!')
            raise ValueError('GOOGLE_API_KEY not set')

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        self._load_parameters()

        # Initialize Gemini client
        self.gemini_client = genai.Client(api_key=self.api_key)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Image data
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.image_lock = threading.Lock()

        # Create callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        # QoS profile for camera subscriptions
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # Subscribe to camera topics
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_image_callback,
            image_qos,
            callback_group=self.callback_group
        )

        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_image_callback,
            image_qos,
            callback_group=self.callback_group
        )

        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Action server
        self._action_server = ActionServer(
            self,
            LocateDrink,
            'locate_drink',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info('Locate Drink Action Server initialized')

    def _declare_parameters(self):
        """Declare all ROS parameters with default values."""
        # Target position parameters
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_z', 0.4)
        self.declare_parameter('position_tolerance', 0.03)

        # Control parameters
        self.declare_parameter('k_linear', 0.5)
        self.declare_parameter('k_angular', 1.0)
        self.declare_parameter('max_linear_vel', 0.3)
        self.declare_parameter('max_angular_vel', 0.5)
        self.declare_parameter('min_movement_threshold', 0.005)

        # Camera parameters
        self.declare_parameter('h_fov', 1.089)  # Horizontal field of view in radians
        self.declare_parameter('aspect_ratio', 4.0/3.0)
        self.declare_parameter('camera_frame', 'depth_camera_link_optical')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        # Operation parameters
        self.declare_parameter('max_detection_attempts', 5)
        self.declare_parameter('control_loop_rate', 2.0)  # Hz
        self.declare_parameter('overall_timeout', 60.0)  # seconds
        self.declare_parameter('api_timeout', 10.0)  # seconds
        self.declare_parameter('movement_settle_time', 0.5)  # seconds

    def _load_parameters(self):
        """Load all parameters from ROS parameter server."""
        # Target position
        self.target_x = self.get_parameter('target_x').value
        self.target_z = self.get_parameter('target_z').value
        self.position_tolerance = self.get_parameter('position_tolerance').value

        # Control gains
        self.k_linear = self.get_parameter('k_linear').value
        self.k_angular = self.get_parameter('k_angular').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.min_movement_threshold = self.get_parameter('min_movement_threshold').value

        # Camera parameters
        self.h_fov = self.get_parameter('h_fov').value
        self.aspect_ratio = self.get_parameter('aspect_ratio').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.v_fov = self.h_fov / self.aspect_ratio

        # Operation parameters
        self.max_detection_attempts = self.get_parameter('max_detection_attempts').value
        self.control_loop_rate = self.get_parameter('control_loop_rate').value
        self.overall_timeout = self.get_parameter('overall_timeout').value
        self.api_timeout = self.get_parameter('api_timeout').value
        self.movement_settle_time = self.get_parameter('movement_settle_time').value

    def rgb_image_callback(self, msg):
        """Callback for RGB camera images."""
        with self.image_lock:
            self.latest_rgb_image = msg

    def depth_image_callback(self, msg):
        """Callback for depth camera images."""
        with self.image_lock:
            self.latest_depth_image = msg

    def goal_callback(self, goal_request):
        """Handle new goal requests."""
        self.get_logger().info(f'Received goal to locate drink: {goal_request.drink_name}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation requests."""
        self.get_logger().info('Goal cancellation requested')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        Execute the locate drink action.

        Main control loop:
        1. Detect drink using Gemini AI
        2. Calculate 3D position
        3. Calculate positioning error
        4. Send velocity commands if error exceeds tolerance
        5. Repeat until positioned or timeout
        """
        self.get_logger().info(f'Executing goal: Locate {goal_handle.request.drink_name}')

        # Initialize result
        result = LocateDrink.Result()
        result.success = False
        result.message = ''
        result.final_position = Point(x=0.0, y=0.0, z=0.0)

        # Initialize feedback
        feedback = LocateDrink.Feedback()
        feedback.detection_attempts = 0

        # Start timing
        start_time = time.time()

        # Wait for camera data
        if not self._wait_for_camera_data(timeout=5.0):
            result.message = 'Camera data not available'
            self.get_logger().error(result.message)
            return result

        # Main control loop
        loop_rate = self.create_rate(self.control_loop_rate)

        try:
            while rclpy.ok():
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > self.overall_timeout:
                    result.message = f'Timeout after {elapsed_time:.1f} seconds'
                    self.get_logger().warn(result.message)
                    self._stop_robot()
                    break

                # Check if goal was cancelled
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.message = 'Goal cancelled by client'
                    self.get_logger().info(result.message)
                    self._stop_robot()
                    return result

                # Attempt to detect and localize the drink
                feedback.detection_attempts += 1
                feedback.current_status = f'Detecting {goal_handle.request.drink_name}... (attempt {feedback.detection_attempts})'

                detection_result = self._detect_and_localize_drink(goal_handle.request.drink_name)

                if detection_result is None:
                    # Detection failed
                    if feedback.detection_attempts >= self.max_detection_attempts:
                        result.message = f'Failed to detect {goal_handle.request.drink_name} after {feedback.detection_attempts} attempts'
                        self.get_logger().error(result.message)
                        self._stop_robot()
                        break

                    # Retry detection
                    feedback.current_status = f'Drink not detected, retrying...'
                    goal_handle.publish_feedback(feedback)
                    loop_rate.sleep()
                    continue

                # Successfully detected drink
                current_x, current_z = detection_result

                # Calculate errors
                error_x = current_x - self.target_x
                error_z = current_z - self.target_z

                # Update feedback
                feedback.current_position = Point(x=current_x, y=0.0, z=current_z)
                feedback.error_x = error_x
                feedback.error_z = error_z
                feedback.current_status = f'Drink located at x={current_x:.3f}m, z={current_z:.3f}m'
                goal_handle.publish_feedback(feedback)

                self.get_logger().info(
                    f'Position: x={current_x:.3f}m, z={current_z:.3f}m | '
                    f'Errors: x={error_x:.3f}m, z={error_z:.3f}m'
                )

                # Check if within tolerance
                if abs(error_x) < self.position_tolerance and abs(error_z) < self.position_tolerance:
                    result.success = True
                    result.message = f'Successfully positioned relative to {goal_handle.request.drink_name}'
                    result.final_position = Point(x=current_x, y=0.0, z=current_z)
                    self.get_logger().info(result.message)
                    self._stop_robot()
                    break

                # Calculate and send velocity commands
                self._send_velocity_commands(error_x, error_z)

                # Wait for movement to settle
                time.sleep(self.movement_settle_time)
                loop_rate.sleep()

        except Exception as e:
            result.message = f'Exception during execution: {str(e)}'
            self.get_logger().error(result.message)
            self._stop_robot()
            return result

        # Set goal status
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    def _wait_for_camera_data(self, timeout=5.0):
        """
        Wait for camera data to become available.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if camera data is available, False otherwise
        """
        start_time = time.time()
        rate = self.create_rate(10)  # 10 Hz check rate

        while rclpy.ok() and (time.time() - start_time) < timeout:
            with self.image_lock:
                if self.latest_rgb_image is not None and self.latest_depth_image is not None:
                    self.get_logger().info('Camera data available')
                    return True
            rate.sleep()

        self.get_logger().error('Camera data not available within timeout')
        return False

    def _detect_and_localize_drink(self, drink_name):
        """
        Detect the specified drink and calculate its 3D position.

        Args:
            drink_name: Name of the drink to detect

        Returns:
            Tuple of (x, z) position in meters, or None if detection failed
        """
        try:
            # Get current images
            with self.image_lock:
                if self.latest_rgb_image is None or self.latest_depth_image is None:
                    self.get_logger().warn('No image data available')
                    return None
                rgb_image = self.latest_rgb_image
                depth_image = self.latest_depth_image

            # Convert RGB image for Gemini
            cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb_image, "bgr8")
            frame_rgb = cv2.cvtColor(cv_rgb_image, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)
            img.thumbnail([1024, 1024])
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)
            image_bytes = image_io.read()

            # Prepare prompt for Gemini
            prompt = f"""Find the {drink_name} in this image.
Return a JSON object with a bounding box in this exact format:
[{{"box_2d": [ymin, xmin, ymax, xmax]}}]

The coordinates should be normalized to 0-1000.
If you cannot find the {drink_name}, return an empty array: []"""

            # Call Gemini API
            self.get_logger().info(f'Calling Gemini API to detect {drink_name}')
            response = self.gemini_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt
                ]
            )

            # Parse response
            response_text = response.text
            self.get_logger().debug(f'Gemini response: {response_text}')

            # Extract JSON from response (may be wrapped in markdown)
            json_output = self._extract_json(response_text)

            if not json_output:
                self.get_logger().warn(f'No JSON found in Gemini response')
                return None

            bounding_boxes = json.loads(json_output)

            if not bounding_boxes or len(bounding_boxes) == 0:
                self.get_logger().warn(f'Drink "{drink_name}" not detected in image')
                return None

            # Get first bounding box
            bounding_box = bounding_boxes[0]
            box_2d = bounding_box.get('box_2d')

            if not box_2d or len(box_2d) != 4:
                self.get_logger().error(f'Invalid bounding box format: {box_2d}')
                return None

            # Convert normalized coordinates to pixel coordinates
            norm_y1 = int(box_2d[0] / 1000 * self.image_height)
            norm_x1 = int(box_2d[1] / 1000 * self.image_width)
            norm_y2 = int(box_2d[2] / 1000 * self.image_height)
            norm_x2 = int(box_2d[3] / 1000 * self.image_width)

            # Calculate center of bounding box
            center_y = (norm_y1 + norm_y2) // 2
            center_x = (norm_x1 + norm_x2) // 2

            self.get_logger().info(f'Bounding box center: ({center_y}, {center_x})')

            # Get depth at center
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
            depth = float(cv_depth_image[center_y, center_x])

            self.get_logger().info(f'Depth at center: {depth:.3f}m')

            # Calculate 3D position
            x_3d, z_3d = self._calculate_3d_position(center_x, center_y, depth)

            return (x_3d, z_3d)

        except Exception as e:
            self.get_logger().error(f'Error in drink detection: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None

    def _extract_json(self, text):
        """
        Extract JSON from text that may contain markdown code blocks.

        Args:
            text: Input text potentially containing JSON

        Returns:
            JSON string or empty string if not found
        """
        lines = text.splitlines()
        json_started = False
        json_lines = []

        for line in lines:
            if line.strip() == "```json":
                json_started = True
                continue
            elif line.strip() == "```" and json_started:
                break
            elif json_started:
                json_lines.append(line)

        if json_lines:
            return "\n".join(json_lines)

        # If no markdown blocks, try to find JSON in the text
        # Look for array brackets
        try:
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                return text[start_idx:end_idx+1]
        except:
            pass

        return ""

    def _calculate_3d_position(self, pixel_x, pixel_y, depth):
        """
        Calculate 3D position from pixel coordinates and depth.

        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            depth: Depth in meters

        Returns:
            Tuple of (x, z) in meters relative to camera frame
        """
        # Calculate angle per pixel
        angle_per_pixel_x = self.h_fov / self.image_width
        angle_per_pixel_y = self.v_fov / self.image_height

        # Offset from image center
        x_offset = pixel_x - self.image_width / 2
        y_offset = pixel_y - self.image_height / 2

        # Calculate angles in radians
        x_ang = x_offset * angle_per_pixel_x
        y_ang = y_offset * angle_per_pixel_y

        # Project to 3D (depth is along camera Z axis)
        z = depth
        x = z * math.tan(x_ang)
        y = z * math.tan(y_ang)

        self.get_logger().debug(f'3D position: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m')

        return (x, z)

    def _send_velocity_commands(self, error_x, error_z):
        """
        Calculate and send velocity commands based on positioning errors.

        Args:
            error_x: Horizontal error in meters (positive = drink is to the right)
            error_z: Distance error in meters (positive = drink is too far)
        """
        # Proportional control
        # Linear velocity: move forward/backward to achieve target distance
        linear_vel = self.k_linear * error_z

        # Angular velocity: rotate to center the drink (negative because positive error means turn left)
        angular_vel = -self.k_angular * error_x

        # Apply velocity limits
        linear_vel = max(-self.max_linear_vel, min(self.max_linear_vel, linear_vel))
        angular_vel = max(-self.max_angular_vel, min(self.max_angular_vel, angular_vel))

        # Apply minimum movement threshold to prevent tiny oscillations
        if abs(linear_vel) < self.min_movement_threshold:
            linear_vel = 0.0
        if abs(angular_vel) < self.min_movement_threshold:
            angular_vel = 0.0

        # Create and publish Twist message
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel

        self.cmd_vel_publisher.publish(cmd_vel)

        self.get_logger().debug(f'Velocity command: linear={linear_vel:.3f}, angular={angular_vel:.3f}')

    def _stop_robot(self):
        """Send zero velocity command to stop the robot."""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)
        self.get_logger().info('Robot stopped')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        action_server = LocateDrinkActionServer()

        # Use MultiThreadedExecutor for concurrent callback execution
        executor = MultiThreadedExecutor()
        executor.add_node(action_server)

        try:
            executor.spin()
        except KeyboardInterrupt:
            action_server.get_logger().info('Keyboard interrupt, shutting down')
        finally:
            executor.shutdown()
            action_server.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
