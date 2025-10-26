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
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge


class LocateDrinkActionServer(Node):
    """
    Action server that locates a specified drink using Gemini AI and
    positions the robot optimally using differential drive control.
    """

    def __init__(self):
        super().__init__('locate_drink_action_server')

        # Load environment variables
        # Try multiple .env file locations
        env_paths = ['/wilson/.env', '/home/trace/wilson/.env', '.env']
        env_loaded = False
        for env_path in env_paths:
            if os.path.exists(env_path):
                load_dotenv(dotenv_path=env_path)
                self.get_logger().info(f'Loaded environment from {env_path}')
                env_loaded = True
                break
        if not env_loaded:
            self.get_logger().warn('No .env file found in standard locations')

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

        # QoS profile for visualization markers (compatible with RViz)
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscribe to camera topics
        self.rgb_subscription = self.create_subscription(
            Image,
            '/rgb_camera/image_raw',
            self.rgb_image_callback,
            image_qos,
            callback_group=self.callback_group
        )

        self.depth_subscription = self.create_subscription(
            Image,
            '/depth_camera/depth/image_raw',
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

        # Publisher for visualization markers
        self.marker_publisher = self.create_publisher(
            Marker,
            '/drink_marker',
            marker_qos
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
        self.declare_parameter('target_z', 0.65)
        self.declare_parameter('position_tolerance', 0.05)

        # Control parameters
        self.declare_parameter('k_linear', 0.3)
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
        self.declare_parameter('depth_sample_size', 5)  # NxN window for depth averaging
        self.declare_parameter('bbox_shift_ratio', 0.0)  # Horizontal shift ratio for depth sampling

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
        self.depth_sample_size = self.get_parameter('depth_sample_size').value
        self.bbox_shift_ratio = self.get_parameter('bbox_shift_ratio').value
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
        self.get_logger().info(f'Received goal to locate drink: {goal_request.drinkname}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation requests."""
        self.get_logger().info('Goal cancellation requested')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        Execute the locate drink action.

        Uses closed-loop control to position the robot optimally relative to the detected drink.
        """
        self.get_logger().info(f'Executing goal: Locate {goal_handle.request.drinkname}')

        # Initialize result
        result = LocateDrink.Result()
        result.success = False
        result.message = ''
        result.final_position = Point(x=0.0, y=0.0, z=0.0)

        # Initialize feedback
        feedback = LocateDrink.Feedback()
        feedback.detection_attempts = 0

        # Wait for camera data
        if not self._wait_for_camera_data(timeout=5.0):
            result.message = 'Camera data not available'
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result

        # Start timing for overall timeout
        start_time = time.time()

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
                feedback.current_status = f'Detecting {goal_handle.request.drinkname}... (attempt {feedback.detection_attempts})'

                detection_result = self._detect_and_localize_drink(goal_handle.request.drinkname)

                if detection_result is None:
                    # Detection failed
                    if feedback.detection_attempts >= self.max_detection_attempts:
                        result.message = f'Failed to detect {goal_handle.request.drinkname} after {feedback.detection_attempts} attempts'
                        self.get_logger().error(result.message)
                        self._stop_robot()
                        break

                    # Retry detection
                    feedback.current_status = f'Drink not detected, retrying...'
                    goal_handle.publish_feedback(feedback)
                    loop_rate.sleep()
                    continue

                # Successfully detected drink
                current_x, current_y, current_z = detection_result

                # Add can diameter offset to z position (0.033m = 33mm can diameter)
                # This accounts for the distance from the can center to its front edge
                current_z_adjusted = current_z + 0.033

                # Calculate errors using adjusted z position
                error_x = current_x - self.target_x
                error_z = current_z_adjusted - self.target_z

                # Update feedback with adjusted position
                feedback.current_position = Point(x=current_x, y=current_y, z=current_z_adjusted)
                feedback.error_x = error_x
                feedback.error_z = error_z
                feedback.current_status = f'Drink located at x={current_x:.3f}m, y={current_y:.3f}m, z={current_z_adjusted:.3f}m (adjusted)'
                goal_handle.publish_feedback(feedback)

                self.get_logger().info(
                    f'Position: x={current_x:.3f}m, y={current_y:.3f}m, z={current_z:.3f}m (raw), z_adj={current_z_adjusted:.3f}m | '
                    f'Errors: x={error_x:.3f}m, z={error_z:.3f}m'
                )

                # Publish RViz marker with adjusted z position
                self._publish_marker(current_x, current_y, current_z_adjusted, goal_handle.request.drinkname)

                # Check if within tolerance
                if abs(error_x) < self.position_tolerance and abs(error_z) < self.position_tolerance:
                    result.success = True
                    result.message = f'Successfully positioned relative to {goal_handle.request.drinkname}'
                    result.final_position = Point(x=current_x, y=current_y, z=current_z_adjusted)
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
            import traceback
            self.get_logger().error(traceback.format_exc())
            self._stop_robot()
            goal_handle.abort()
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
            Tuple of (x, y, z) position in meters, or None if detection failed
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

            # Store original dimensions
            original_width = cv_rgb_image.shape[1]
            original_height = cv_rgb_image.shape[0]

            # Thumbnail and store new dimensions
            img.thumbnail([1024, 1024])
            thumbnail_width, thumbnail_height = img.size

            self.get_logger().info(f'Original image: {original_width}x{original_height}')
            self.get_logger().info(f'Thumbnail image: {thumbnail_width}x{thumbnail_height}')

            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)
            image_bytes = image_io.read()

            # Prepare prompt for Gemini do not un-tab
            prompt = f"""Find the {drink_name} in this image.
                Return a JSON object with a bounding box in this exact format:
                [{{"box_2d": [ymin, xmin, ymax, xmax]}}]

                The coordinates should be normalized to 0-1000.
                If you cannot find the {drink_name}, return an empty array: []"""

            # Call Gemini API
            self.get_logger().info(f'Calling Gemini API to detect {drink_name}')
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash-lite',
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
                self.get_logger().warn('No JSON found in Gemini response')
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

            # Convert normalized coordinates (0-1000) to pixel coordinates in THUMBNAIL space
            thumb_y1 = box_2d[0] / 1000 * thumbnail_height
            thumb_x1 = box_2d[1] / 1000 * thumbnail_width
            thumb_y2 = box_2d[2] / 1000 * thumbnail_height
            thumb_x2 = box_2d[3] / 1000 * thumbnail_width

            self.get_logger().info(f'Thumbnail bbox: ({thumb_x1:.1f}, {thumb_y1:.1f}) -> ({thumb_x2:.1f}, {thumb_y2:.1f})')

            # Scale from thumbnail space to ORIGINAL image space
            scale_x = original_width / thumbnail_width
            scale_y = original_height / thumbnail_height

            norm_y1 = int(thumb_y1 * scale_y)
            norm_x1 = int(thumb_x1 * scale_x)
            norm_y2 = int(thumb_y2 * scale_y)
            norm_x2 = int(thumb_x2 * scale_x)

            self.get_logger().info(f'Original bbox: ({norm_x1}, {norm_y1}) -> ({norm_x2}, {norm_y2})')
            self.get_logger().info(f'Scale factors: x={scale_x:.3f}, y={scale_y:.3f}')

            # Calculate RGB bounding box dimensions
            bbox_width = norm_x2 - norm_x1
            bbox_height = norm_y2 - norm_y1

            self.get_logger().info(f'RGB bbox dimensions: width={bbox_width}px, height={bbox_height}px')

            # Calculate horizontal shift for depth image based on bbox height
            # Positive shift = move to the right (compensates for RGB/depth camera offset)
            horizontal_shift = int(self.bbox_shift_ratio * bbox_height)

            self.get_logger().info(f'Calculated horizontal shift for depth: {horizontal_shift}px (ratio={self.bbox_shift_ratio})')

            # Calculate center of RGB bounding box
            center_y = (norm_y1 + norm_y2) // 2
            center_x = (norm_x1 + norm_x2) // 2

            self.get_logger().info(f'RGB bbox center: ({center_x}, {center_y})')

            # Apply shift to get depth sampling position
            depth_center_x = center_x + horizontal_shift
            depth_center_y = center_y  # Y stays the same

            # Calculate shifted bounding box for depth image
            depth_x1 = norm_x1 + horizontal_shift
            depth_x2 = norm_x2 + horizontal_shift
            depth_y1 = norm_y1
            depth_y2 = norm_y2

            self.get_logger().info(f'Depth sampling center (shifted): ({depth_center_x}, {depth_center_y})')
            self.get_logger().info(f'Depth bbox (shifted): ({depth_x1}, {depth_y1}) -> ({depth_x2}, {depth_y2})')

            # Get depth image
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
            img_height, img_width = cv_depth_image.shape

            self.get_logger().info(f'Depth image dimensions: {img_width}x{img_height}')

            # Scale coordinates from RGB image space to depth image space
            # RGB image is {original_width}x{original_height}, depth is {img_width}x{img_height}
            depth_scale_x = img_width / original_width
            depth_scale_y = img_height / original_height

            self.get_logger().info(f'Depth scaling factors: x={depth_scale_x:.4f}, y={depth_scale_y:.4f}')

            # Scale center coordinates to depth image space
            depth_center_x_scaled = int(depth_center_x * depth_scale_x)
            depth_center_y_scaled = int(depth_center_y * depth_scale_y)

            # Scale bbox corners to depth image space for visualization
            depth_x1_scaled = int(depth_x1 * depth_scale_x)
            depth_x2_scaled = int(depth_x2 * depth_scale_x)
            depth_y1_scaled = int(depth_y1 * depth_scale_y)
            depth_y2_scaled = int(depth_y2 * depth_scale_y)

            # Scale RGB center for visualization on depth image
            center_x_scaled = int(center_x * depth_scale_x)
            center_y_scaled = int(center_y * depth_scale_y)

            # Scale RGB bbox for visualization on depth image
            norm_x1_scaled = int(norm_x1 * depth_scale_x)
            norm_x2_scaled = int(norm_x2 * depth_scale_x)
            norm_y1_scaled = int(norm_y1 * depth_scale_y)
            norm_y2_scaled = int(norm_y2 * depth_scale_y)

            self.get_logger().info(f'Scaled depth center: ({depth_center_x_scaled}, {depth_center_y_scaled}) [was ({depth_center_x}, {depth_center_y}) in RGB space]')

            # Calculate depth as average of NxN window centered at SCALED depth_center
            half_window = self.depth_sample_size // 2

            # Calculate window bounds using SCALED coordinates
            y_min = max(0, depth_center_y_scaled - half_window)
            y_max = min(img_height, depth_center_y_scaled + half_window + 1)
            x_min = max(0, depth_center_x_scaled - half_window)
            x_max = min(img_width, depth_center_x_scaled + half_window + 1)

            # Ensure window is within image bounds
            if y_min >= img_height or y_max <= 0 or x_min >= img_width or x_max <= 0:
                self.get_logger().error(f'Depth sampling window out of bounds! Center: ({depth_center_x_scaled}, {depth_center_y_scaled}), Image: {img_width}x{img_height}')
                return None

            # Extract depth window and calculate average (filtering out zeros/invalid)
            depth_window = cv_depth_image[y_min:y_max, x_min:x_max]
            valid_depths = depth_window[depth_window > 0]

            if len(valid_depths) == 0:
                self.get_logger().warn('No valid depth readings in sampling window')
                return None

            depth = float(np.mean(valid_depths))
            depth_std = float(np.std(valid_depths))

            self.get_logger().info(f'Depth sampling: scaled_center=({depth_center_x_scaled}, {depth_center_y_scaled}), window={self.depth_sample_size}x{self.depth_sample_size}, samples={len(valid_depths)}')
            self.get_logger().info(f'Depth average: {depth:.3f}m (std: {depth_std:.3f}m)')

            # Calculate 3D position using the ORIGINAL RGB-space depth center position
            # This ensures the 3D calculation uses the correct FOV math based on RGB image dimensions
            # The depth value comes from the scaled coordinates, but the angle calculation needs RGB space
            x_3d, y_3d, z_3d = self._calculate_3d_position(depth_center_x, depth_center_y, depth)

            # === VISUALIZATION DEBUG ===
            # Visualization disabled for production use
            # Uncomment the code below to enable debug visualization windows

            # # Display 3 debug images in separate windows
            # # Window 1: Gemini input image (thumbnailed)
            # gemini_display_img = np.array(img)
            # gemini_bgr = cv2.cvtColor(gemini_display_img, cv2.COLOR_RGB2BGR)
            #
            # # Window 2: Original RGB image with bounding box
            # rgb_with_bbox = cv_rgb_image.copy()
            # cv2.rectangle(rgb_with_bbox, (norm_x1, norm_y1), (norm_x2, norm_y2), (0, 255, 0), 3)
            # cv2.circle(rgb_with_bbox, (center_x, center_y), 8, (255, 0, 0), -1)
            # # Add text with position info
            # text = f"RGB Center: ({center_x}, {center_y})"
            # cv2.putText(rgb_with_bbox, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # text2 = f"Shift: {horizontal_shift}px ({self.bbox_shift_ratio} * height={bbox_height})"
            # cv2.putText(rgb_with_bbox, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # text3 = f"BBox: ({norm_x1},{norm_y1}) -> ({norm_x2},{norm_y2})"
            # cv2.putText(rgb_with_bbox, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #
            # # Window 3: Depth image with BOTH bounding boxes (SCALED to depth image resolution)
            # # Normalize depth image for visualization
            # depth_normalized = cv2.normalize(cv_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            #
            # # Draw original RGB bbox position in RED (for reference) - SCALED to depth image
            # cv2.rectangle(depth_colored, (norm_x1_scaled, norm_y1_scaled), (norm_x2_scaled, norm_y2_scaled), (0, 0, 255), 2)
            # cv2.circle(depth_colored, (center_x_scaled, center_y_scaled), 4, (0, 0, 255), -1)
            #
            # # Draw shifted depth bbox in GREEN (actual sampling position) - SCALED to depth image
            # cv2.rectangle(depth_colored, (depth_x1_scaled, depth_y1_scaled), (depth_x2_scaled, depth_y2_scaled), (0, 255, 0), 3)
            # cv2.circle(depth_colored, (depth_center_x_scaled, depth_center_y_scaled), 5, (255, 255, 255), -1)
            #
            # # Add text
            # text_depth = f"Depth Center: ({depth_center_x_scaled}, {depth_center_y_scaled}) = {depth:.3f}m"
            # cv2.putText(depth_colored, text_depth, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # text_scale = f"Scale: {depth_scale_x:.3f}x, {depth_scale_y:.3f}y"
            # cv2.putText(depth_colored, text_scale, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # text_3d = f"3D: x={x_3d:.3f}m, z={z_3d:.3f}m"
            # cv2.putText(depth_colored, text_3d, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # text_legend = "RED=RGB position, GREEN=Shifted depth sampling"
            # cv2.putText(depth_colored, text_legend, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            #
            # # Create named windows and display
            # cv2.namedWindow('1. Gemini Input', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('2. RGB with BBox', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('3. Depth with BBox', cv2.WINDOW_NORMAL)
            #
            # cv2.imshow('1. Gemini Input', gemini_bgr)
            # cv2.imshow('2. RGB with BBox', rgb_with_bbox)
            # cv2.imshow('3. Depth with BBox', depth_colored)
            #
            # self.get_logger().info('=== DEBUG WINDOWS DISPLAYED ===')
            # self.get_logger().info('Press any key on one of the windows to continue...')
            # cv2.waitKey(0)  # Wait indefinitely for key press
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)  # Small delay to ensure windows close properly
            # === END VISUALIZATION ===

            return (x_3d, y_3d, z_3d)

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
            depth: Depth in meters from camera optical frame

        Returns:
            Tuple of (x, y, z) in meters relative to camera optical frame
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

        return (x, y, z)

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

    def _publish_marker(self, x, y, z, drink_name):
        """
        Publish a visualization marker for the detected drink.

        Args:
            x: X position in camera frame (meters)
            y: Y position in camera frame (meters)
            z: Z position in camera frame (meters)
            drink_name: Name of the drink for the marker text
        """
        self.get_logger().info(f'_publish_marker called with x={x:.3f}, y={y:.3f}, z={z:.3f}, drink_name={drink_name}')

        marker = Marker()
        marker.header.frame_id = self.camera_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "drink_detection"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        # Set scale (10cm sphere)
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Set color (bright green)
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

        # Set lifetime
        marker.lifetime.sec = 30  # 30 seconds

        self.get_logger().info(f'Publishing marker on topic /drink_marker with frame_id={self.camera_frame}')
        self.marker_publisher.publish(marker)
        self.get_logger().info(f'Marker published successfully for {drink_name} at x={x:.3f}, y={y:.3f}, z={z:.3f}')


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
