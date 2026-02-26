#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from action_msgs.msg import GoalStatus

from navigate_to_location_action.action import NavigateToLocation
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import math


class NavigateToLocationServer(Node):
    """
    Action server that provides a simple interface to Nav2's navigate_to_pose.
    Takes (x, y, z) position and (qx, qy, qz, qw) quaternion in map frame.
    Returns clear success/failure messages that Gemini can understand.
    """

    def __init__(self):
        super().__init__('navigate_to_location_server')

        # Create callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        # Feedback throttling configuration
        self.feedback_rate = 1  # Hz
        self.last_feedback_time = {}  # Track last publish time per goal_handle

        # Create action server for our custom action
        self._action_server = ActionServer(
            self,
            NavigateToLocation,
            'navigate_to_location',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        # Create action client to Nav2
        self._nav_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose',
            callback_group=self.callback_group
        )

        # Wait for Nav2 action server
        self.get_logger().info('Waiting for navigate_to_pose action server...')
        if not self._nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('navigate_to_pose action server not available!')
        else:
            self.get_logger().info('navigate_to_pose action server is ready')

        self.get_logger().info('NavigateToLocation action server started')

    def goal_callback(self, goal_request):
        """Accept all incoming goals."""
        self.get_logger().info(f'Received goal: ({goal_request.x:.2f}, {goal_request.y:.2f})')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept all cancellation requests."""
        self.get_logger().info('Received cancellation request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the navigation action."""
        self.get_logger().info('Executing navigate_to_location...')

        # Get the goal
        goal = goal_handle.request

        # Create result
        result = NavigateToLocation.Result()

        try:
            # Create NavigateToPose goal
            nav_goal = NavigateToPose.Goal()
            nav_goal.pose = PoseStamped()
            nav_goal.pose.header.frame_id = 'map'
            nav_goal.pose.header.stamp = self.get_clock().now().to_msg()

            # Set position
            nav_goal.pose.pose.position.x = goal.x
            nav_goal.pose.pose.position.y = goal.y
            nav_goal.pose.pose.position.z = goal.z

            # Set orientation
            nav_goal.pose.pose.orientation.x = goal.qx
            nav_goal.pose.pose.orientation.y = goal.qy
            nav_goal.pose.pose.orientation.z = goal.qz
            nav_goal.pose.pose.orientation.w = goal.qw

            # Send goal to Nav2
            self.get_logger().info(f'Sending navigation goal to ({goal.x:.2f}, {goal.y:.2f})')
            send_goal_future = self._nav_client.send_goal_async(
                nav_goal,
                feedback_callback=lambda feedback_msg: self._nav_feedback_callback(
                    feedback_msg, goal_handle
                )
            )

            # Wait for goal to be accepted
            nav_goal_handle = await send_goal_future

            if not nav_goal_handle.accepted:
                self.get_logger().error('Navigation goal rejected by Nav2')
                result.success = False
                result.message = 'Navigation goal rejected'
                goal_handle.abort()
                return result

            self.get_logger().info('Navigation goal accepted by Nav2')

            # Wait for result (cancellation is handled by Nav2 internally)
            nav_result = await nav_goal_handle.get_result_async()

            # Map Nav2 status to our result
            status = nav_result.status

            if status == GoalStatus.STATUS_SUCCEEDED:
                result.success = True
                result.message = f'Successfully reached location at ({goal.x:.2f}, {goal.y:.2f})'
                self.get_logger().info(f'Navigation succeeded: {result.message}')
                goal_handle.succeed()

            elif status == GoalStatus.STATUS_CANCELED:
                result.success = False
                result.message = 'Navigation canceled'
                self.get_logger().info('Navigation was canceled')
                goal_handle.canceled()

            elif status == GoalStatus.STATUS_ABORTED:
                result.success = False
                result.message = f'Navigation failed - could not reach target location at ({goal.x:.2f}, {goal.y:.2f})'
                self.get_logger().warn(f'Navigation aborted: {result.message}')
                goal_handle.abort()

            else:
                result.success = False
                result.message = f'Navigation ended with unknown status: {status}'
                self.get_logger().error(result.message)
                goal_handle.abort()

        except Exception as e:
            self.get_logger().error(f'Navigation failed with exception: {str(e)}')
            result.success = False
            result.message = f'Navigation failed: {str(e)}'
            goal_handle.abort()
        finally:
            # Clean up feedback time tracking to prevent memory leaks
            goal_id = id(goal_handle)
            if goal_id in self.last_feedback_time:
                del self.last_feedback_time[goal_id]

        return result

    def _nav_feedback_callback(self, feedback_msg, goal_handle):
        """Relay Nav2 feedback to our action clients with throttling."""
        # Get current time
        current_time = self.get_clock().now()

        # Get goal_handle ID (use id() as unique identifier)
        goal_id = id(goal_handle)

        # Check if enough time has elapsed since last feedback
        if goal_id in self.last_feedback_time:
            time_since_last = (current_time - self.last_feedback_time[goal_id]).nanoseconds / 1e9
            min_interval = 1.0 / self.feedback_rate  # 2.0 seconds for 0.5 Hz

            # Skip if not enough time has passed
            if time_since_last < min_interval:
                return

        # Update last feedback time
        self.last_feedback_time[goal_id] = current_time

        nav_feedback = feedback_msg.feedback

        # Calculate distance to goal
        current_pose = nav_feedback.current_pose.pose
        distance = math.sqrt(
            (goal_handle.request.x - current_pose.position.x) ** 2 +
            (goal_handle.request.y - current_pose.position.y) ** 2
        )

        # Publish our own feedback
        feedback = NavigateToLocation.Feedback()
        feedback.status = f'Navigating to ({goal_handle.request.x:.2f}, {goal_handle.request.y:.2f})'
        feedback.distance_remaining = distance

        goal_handle.publish_feedback(feedback)


def main(args=None):
    rclpy.init(args=args)

    action_server = NavigateToLocationServer()

    # Use a MultiThreadedExecutor to enable concurrent execution
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
