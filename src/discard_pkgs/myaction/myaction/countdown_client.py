#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from myaction.action import CountDown


class CountDownActionClient(Node):
    """A simple countdown action client for testing the countdown server."""

    def __init__(self):
        super().__init__('countdown_action_client')
        self._action_client = ActionClient(self, CountDown, 'countdown')

    def send_goal(self, count_from):
        """Send a countdown goal to the action server."""
        goal_msg = CountDown.Goal()
        goal_msg.count_from = count_from

        self.get_logger().info(f'Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info(f'Sending goal: count down from {count_from}')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle the goal response from the server."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle the final result from the server."""
        result = future.result().result
        if result.success:
            self.get_logger().info(
                f'Result: SUCCESS! Final count: {result.final_count}'
            )
        else:
            self.get_logger().info(
                f'Result: FAILED. Final count: {result.final_count}'
            )
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        """Handle feedback messages from the server."""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback: Current count: {feedback.current_count}')


def main(args=None):
    rclpy.init(args=args)

    action_client = CountDownActionClient()

    # Count down from 25 (will take about 25 seconds)
    action_client.send_goal(25)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()


if __name__ == '__main__':
    main()