#!/usr/bin/env python3

import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from myaction.action import CountDown


class CountDownActionServer(Node):
    """A simple countdown action server that demonstrates long-running actions with feedback."""

    def __init__(self):
        super().__init__('countdown_action_server')
        self._action_server = ActionServer(
            self,
            CountDown,
            'countdown',
            self.execute_callback
        )
        self.get_logger().info('Countdown Action Server Started!')

    def execute_callback(self, goal_handle):
        """Execute the countdown goal."""
        self.get_logger().info(f'Executing countdown from {goal_handle.request.count_from}...')
        
        # Initialize feedback message
        feedback_msg = CountDown.Feedback()
        
        # Get the starting count
        start_count = goal_handle.request.count_from
        current_count = start_count
        
        # Countdown loop - runs for about 20-30 seconds depending on starting number
        while current_count > 0:
            # Note: In this simple example, we'll skip cancellation checking
            # In a production system, you would implement proper cancellation handling
            
            # Update feedback
            feedback_msg.current_count = current_count
            
            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Countdown: {current_count}')
            
            # Wait 1 second before next count
            time.sleep(1.0)
            current_count -= 1
        
        # Goal completed successfully
        goal_handle.succeed()
        
        # Create result
        result = CountDown.Result()
        result.success = True
        result.final_count = 0
        
        self.get_logger().info('Countdown completed successfully!')
        return result


def main(args=None):
    rclpy.init(args=args)
    
    countdown_action_server = CountDownActionServer()
    
    try:
        rclpy.spin(countdown_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        countdown_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()