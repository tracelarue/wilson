#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import MagneticField
import random
import time


class MLXTestPublisher(Node):
    """Test publisher node that publishes random magnetic field values to /mlx topic."""

    def __init__(self):
        super().__init__('mlx_test_publisher')

        # Create publisher
        self.publisher = self.create_publisher(MagneticField, '/mlx', 10)

        # Publish rate (Hz)
        self.publish_rate = 100.0
        timer_period = 1.0 / self.publish_rate

        # Create timer
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Value ranges
        self.min_val = -1.0
        self.max_val = 5.0

        self.get_logger().info(f'MLX Test Publisher started, publishing at {self.publish_rate} Hz')
        self.get_logger().info(f'Random values range: [{self.min_val}, {self.max_val}] µT')

    def timer_callback(self):
        """Publish random magnetic field data."""
        msg = MagneticField()

        # Set header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'mlx_sensor'

        # Generate random magnetic field values
        msg.magnetic_field.x = random.uniform(self.min_val, self.max_val)
        msg.magnetic_field.y = random.uniform(self.min_val, self.max_val)
        msg.magnetic_field.z = random.uniform(self.min_val, self.max_val)

        # Set covariance (zeros for test data)
        msg.magnetic_field_covariance = [0.0] * 9

        # Publish
        self.publisher.publish(msg)


def main(args=None):
    """Main entry point for test publisher."""
    rclpy.init(args=args)

    node = MLXTestPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
