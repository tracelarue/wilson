#!/usr/bin/env python3
"""
Initial Pose Publisher for Wilson Robot

This node publishes an initial pose estimate to the /initialpose topic
to help AMCL localization start with a known position in simulation.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import math


class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__('initial_pose_publisher')
        
        # Declare parameters
        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 0.0)
        self.declare_parameter('initial_pose_yaw', 0.0)
        
        # Get parameters
        self.x = self.get_parameter('initial_pose_x').get_parameter_value().double_value
        self.y = self.get_parameter('initial_pose_y').get_parameter_value().double_value
        self.yaw = self.get_parameter('initial_pose_yaw').get_parameter_value().double_value
        
        # Create publisher
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/initialpose', 
            10
        )
        
        # Create timer to publish initial pose after a delay
        self.timer = self.create_timer(2.0, self.publish_initial_pose)
        self.published = False
        
        self.get_logger().info(
            f'Initial pose publisher ready - will publish pose: '
            f'x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.2f}'
        )
    
    def publish_initial_pose(self):
        """Publish the initial pose estimate once."""
        if self.published:
            return
            
        # Create pose message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        # Set position
        pose_msg.pose.pose.position.x = self.x
        pose_msg.pose.pose.position.y = self.y
        pose_msg.pose.pose.position.z = 0.0
        
        # Set orientation from yaw angle
        pose_msg.pose.pose.orientation.x = 0.0
        pose_msg.pose.pose.orientation.y = 0.0
        pose_msg.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        pose_msg.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        
        # Set covariance matrix (uncertainty)
        # Format: [x, y, z, rotation about X, rotation about Y, rotation about Z]
        pose_msg.pose.covariance = [
            0.25, 0.0,  0.0, 0.0, 0.0, 0.0,    # x variance = 0.25
            0.0,  0.25, 0.0, 0.0, 0.0, 0.0,    # y variance = 0.25  
            0.0,  0.0,  0.0, 0.0, 0.0, 0.0,    # z variance = 0.0 (2D)
            0.0,  0.0,  0.0, 0.0, 0.0, 0.0,    # roll variance = 0.0 (2D)
            0.0,  0.0,  0.0, 0.0, 0.0, 0.0,    # pitch variance = 0.0 (2D)
            0.0,  0.0,  0.0, 0.0, 0.0, 0.068   # yaw variance = 0.068 (~15 degrees)
        ]
        
        # Publish the pose
        self.pose_pub.publish(pose_msg)
        self.published = True
        
        self.get_logger().info(
            f'Published initial pose: x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.2f}'
        )
        
        # Destroy timer after publishing
        self.timer.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = InitialPosePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()