#!/usr/bin/env python3

"""
Launch file for real robot cameras and locate drink action server.

This launch file starts:
1. Depth camera (Arducam ToF) driver
2. V4L2 USB camera for RGB images
3. Locate drink action server with real robot parameters

Usage:
    ros2 launch wilson cameras_and_locate.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for cameras and locate drink action."""

    # Get package directories
    locate_drink_pkg_dir = get_package_share_directory('locate_drink_action')

    # Depth camera (Arducam ToF) node
    depth_field = Node(
        package='depth_cam',
        executable='depth_field',
        name='depth_field',
        output='screen',
        parameters=[{
            'frame_id': 'depth_camera_link_optical'
        }],
        remappings=[
            ('/depth_field', '/depth_camera/depth/image_raw')
        ]
    )

    # RGB camera (V4L2) node
    v4l2_camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera',
        output='screen',
        parameters=[{
            'video_device': '/dev/video8',
            'camera_frame_id': 'camera_link_optical',
            'pixel_format': 'YUYV',
            'image_size': [1080, 1080],
            'framerate': 30.0,
        }],
        remappings=[
            ('/v4l2_camera/image_raw', '/rgb_camera/image_raw')
        ]
    )

    # Locate Drink Action Server node (using real robot parameters)
    locate_drink_params_file = os.path.join(
        locate_drink_pkg_dir, 'config', 'locate_drink_params_real.yaml'
    )

    locate_drink_server_node = Node(
        package='locate_drink_action',
        executable='locate_drink_action_server',
        name='locate_drink_action_server',
        output='screen',
        parameters=[locate_drink_params_file],
        emulate_tty=True,
    )

    # Launch all nodes
    return LaunchDescription([
        depth_field,
        v4l2_camera_node,
        locate_drink_server_node,
    ])
