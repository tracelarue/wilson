#!/usr/bin/env python3

"""
Launch file for the Locate Drink Action Server.

This launch file starts the action server with parameters loaded from YAML.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for locate drink action server."""

    # Get package directory
    pkg_dir = get_package_share_directory('locate_drink_action')

    # Path to parameter file
    params_file = os.path.join(pkg_dir, 'config', 'locate_drink_params.yaml')

    # Locate Drink Action Server node
    locate_drink_server_node = Node(
        package='locate_drink_action',
        executable='locate_drink_action_server',
        name='locate_drink_action_server',
        output='screen',
        parameters=[params_file],
        emulate_tty=True,
    )

    return LaunchDescription([
        locate_drink_server_node,
    ])
