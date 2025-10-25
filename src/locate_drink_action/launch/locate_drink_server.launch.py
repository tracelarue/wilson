#!/usr/bin/env python3

"""
Launch file for the Locate Drink Action Server.

This launch file starts the action server with parameters loaded from YAML.
Supports switching between simulation and real robot camera parameters.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for locate drink action server."""

    # Get package directory
    pkg_dir = get_package_share_directory('locate_drink_action')

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Whether to use simulation parameters (true) or real robot parameters (false)'
    )

    # Get launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Determine which parameter file to use based on use_sim_time argument
    params_file = PythonExpression([
        "'", os.path.join(pkg_dir, 'config', 'locate_drink_params_sim.yaml'), "'",
        " if '", use_sim_time, "' == 'true' else ",
        "'", os.path.join(pkg_dir, 'config', 'locate_drink_params_real.yaml'), "'"
    ])

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
        use_sim_time_arg,
        locate_drink_server_node,
    ])
