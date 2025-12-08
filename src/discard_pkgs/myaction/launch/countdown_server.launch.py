#!/usr/bin/env python3
import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    myaction = Node(
        package='myaction',
        executable='countdown_server',
        name='countdown_action_server',
        output='screen',
        parameters=[],
    )

    rosbridge_server = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rosbridge_server'), 'launch', 'rosbridge_websocket_launch.xml')
        )
    )

    return LaunchDescription([
        myaction,
        rosbridge_server
    ])

# ros2 action send_goal /countdown myaction/action/CountDown "{count_from: 25}" --feedback