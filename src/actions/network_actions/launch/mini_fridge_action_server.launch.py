from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mini_fridge_action_server = Node(
        package='network_actions',
        executable='mini_fridge_action_server',
        name='mini_fridge_action_server',
        output='screen',
        parameters=[
            {'esp32_host': '192.168.1.112'},
            {'esp32_port': 80},
            {'request_path': '/mini_fridge'},
            {'wait_ms': 5000},
            {'request_timeout_ms': 12000},
        ],
    )

    return LaunchDescription([
        mini_fridge_action_server,
    ])
