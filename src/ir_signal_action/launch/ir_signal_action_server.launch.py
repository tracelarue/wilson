from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ir_signal_action_server = Node(
        package='ir_signal_action',
        executable='ir_signal_action_server',
        name='ir_signal_action_server',
        output='screen',
        parameters=[
            {'mode': 'robot'},
            {'gpio_pin': 18},
            {'pwm_channel': 2},
            {'pwm_chip': 0},
            {'burst_duration_ms': 1000},
            {'wait_ms': 5000},
            {'carrier_frequency_hz': 38000},
            {'sim_topic': '/ir_signal'},
        ],
    )

    return LaunchDescription([
        ir_signal_action_server,
    ])
