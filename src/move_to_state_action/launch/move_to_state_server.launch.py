from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the MoveToState action server."""

    # Declare arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # Get launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time')

    # MoveToState action server node
    move_to_state_server_node = Node(
        package='move_to_state_action',
        executable='move_to_state_action_server',
        name='move_to_state_action_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        move_to_state_server_node,
    ])
