from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    locate_object_params_file = LaunchConfiguration('locate_object_params_file')
    mini_fridge_request_timeout_ms = LaunchConfiguration('mini_fridge_request_timeout_ms')

    moveit_config = MoveItConfigsBuilder(
        'wilson', package_name='wilson_moveit_config'
    ).to_moveit_configs()

    locate_object_server_node = Node(
        package='locate_object_action',
        executable='locate_object_action_server',
        name='locate_object_action_server',
        output='screen',
        parameters=[locate_object_params_file],
    )

    grab_object_server_node = Node(
        package='grab_object_action',
        executable='grab_object_action_server',
        name='grab_object_action_server',
        output='screen',
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    move_to_state_server_node = Node(
        package='move_to_state_action',
        executable='move_to_state_action_server',
        name='move_to_state_action_server',
        output='screen',
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {'use_sim_time': use_sim_time},
        ],
    )

    navigate_to_location_server_node = Node(
        package='navigate_to_location_action',
        executable='navigate_to_location_server',
        name='navigate_to_location_server',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    mini_fridge_action_server_node = Node(
        package='network_actions',
        executable='mini_fridge_action_server',
        name='mini_fridge_action_server',
        output='screen',
        parameters=[
            {'esp32_host': '192.168.1.112'},
            {'esp32_port': 80},
            {'request_path': '/mini_fridge'},
            {'wait_ms': 5000},
            {
                'request_timeout_ms': ParameterValue(
                    mini_fridge_request_timeout_ms,
                    value_type=int,
                )
            },
            {'use_sim_time': use_sim_time},
        ],
    )

    # Launch helper include: expects caller to provide launch configurations.
    return LaunchDescription([
        locate_object_server_node,
        grab_object_server_node,
        move_to_state_server_node,
        navigate_to_location_server_node,
        mini_fridge_action_server_node,
    ])
