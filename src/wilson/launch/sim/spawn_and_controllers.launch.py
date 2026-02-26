import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_path = get_package_share_directory('wilson')
    use_sim_time = LaunchConfiguration('use_sim_time')

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-timeout', '120',
            '-topic', 'robot_description',
            '-entity', 'my_bot',
            '-x', '0', '-y', '0', '-z', '0.0181',
            '-reference_frame', 'world'
        ],
        output='screen',
    )

    joint_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    diff_drive_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'diff_drive_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'arm_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'gripper_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        parameters=[
            os.path.join(pkg_path, 'config', 'twist_mux.yaml'),
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('/cmd_vel_out', '/diff_drive_controller/cmd_vel_unstamped')],
        output='screen',
    )

    delayed_joint_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_broadcaster],
        )
    )

    delayed_controllers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_broadcaster,
            on_exit=[diff_drive_controller, arm_controller, gripper_controller, twist_mux],
        )
    )

    return LaunchDescription([
        spawn_robot,
        delayed_joint_broadcaster,
        delayed_controllers,
    ])
