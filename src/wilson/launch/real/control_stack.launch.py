import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'wilson'
    pkg_path = get_package_share_directory(package_name)

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    use_ros2_control = LaunchConfiguration('use_ros2_control')
    params_file = LaunchConfiguration('params_file')

    robot_description = Command([
        'xacro ',
        os.path.join(pkg_path, 'urdf', 'wilson_real.urdf.xacro'),
        ' use_sim_time:=', use_sim_time,
        ' use_fake_hardware:=', use_fake_hardware,
        ' use_ros2_control:=', use_ros2_control,
    ])
    controller_params_file = os.path.join(pkg_path, 'config', 'robot_controller_manager.yaml')

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'base', 'rsp.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_ros2_control': use_ros2_control,
            'use_fake_hardware': use_fake_hardware,
            'params_file': params_file,
        }.items()
    )

    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{'robot_description': robot_description}, controller_params_file],
        output='screen',
    )

    jointstate_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        output='screen',
    )
    delayed_jointstate_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[jointstate_broadcaster_spawner],
        )
    )

    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'diff_drive_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        output='screen',
    )
    delayed_diff_drive_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=jointstate_broadcaster_spawner,
            on_exit=[diff_drive_spawner],
        )
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'arm_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        output='screen',
    )
    delayed_arm_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=jointstate_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'gripper_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        output='screen',
    )
    delayed_gripper_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=jointstate_broadcaster_spawner,
            on_exit=[gripper_controller_spawner],
        )
    )

    twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        parameters=[
            os.path.join(pkg_path, 'config', 'twist_mux.yaml'),
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('/cmd_vel_out', '/diff_drive_controller/cmd_vel_unstamped')],
    )

    return LaunchDescription([
        rsp,
        controller_manager,
        delayed_jointstate_broadcaster_spawner,
        delayed_diff_drive_spawner,
        delayed_arm_controller_spawner,
        delayed_gripper_controller_spawner,
        twist_mux,
    ])
