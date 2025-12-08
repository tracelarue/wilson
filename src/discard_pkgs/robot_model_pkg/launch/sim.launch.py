import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'robot_model_pkg'
    gazebo_params_file = os.path.join(
        get_package_share_directory(package_name),
        'config','gazebo_params.yaml'
    )
    world_file_path = os.path.join(
        get_package_share_directory(package_name),
        'worlds','downstairs_combined.world'
    )

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory(package_name),
                'launch','base','rsp.launch.py'
            )
        ]),
        launch_arguments={
            'use_sim_time': 'true',
            'use_ros2_control': 'true'
        }.items()
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch','gazebo.launch.py'
            )
        ]),
        launch_arguments={
            'world': world_file_path,
            'extra_gazebo_args': '--ros-args --params-file ' + gazebo_params_file
        }.items()
    )

    spawn_robot = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=[
            '-topic','robot_description',
            '-entity','my_bot',
            '-x','0','-y','0','-z','0',
            '-reference_frame','world'
        ],
    )

    # these use the normal spawner (it will load→configure for you)
    diff_drive_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_cont"],
        output="screen"
    )
    joint_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_broad"],
        output="screen"
    )

    twist_mux_params = os.path.join(
        get_package_share_directory(package_name),
        'config','twist_mux.yaml'
    )
    twist_mux = Node(
        package="twist_mux",
        executable="twist_mux",
        parameters=[twist_mux_params, {'use_sim_time': True}],
        remappings=[('/cmd_vel_out','/diff_cont/cmd_vel_unstamped')]
    )

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_robot,
        diff_drive_controller,
        joint_broadcaster, 
        twist_mux,
    ])