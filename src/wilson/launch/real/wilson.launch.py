import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    package_name = 'wilson'
    
    # Load real robot parameters
    real_params_file = os.path.join(get_package_share_directory(package_name), 'config', 'real_params.yaml')

    launch_dir = os.path.join(
        get_package_share_directory(
            'wilson_moveit_config'), 'launch')
    
    # Declare launch arguments - can override parameters from file
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    declare_map_yaml = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(get_package_share_directory(package_name), 'maps', 'downstairs_sim.yaml'),
        description='Full path to map yaml file'
    )
    
    declare_autostart = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack'
    )

        # Launch real robot
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(package_name), 'launch', 'real', 'robot.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'use_ros2_control': 'true',
            'params_file': real_params_file
        }.items()
    )

    # Launch Navigation2 servers (planning, control, etc.)
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(package_name), 'launch', 'base', 'navigation_launch.py')
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'autostart': LaunchConfiguration('autostart'),
            'params_file': os.path.join(get_package_share_directory(package_name), 'config', 'turtlebot3_use_sim_time.yaml'),
            'map_subscribe_transient_local': 'true'
        }.items()
    )

    # Launch Localization (map server + AMCL)
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(package_name), 'launch', 'base', 'localization_launch.py')
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'autostart': LaunchConfiguration('autostart'),
            'params_file': os.path.join(get_package_share_directory(package_name), 'config', 'nav2_params.yaml'),
            'map': LaunchConfiguration('map')
        }.items()
    )

    # Launch MoveIt move_group
    move_group_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_dir, '/move_group.launch.py']),
            launch_arguments={
                'use_sim': 'false',
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }.items(),
    )

    # Add timer to start MoveIt after navigation and localization
    move_group_timer = TimerAction(
        period=3.0,
        actions=[move_group_launch]
    )

    # Add timer to start navigation after Gazebo is ready
    nav2_timer = TimerAction(
        period=10.0,
        actions=[nav2_launch]
    )

    # Add timer to start localization after Gazebo is ready
    localization_timer = TimerAction(
        period=10.0,
        actions=[localization_launch]
    )

    gemini = Node(
        package='gemini',
        executable='gemini_node',
        name='gemini',
        output='screen',
        arguments=['--mode', 'robot', '--video', 'camera'],  # Changed: removed extra --ros-args
        parameters=[{'use_sim_time': False}],
        emulate_tty=True
    )

    gemini_timer = TimerAction(
        period=5.0,
        actions=[gemini]
    )


    return LaunchDescription([
        declare_use_sim_time,
        declare_map_yaml,
        declare_autostart,
        robot_launch,
        move_group_timer,
        nav2_timer,
        localization_timer,
        #gemini_timer
    ])
