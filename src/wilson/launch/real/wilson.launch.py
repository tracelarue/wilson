import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    package_name = 'wilson'
    gemini_mcp_path = '/wilson/src/gemini_mcp'
    
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
        default_value=os.path.join(get_package_share_directory(package_name), 'maps', 'downstairs_save.yaml'),
        description='Full path to map yaml file'
    )
    
    declare_autostart = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack'
    )

    declare_initial_pose_x = DeclareLaunchArgument(
        'initial_pose_x',
        default_value='0.0',
        description='Initial pose X coordinate for AMCL'
    )

    declare_initial_pose_y = DeclareLaunchArgument(
        'initial_pose_y',
        default_value='0.0',
        description='Initial pose Y coordinate for AMCL'
    )

    declare_initial_pose_yaw = DeclareLaunchArgument(
        'initial_pose_yaw',
        default_value='0.0',
        description='Initial pose yaw angle in radians for AMCL'
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

    # Load MoveIt configuration for action servers
    moveit_config = MoveItConfigsBuilder("wilson", package_name="wilson_moveit_config").to_moveit_configs()

    # Locate Drink Action Server node (using real robot parameters)
    locate_drink_params_file = os.path.join(
        get_package_share_directory('locate_drink_action'),
        'config',
        'locate_drink_params_real.yaml'
    )

    locate_drink_server_node = Node(
        package='locate_drink_action',
        executable='locate_drink_action_server',
        name='locate_drink_action_server',
        output='screen',
        parameters=[locate_drink_params_file],
        emulate_tty=True,
    )

    # Grab Drink Action Server node
    grab_drink_server_node = Node(
        package='grab_drink_action',
        executable='grab_drink_action_server',
        name='grab_drink_action_server',
        output='screen',
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # Timer for action servers - start after move_group is ready
    action_servers_timer = TimerAction(
        period=13.0,
        actions=[locate_drink_server_node, grab_drink_server_node]
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

    # Optional external processes
    gemini = ExecuteProcess(
        cmd=['tilix', '-e', 'ros2', 'run', 'gemini', 'gemini_node', '--mode', 'robot', '--video', 'camera'],
        output='screen',
    )


    rosbridge_server = ExecuteProcess(
        cmd=['tilix', '-e', 'ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml'],
        cwd='/wilson',
        output='screen',
    )

    # Timed optional processes
    rosbridge_timer = TimerAction(
        period=1.0,
        actions=[rosbridge_server]
    )

    gemini_ros_mcp = ExecuteProcess(
        cmd=['tilix', '-e', 'python3', 'gemini_client.py', '--mode=robot'],
        cwd=gemini_mcp_path,
        output='screen',
    )

    gemini_ros_mcp_timer = TimerAction(
        period=3.0,
        actions=[gemini_ros_mcp]
    )


    # Teleop keyboard in separate terminal
    teleop = ExecuteProcess(
        cmd=['tilix', '-e', 'ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/diff_drive_controller/cmd_vel_unstamped'],
        output='screen',
    )

    # Initial pose publisher - sets 2D pose estimate for AMCL
    initial_pose_publisher = Node(
        package='wilson',
        executable='initial_pose_publisher.py',
        name='initial_pose_publisher',
        parameters=[
            {'initial_pose_x': LaunchConfiguration('initial_pose_x')},
            {'initial_pose_y': LaunchConfiguration('initial_pose_y')},
            {'initial_pose_yaw': LaunchConfiguration('initial_pose_yaw')},
            {'use_sim_time': False}
        ],
        output='screen'
    )

    # Timer for initial pose publisher - start after localization is ready
    initial_pose_timer = TimerAction(
        period=30.0,  # Wait for gazebo to be ready
        actions=[initial_pose_publisher]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_map_yaml,
        declare_autostart,
        declare_initial_pose_x,
        declare_initial_pose_y,
        declare_initial_pose_yaw,
        robot_launch,
        move_group_timer,
        nav2_timer,
        localization_timer,
        action_servers_timer,
        #gemini,
        teleop,
        rosbridge_timer,
        gemini_ros_mcp_timer,
        #initial_pose_timer
    ])
