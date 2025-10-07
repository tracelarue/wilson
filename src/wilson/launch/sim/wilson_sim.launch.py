import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def load_yaml_params(file_path):
    """Load parameters from YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def generate_launch_description():
    # Package configuration
    package_name = 'wilson'
    pkg_path = get_package_share_directory(package_name)
    
    # Configuration files
    sim_params_file = os.path.join(pkg_path, 'config', 'sim_params.yaml')
    nav2_params_file = os.path.join(pkg_path, 'config', 'nav2_params.yaml')
    turtlebot3_params_file = os.path.join(pkg_path, 'config', 'turtlebot3_use_sim_time.yaml')
    
    # Load simulation parameters
    sim_params = load_yaml_params(sim_params_file)['/**']['ros__parameters']
    
    # Convert boolean parameters to strings for launch arguments
    use_sim_time = str(sim_params['use_sim_time']).lower()
    use_ros2_control = str(sim_params['use_ros2_control']).lower()
    
    # Other directories
    moveit_launch_dir = os.path.join(get_package_share_directory('wilson_moveit_config'), 'launch')
    
    # Launch arguments
    use_sim_time_config = LaunchConfiguration('use_sim_time')
    map_config = LaunchConfiguration('map')
    autostart_config = LaunchConfiguration('autostart')
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value=use_sim_time,
        description='Use simulation time'
    )
    
    declare_map_yaml = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(pkg_path, 'maps', 'downstairs_sim.yaml'),
        description='Full path to map yaml file'
    )
    
    declare_autostart = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack'
    )
    
    # Initial pose arguments for AMCL localization (from sim_params.yaml)
    declare_initial_pose_x = DeclareLaunchArgument(
        'initial_pose_x',
        default_value=str(sim_params.get('initial_pose_x', 0.0)),
        description='Initial pose X coordinate (from sim_params.yaml)'
    )
    
    declare_initial_pose_y = DeclareLaunchArgument(
        'initial_pose_y', 
        default_value=str(sim_params.get('initial_pose_y', 0.0)),
        description='Initial pose Y coordinate (from sim_params.yaml)'
    )
    
    declare_initial_pose_yaw = DeclareLaunchArgument(
        'initial_pose_yaw',
        default_value=str(sim_params.get('initial_pose_yaw', 0.0)), 
        description='Initial pose yaw angle in radians (from sim_params.yaml)'
    )

    # Core simulation launch
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'sim', 'sim.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time_config,
            'use_ros2_control': use_ros2_control,
            'params_file': sim_params_file
        }.items()
    )

    # Navigation2 servers (planning, control, etc.)
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'base', 'navigation_launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time_config,
            'autostart': autostart_config,
            'params_file': turtlebot3_params_file,
            'map_subscribe_transient_local': 'true'
        }.items()
    )

    # Localization (map server + AMCL)
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'base', 'localization_launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time_config,
            'autostart': autostart_config,
            'params_file': nav2_params_file,
            'map': map_config
        }.items()
    )

    # MoveIt move_group
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(moveit_launch_dir, 'move_group.launch.py')
        ]),
        launch_arguments={
            'use_sim': 'true',
        }.items(),
    )

    # Timed launches to ensure proper startup sequence
    nav2_timer = TimerAction(
        period=3.0,
        actions=[nav2_launch]
    )
    
    localization_timer = TimerAction(
        period=5.0,
        actions=[localization_launch]
    )
    
    move_group_timer = TimerAction(
        period=8.0,
        actions=[move_group_launch]
    )
    
    # Optional external processes
    gemini = ExecuteProcess(
        cmd=['tilix', '-e', 'ros2', 'run', 'gemini', 'gemini_node', '--mode', 'sim', '--video', 'camera'],
        output='screen',
    )

    # Teleop keyboard in separate terminal
    teleop = ExecuteProcess(
        cmd=['tilix', '-e', 'ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/diff_drive_controller/cmd_vel_unstamped'],
        output='screen',
    )
    
    # for use with ros-mcp-server
    rosbridge_server = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rosbridge_server'), 'launch', 'rosbridge_websocket_launch.xml')
        )
    )
    
    # Timed optional processes
    rosbridge_timer = TimerAction(
        period=1.0,
        actions=[rosbridge_server]
    )

    gemini_ros_mcp = ExecuteProcess(
        cmd=['tilix', '-e', 'cd /home/trace/wilson/gemini_live && uv run gemini_live.py'],
        output='screen',
    )

    gemini_ros_mcp_timer = TimerAction(
        period=3.0,
        actions=[gemini_ros_mcp]
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
            {'use_sim_time': sim_params['use_sim_time']}
        ],
        output='screen'
    )
    
    # Timer for initial pose publisher - start after localization is ready
    initial_pose_timer = TimerAction(
        period=20.0,  # Wait for gazebo to be ready
        actions=[initial_pose_publisher]
    )
    
    # Launch description
    return LaunchDescription([
        # Launch arguments
        declare_use_sim_time,
        declare_map_yaml,
        declare_autostart,
        declare_initial_pose_x,
        declare_initial_pose_y,
        declare_initial_pose_yaw,
        
        # Core simulation
        sim_launch,
        
        # Timed component launches
        nav2_timer,
        localization_timer,
        move_group_timer,
        initial_pose_timer,
        
        # Optional components
        #gemini,
        teleop,
        rosbridge_timer,
        #gemini_ros_mcp_timer
    ])
