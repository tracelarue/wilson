import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _workspace_root_from_share(pkg_share_path):
    return os.path.abspath(os.path.join(pkg_share_path, '..', '..', '..', '..'))


def _event_text(event):
    cmd = event.cmd
    if isinstance(cmd, (list, tuple)):
        cmd = ' '.join(str(part) for part in cmd)
    return f'{event.process_name} {cmd}'


def _start_once_on_process_start(stage_state, stage_key, matcher, action, label):
    def _handler(event, _context):
        if stage_state.get(stage_key):
            return None
        if not matcher(event):
            return None
        stage_state[stage_key] = True
        return [
            LogInfo(msg=f'[wilson real] starting {label} after {event.process_name} start'),
            action,
        ]

    return _handler


def generate_launch_description():
    package_name = 'wilson'
    pkg_path = get_package_share_directory(package_name)
    workspace_root = _workspace_root_from_share(pkg_path)
    gemini_mcp_path = os.path.join(workspace_root, 'src', 'gemini_mcp')

    real_params_file = os.path.join(pkg_path, 'config', 'real_params.yaml')
    nav2_params_file = os.path.join(pkg_path, 'config', 'nav2_params_real.yaml')
    moveit_launch_dir = os.path.join(get_package_share_directory('wilson_moveit_config'), 'launch')

    use_sim_time_config = LaunchConfiguration('use_sim_time')
    map_config = LaunchConfiguration('map')
    autostart_config = LaunchConfiguration('autostart')
    publish_initial_pose_config = LaunchConfiguration('publish_initial_pose')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    declare_map_yaml = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(pkg_path, 'maps', 'downstairs_save_edit.yaml'),
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
    declare_publish_initial_pose = DeclareLaunchArgument(
        'publish_initial_pose',
        default_value='false',
        description='Publish an initial AMCL pose after startup'
    )

    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'real', 'robot.launch.py')
        ]),
        launch_arguments={
            'params_file': real_params_file,
        }.items()
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'base', 'localization_launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time_config,
            'autostart': autostart_config,
            'params_file': nav2_params_file,
            'map': map_config,
        }.items()
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'base', 'navigation_launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time_config,
            'autostart': autostart_config,
            'params_file': nav2_params_file,
            'map_subscribe_transient_local': 'true',
        }.items()
    )

    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(moveit_launch_dir, 'move_group.launch.py')]),
        launch_arguments={
            'use_sim': 'false',
            'use_sim_time': use_sim_time_config,
        }.items(),
    )

    locate_object_params_file = os.path.join(
        get_package_share_directory('locate_object_action'),
        'config',
        'locate_object_params_real.yaml',
    )
    action_servers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'base', 'action_servers.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time_config,
            'locate_object_params_file': locate_object_params_file,
            'mini_fridge_request_timeout_ms': '8000',
        }.items()
    )

    rosbridge_server = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rosbridge_server'),
                'launch',
                'rosbridge_websocket_launch.xml',
            )
        ),
    )

    gemini_ros_mcp = ExecuteProcess(
        cmd=[
            'tilix',
            '-e',
            'bash',
            '-c',
            'python3 gemini.py --mode=robot; echo "Process finished. Press enter to close..."; read',
        ],
        cwd=gemini_mcp_path,
        output='screen',
    )
    teleop = ExecuteProcess(
        cmd=[
            'tilix',
            '-e',
            'ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap '
            'cmd_vel:=/diff_drive_controller/cmd_vel_unstamped',
        ],
        output='screen',
    )

    initial_pose_publisher = Node(
        package='wilson',
        executable='initial_pose_publisher.py',
        name='initial_pose_publisher',
        parameters=[
            {'initial_pose_x': LaunchConfiguration('initial_pose_x')},
            {'initial_pose_y': LaunchConfiguration('initial_pose_y')},
            {'initial_pose_yaw': LaunchConfiguration('initial_pose_yaw')},
            {'use_sim_time': use_sim_time_config},
        ],
        output='screen',
        condition=IfCondition(publish_initial_pose_config),
    )
    initial_pose_after_amcl_timer = TimerAction(period=30.0, actions=[initial_pose_publisher])

    startup_sequence_state = {
        'localization': False,
        'nav2': False,
        'move_group': False,
        'action_servers': False,
        'initial_pose_timer': False,
    }

    start_localization = RegisterEventHandler(
        event_handler=OnProcessStart(
            on_start=_start_once_on_process_start(
                startup_sequence_state,
                'localization',
                lambda event: 'spawner' in _event_text(event) and 'diff_drive_controller' in _event_text(event),
                localization_launch,
                'localization',
            )
        )
    )
    start_nav2 = RegisterEventHandler(
        event_handler=OnProcessStart(
            on_start=_start_once_on_process_start(
                startup_sequence_state,
                'nav2',
                lambda event: 'lifecycle_manager_localization' in _event_text(event),
                nav2_launch,
                'navigation',
            )
        )
    )
    start_move_group = RegisterEventHandler(
        event_handler=OnProcessStart(
            on_start=_start_once_on_process_start(
                startup_sequence_state,
                'move_group',
                lambda event: 'lifecycle_manager_navigation' in _event_text(event),
                move_group_launch,
                'move_group',
            )
        )
    )
    start_action_servers = RegisterEventHandler(
        event_handler=OnProcessStart(
            on_start=_start_once_on_process_start(
                startup_sequence_state,
                'action_servers',
                lambda event: 'move_group' in _event_text(event),
                action_servers_launch,
                'action servers',
            )
        )
    )
    start_initial_pose_timer = RegisterEventHandler(
        event_handler=OnProcessStart(
            on_start=_start_once_on_process_start(
                startup_sequence_state,
                'initial_pose_timer',
                lambda event: 'amcl' in _event_text(event),
                initial_pose_after_amcl_timer,
                'initial pose timer',
            )
        )
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_map_yaml,
        declare_autostart,
        declare_initial_pose_x,
        declare_initial_pose_y,
        declare_initial_pose_yaw,
        declare_publish_initial_pose,
        start_localization,
        start_nav2,
        start_move_group,
        start_action_servers,
        start_initial_pose_timer,
        robot_launch,
        rosbridge_server,
        gemini_ros_mcp,
        teleop,
        # Quick local toggles (comment/uncomment while iterating):
        # teleop,
        # gemini_ros_mcp,
        # initial_pose_after_amcl_timer,
    ])
