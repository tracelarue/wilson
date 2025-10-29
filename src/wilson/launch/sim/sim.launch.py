import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable


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
    gazebo_params_file = os.path.join(pkg_path, 'config', 'gazebo_params.yaml')
    twist_mux_params_file = os.path.join(pkg_path, 'config', 'twist_mux.yaml')
    world_file_path = os.path.join(pkg_path, 'worlds', 'downstairs_combined.world')
    
    # Load simulation parameters
    sim_params = load_yaml_params(sim_params_file)['/**']['ros__parameters']
    
    # Launch arguments
    params_file = LaunchConfiguration('params_file')
    
    # Declare launch arguments  
    declare_params_file = DeclareLaunchArgument('params_file',
        default_value=sim_params_file,
        description='Path to parameters file'
    )
    
    # Convert boolean parameters to strings for launch arguments
    use_sim_time = str(sim_params['use_sim_time']).lower()
    use_ros2_control = str(sim_params['use_ros2_control']).lower()
    use_fake_hardware = str(sim_params['use_fake_hardware']).lower()
    
    # Include robot state publisher
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'base', 'rsp.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_ros2_control': use_ros2_control,
            'use_fake_hardware': use_fake_hardware,
            'params_file': params_file
        }.items()
    )

    # Include Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py'
            )
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'world': world_file_path,
            'extra_gazebo_args': '--ros-args --params-file ' + gazebo_params_file
        }.items()
    )
    
    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_bot',
            '-x', '0', '-y', '0', '-z', '0.0181',
            '-reference_frame', 'world'
        ]
    )
    
    # Controller nodes
    joint_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        parameters=[{'use_sim_time': sim_params['use_sim_time']}],
        output="screen"
    )
    
    diff_drive_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_controller"],
        parameters=[{'use_sim_time': sim_params['use_sim_time']}],
        output="screen"
    )
    
    arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        parameters=[{'use_sim_time': sim_params['use_sim_time']}],
        output="screen"
    )
    
    gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller"],
        parameters=[{'use_sim_time': sim_params['use_sim_time']}],
        output="screen"
    )
    
    # Twist multiplexer for velocity command handling
    twist_mux = Node(
        package="twist_mux",
        executable="twist_mux",
        parameters=[twist_mux_params_file, {'use_sim_time': sim_params['use_sim_time']}],
        remappings=[('/cmd_vel_out', '/diff_drive_controller/cmd_vel_unstamped')]
    )
    
    # Launch description
    return LaunchDescription([
        # Launch arguments
        declare_params_file,
        
        # Core simulation components
        rsp,
        gazebo,
        spawn_robot,
        
        # Controllers
        joint_broadcaster,
        diff_drive_controller,
        arm_controller,
        gripper_controller,
        
        # Additional nodes
        twist_mux,
    ])