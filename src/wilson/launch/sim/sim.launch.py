import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def load_yaml_params(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def generate_launch_description():
    package_name = 'wilson'
    pkg_path = get_package_share_directory(package_name)

    sim_params_file = os.path.join(pkg_path, 'config', 'sim_params.yaml')
    gazebo_params_file = os.path.join(pkg_path, 'config', 'gazebo_params.yaml')
    world_file_path = os.path.join(pkg_path, 'worlds', 'downstairs_combined.world')

    sim_params = load_yaml_params(sim_params_file)['/**']['ros__parameters']

    params_file = LaunchConfiguration('params_file')
    use_sim_time_config = LaunchConfiguration('use_sim_time')
    use_ros2_control_config = LaunchConfiguration('use_ros2_control')
    use_fake_hardware_config = LaunchConfiguration('use_fake_hardware')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=sim_params_file,
        description='Path to parameters file'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value=str(sim_params['use_sim_time']).lower(),
        description='Use simulation clock'
    )
    declare_use_ros2_control = DeclareLaunchArgument(
        'use_ros2_control',
        default_value=str(sim_params['use_ros2_control']).lower(),
        description='Enable ros2_control in simulation'
    )
    declare_use_fake_hardware = DeclareLaunchArgument(
        'use_fake_hardware',
        default_value=str(sim_params['use_fake_hardware']).lower(),
        description='Use fake hardware interfaces'
    )

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'base', 'sim_rsp.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time_config,
            'use_ros2_control': use_ros2_control_config,
            'use_fake_hardware': use_fake_hardware_config,
            'params_file': params_file,
        }.items()
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time_config,
            'world': world_file_path,
            'extra_gazebo_args': '--ros-args --params-file ' + gazebo_params_file,
        }.items()
    )

    spawn_and_controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'sim', 'spawn_and_controllers.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time_config,
        }.items()
    )

    return LaunchDescription([
        declare_params_file,
        declare_use_sim_time,
        declare_use_ros2_control,
        declare_use_fake_hardware,
        rsp,
        gazebo,
        spawn_and_controllers,
    ])
