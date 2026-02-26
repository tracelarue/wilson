import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_name = 'wilson'
    pkg_path = get_package_share_directory(package_name)

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_path, 'config', 'real_params.yaml'),
        description='Path to parameters file'
    )

    control_stack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'real', 'control_stack.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': 'false',
            'use_ros2_control': 'true',
            'use_fake_hardware': 'false',
            'params_file': LaunchConfiguration('params_file'),
        }.items()
    )

    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'real', 'sensors.launch.py')
        ])
    )

    return LaunchDescription([
        declare_params_file,
        control_stack_launch,
        sensors_launch,
    ])
