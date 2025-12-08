import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    package_name = 'robot_arm'

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package_name), 'launch', 'sim.launch.py')
        )
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package_name), 'launch', 'base', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'autostart': 'true',
            'params_file': os.path.join(get_package_share_directory(package_name), 'config', 'nav2_params.yaml'),
            'map_subscribe_transient_local': 'true'
        }.items()
    )

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package_name), 'launch', 'base', 'localization_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'autostart': 'true',
            'params_file': os.path.join(get_package_share_directory(package_name), 'config', 'nav2_params.yaml'),
            'map': os.path.join(get_package_share_directory(package_name), 'maps', 'downstairs_sim.yaml')
        }.items()
    )

    gemini = ExecuteProcess(
        cmd=['tilix', '-e', 'ros2', 'run', 'gemini', 'multimodal', '--mode', 'sim'],
        output='screen'
    )

    return LaunchDescription([
        SetEnvironmentVariable('QT_AUTO_SCREEN_SCALE_FACTOR', '0'),
        SetEnvironmentVariable('QT_SCALE_FACTOR', '1'),
        sim,
        #nav2,
        #localization,
        gemini
    ])
