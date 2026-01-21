import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command
from launch.event_handlers import OnProcessStart



def generate_launch_description():


    # Include the robot_state_publisher launch file, provided by our own package. Force sim time to be enabled
    # !!! MAKE SURE YOU SET THE PACKAGE NAME CORRECTLY !!!

    package_name='wilson' #<--- CHANGE ME
    slam_params = os.path.join(get_package_share_directory(package_name),'config','raspi_mapper_params_online_async.yaml')


    #robot = IncludeLaunchDescription(
    #    PythonLaunchDescriptionSource([os.path.join(
    #        get_package_share_directory(package_name),'launch','real','robot.launch.py')]), 
    #)

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('slam_toolbox'),'launch','online_async_launch.py')]),
        launch_arguments={
            'slam_params_file': slam_params,
            'use_sim_time': 'false'
        }.items()
    )
    # Launch them all!
    return LaunchDescription([
        slam,
    ])