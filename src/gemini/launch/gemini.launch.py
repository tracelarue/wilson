from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    
    gemini = ExecuteProcess(
        cmd=['tilix', '-e', 'ros2', 'run', 'gemini', 'gemini_node', '--mode', 'sim','--video', 'camera'],
        output='screen'
    )

    # Define launch description
    return LaunchDescription([
        # Google API key from environment variable
        SetEnvironmentVariable('GOOGLE_API_KEY', EnvironmentVariable('GOOGLE_API_KEY', default_value='')),
        gemini
    ])