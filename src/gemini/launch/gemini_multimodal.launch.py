from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    video_mode_arg = DeclareLaunchArgument(
        'video_mode',
        default_value='camera',
        description='Video input mode (none, camera, or screen)'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug logging'
    )
    
    # Set up the node
    gemini_node = Node(
        package='gemini',
        executable='multimodal',
        name='gemini_multimodal',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time', default='true')
        }],
        arguments=[
            '--video', LaunchConfiguration('video_mode'),
            # Add debug flag if debug is true
            LaunchConfiguration('debug', default='false')
        ]
    )
    
    # Define launch description
    return LaunchDescription([
        # Google API key from environment variable
        SetEnvironmentVariable('GOOGLE_API_KEY', EnvironmentVariable('GOOGLE_API_KEY', default_value='')),
        
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        video_mode_arg,
        debug_arg,
        
        # Nodes
        gemini_node
    ])