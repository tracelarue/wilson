import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    # Package configuration
    pkg_name = 'wilson'
    pkg_path = get_package_share_directory(pkg_name)
    
    # Launch arguments configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    use_ros2_control = LaunchConfiguration('use_ros2_control')
    params_file = LaunchConfiguration('params_file')
    
    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='false', description='Use sim time if true')
    declare_use_fake_hardware = DeclareLaunchArgument('use_fake_hardware', default_value='false', description='Use fake hardware interface')
    declare_use_ros2_control = DeclareLaunchArgument('use_ros2_control', default_value='true', description='Use ROS2 Control')
    declare_params_file = DeclareLaunchArgument('params_file', default_value=os.path.join(pkg_path, 'config', 'real_params.yaml'), description='Path to parameters file')
    
    # Environment setup
    set_rviz_log_level = SetEnvironmentVariable('ROSCONSOLE_MIN_SEVERITY', 'WARN')
    
    # Robot description setup
    xacro_file = os.path.join(pkg_path, 'urdf', 'wilson_real.urdf.xacro')
    robot_description_config = Command([
        'xacro ', xacro_file, 
        ' use_sim_time:=', use_sim_time,
        ' use_fake_hardware:=', use_fake_hardware,
        ' use_ros2_control:=', use_ros2_control
    ])
    
    # Node parameters
    robot_params = [{'robot_description': robot_description_config, 'use_sim_time': use_sim_time}, params_file]
    
    # Node definitions
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=robot_params,
    )
    
    node_rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_path, 'config', 'rviz', 'sim.rviz'), '--ros-args', '--log-level', 'WARN'],
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    # Launch description
    return LaunchDescription([
        # Environment setup
        #set_rviz_log_level,
        
        # Launch arguments
        declare_use_sim_time,
        declare_use_fake_hardware,
        declare_use_ros2_control,
        declare_params_file,
        
        # Nodes
        node_robot_state_publisher,
        #node_rviz2
    ])
