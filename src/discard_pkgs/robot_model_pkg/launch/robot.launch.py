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

    package_name='robot_model_pkg' #<--- CHANGE ME
    robot_description = Command(['ros2 param get --hide-type /robot_state_publisher robot_description'])
    controller_params_file = os.path.join(get_package_share_directory(package_name), 'config', 'my_controllers.yaml')

    rsp = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','base','rsp.launch.py')]), 
                launch_arguments={
                    'use_sim_time': 'false',
                    'use_ros2_control': 'true'
                }.items()
    )


    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[{'robot_description':robot_description},
                    controller_params_file],
    )
    delayed_controller_manager = TimerAction(
        period=3.0, 
        actions=[controller_manager],
    )


    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_cont"],
    )
    delayed_diff_drive_spawner = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[diff_drive_spawner],
        )
    )

    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_broad"],
    )
    delayed_joint_broad_spawner = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[joint_broad_spawner],
        )
    )

    twist_mux_params = os.path.join(get_package_share_directory(package_name),'config','twist_mux.yaml')
    twist_mux = Node(
        package="twist_mux",
        executable="twist_mux",
        parameters=[twist_mux_params, {'use_sim_time': False}],
        remappings=[('/cmd_vel_out','/diff_cont/cmd_vel_unstamped')]
    )


    ld19_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('ldlidar_stl_ros2'),'launch','ld19.launch.py')])
    )

    tof_pointcloud = Node(
        package='depth_cam',
        executable='tof_pointcloud',
        name='tof_pointcloud',
        output='screen',
        # No arguments to avoid issues with argument parsing
        parameters=[{
            'frame_id': 'depth_camera_link_optical'
        }]
    )
    

    v4l2_camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera',
        output='screen',
        parameters=[{
            'video_device': '/dev/video8',          # Adjust as needed
            'camera_frame_id': 'camera_link_optical',
            'pixel_format': 'YUYV',                 # Common format
            'image_size': [1280, 480],               # Adjust as needed
            'framerate': 30.0,
        }]
    )



    # Launch them all!
    return LaunchDescription([
        rsp,
        delayed_controller_manager,
        delayed_diff_drive_spawner,
        delayed_joint_broad_spawner,
        twist_mux,
        ld19_launch,
        tof_pointcloud,
        v4l2_camera_node,

    ])