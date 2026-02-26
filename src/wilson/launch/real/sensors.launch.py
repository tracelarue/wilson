import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    ld19_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ldlidar_stl_ros2'), 'launch', 'ld19.launch.py')
        ])
    )

    depth_field = Node(
        package='depth_cam',
        executable='depth_field',
        name='depth_field',
        output='screen',
        parameters=[{
            'frame_id': 'depth_camera_link_optical'
        }],
        remappings=[
            ('/depth_field', '/depth_camera/depth/image_raw')
        ]
    )

    v4l2_camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera',
        output='screen',
        parameters=[{
            'video_device': '/dev/wilson/rgb_camera',
            'camera_frame_id': 'camera_link_optical',
            'pixel_format': 'YUYV',
            'image_size': [1280, 720],  # 1280x720 max
            'framerate': 30.0,
        }],
        remappings=[
            ('/image_raw', '/rgb_camera/image_raw')
        ]
    )

    delayed_v4l2_camera_node = TimerAction(
        period=4.0,
        actions=[v4l2_camera_node],
    )
    delayed_depth_field = TimerAction(
        period=4.0,
        actions=[depth_field],
    )

    return LaunchDescription([
        ld19_launch,
        delayed_depth_field,
        delayed_v4l2_camera_node,
    ])
