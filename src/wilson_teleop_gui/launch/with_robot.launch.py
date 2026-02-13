import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    wilson_pkg_dir = get_package_share_directory("wilson")
    default_params_file = os.path.join(wilson_pkg_dir, "config", "real_params.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to Wilson real robot params file",
    )
    cmd_vel_topic_arg = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="Twist topic to publish to",
    )
    linear_speed_arg = DeclareLaunchArgument(
        "linear_speed",
        default_value="0.25",
        description="Linear speed in m/s",
    )
    angular_speed_arg = DeclareLaunchArgument(
        "angular_speed",
        default_value="0.9",
        description="Angular speed in rad/s",
    )
    linear_deadband_arg = DeclareLaunchArgument(
        "linear_deadband",
        default_value="0.08",
        description="Linear axis deadband in normalized joystick units [0..1)",
    )
    angular_deadband_arg = DeclareLaunchArgument(
        "angular_deadband",
        default_value="0.08",
        description="Angular axis deadband in normalized joystick units [0..1)",
    )

    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(wilson_pkg_dir, "launch", "real", "robot.launch.py")
        ),
        launch_arguments={"params_file": LaunchConfiguration("params_file")}.items(),
    )

    teleop_node = Node(
        package="wilson_teleop_gui",
        executable="teleop_gui_node",
        output="screen",
        parameters=[
            {
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                "linear_speed": LaunchConfiguration("linear_speed"),
                "angular_speed": LaunchConfiguration("angular_speed"),
                "linear_deadband": LaunchConfiguration("linear_deadband"),
                "angular_deadband": LaunchConfiguration("angular_deadband"),
            }
        ],
    )

    return LaunchDescription(
        [
            params_file_arg,
            cmd_vel_topic_arg,
            linear_speed_arg,
            angular_speed_arg,
            linear_deadband_arg,
            angular_deadband_arg,
            robot_launch,
            teleop_node,
        ]
    )
