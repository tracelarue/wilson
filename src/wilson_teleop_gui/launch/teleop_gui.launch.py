from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
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
    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate_hz",
        default_value="20.0",
        description="Continuous publish rate in Hz",
    )

    teleop_node = Node(
        package="wilson_teleop_gui",
        executable="teleop_gui_node",
        name="wilson_teleop_gui",
        output="screen",
        parameters=[
            {
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                "linear_speed": LaunchConfiguration("linear_speed"),
                "angular_speed": LaunchConfiguration("angular_speed"),
                "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
            }
        ],
    )

    return LaunchDescription(
        [
            cmd_vel_topic_arg,
            linear_speed_arg,
            angular_speed_arg,
            publish_rate_arg,
            teleop_node,
        ]
    )
