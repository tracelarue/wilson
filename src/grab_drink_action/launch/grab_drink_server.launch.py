from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Load MoveIt configuration (adjust the package name to match your robot's MoveIt config)
    moveit_config = MoveItConfigsBuilder("wilson", package_name="wilson_moveit_config").to_moveit_configs()

    # Grab Drink Action Server Node
    grab_drink_server_node = Node(
        package="grab_drink_action",
        executable="grab_drink_action_server",
        name="grab_drink_action_server",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # Include mtc_demo launch file
    mtc_demo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("mtc_tutorial"),
                "launch",
                "mtc_demo.launch.py"
            ])
        ])
    )

    return LaunchDescription([
        grab_drink_server_node,
        mtc_demo_launch,
    ])
