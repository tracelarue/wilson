from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("wilson").to_dict()

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

    # MTC Demo node
    pick_place_demo = Node(
        package="mtc_tutorial",
        executable="mtc_tutorial",
        output="screen",
        parameters=[
            moveit_config,
        ],
    )

    # Delay pick_place_demo node by 10 seconds
    delayed_pick_place_demo = TimerAction(
        period=10.0,
        actions=[pick_place_demo]
    )

    return LaunchDescription([
        mtc_demo_launch,
        delayed_pick_place_demo
    ])