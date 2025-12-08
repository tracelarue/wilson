import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    
    map_file = os.path.join(get_package_share_directory('robot_arm'),'maps', 'downstairs_save.yaml')
    
    
    
    return LaunchDescription([
    
        Node(package="nav2_map_server",
             executable="map_server",
             name="map_server",
             output="screen",
             parameters=[{"use_sim_time":False},
                         {"yaml_filename": map_file}],
             emulate_tty=True),

        Node(package="nav2_lifecycle_manager",
             executable="lifecycle_manager",
             name="lifecycle_manager_mapper",
             output="screen",
             parameters=[{"use_sim_time":False},
                         {"autostart": True},
                         {"node_names": ["map_server"]}])

    ])
