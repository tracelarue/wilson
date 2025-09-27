source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
cd wilson
source install/setup.bash

# Gazebo environment variables - the key is adding /wilson/install/wilson/share to resolve package:// URIs
export GAZEBO_MODEL_PATH=/wilson/src/wilson/worlds:/wilson/install/wilson/share/wilson/worlds:$GAZEBO_MODEL_PATH
export GAZEBO_RESOURCE_PATH=/wilson/src/wilson:/wilson/install/wilson/share/wilson:/wilson/install/wilson/share:$GAZEBO_RESOURCE_PATH
export GAZEBO_PLUGIN_PATH=/wilson/install/wilson/lib:$GAZEBO_PLUGIN_PATH