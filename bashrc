source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
cd /wilson
if [ -f install/setup.bash ]; then
    source install/setup.bash
fi

# Gazebo environment variables - prioritize local models first to avoid downloads
export GAZEBO_MODEL_PATH=/wilson/.gazebo/models:/wilson/src/wilson/worlds:/wilson/install/wilson/share/wilson/worlds:$GAZEBO_MODEL_PATH
export GAZEBO_RESOURCE_PATH=/wilson/src/wilson:/wilson/install/wilson/share/wilson:/wilson/install/wilson/share:$GAZEBO_RESOURCE_PATH
export GAZEBO_PLUGIN_PATH=/wilson/install/wilson/lib:$GAZEBO_PLUGIN_PATH