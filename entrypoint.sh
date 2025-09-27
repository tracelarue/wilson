#!/bin/bash

set -e

source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
cd wilson
source install/setup.bash

# Gazebo environment variables for mesh and model loading
export GAZEBO_MODEL_PATH=/wilson/src/wilson/worlds:/wilson/install/wilson/share/wilson/worlds:$GAZEBO_MODEL_PATH
export GAZEBO_RESOURCE_PATH=/wilson/src/wilson:/wilson/install/wilson/share/wilson:/wilson/install/wilson/share:$GAZEBO_RESOURCE_PATH
export GAZEBO_PLUGIN_PATH=/wilson/install/wilson/lib:$GAZEBO_PLUGIN_PATH


echo "Provided arguments: $@"

exec $@