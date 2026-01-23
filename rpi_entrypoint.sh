#!/bin/bash

set -e

# Set DISPLAY for GUI applications (OpenCV windows)
export DISPLAY=:0
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

source /opt/ros/humble/setup.bash
if [ -f install/setup.bash ]; then
    source install/setup.bash
fi


echo "Provided arguments: $@"

exec $@