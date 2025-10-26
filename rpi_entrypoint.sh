#!/bin/bash

set -e

# Set DISPLAY for GUI applications (OpenCV windows)
export DISPLAY=:0

source /opt/ros/humble/setup.bash
source /moveit_task_constructor/install/setup.bash
if [ -f install/setup.bash ]; then
    source install/setup.bash
fi


echo "Provided arguments: $@"

exec $@