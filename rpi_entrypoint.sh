#!/bin/bash

set -e

source /opt/ros/humble/setup.bash
if [ -f install/setup.bash ]; then
    source install/setup.bash
fi


echo "Provided arguments: $@"

exec $@