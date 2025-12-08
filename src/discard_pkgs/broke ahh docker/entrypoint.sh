#!/bin/bash

set -e

source /opt/ros/humble/setup.bash
cd wilson

echo "Provided arguments: $@"

exec $@