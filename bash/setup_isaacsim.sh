#!/bin/bash

# ================================
# Set ROS 2 environment variables
# ================================

# choose RMW
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# make sure all terminal share the same ros domain id
export ROS_DOMAIN_ID=0

# make sure ROS on multiple devices can communicate
export ROS_LOCALHOST_ONLY=0
