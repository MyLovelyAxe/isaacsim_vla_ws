#!/bin/bash

# ====================================
# Give target state to /joint_command
# ====================================

# default home position:
# [0.0009, 0.0258, 0.0, 0.0, 0.0, -0.0065]

# 1st action
# [-0.726, 0.251, 0.321, 0.364, 1.441, 2.776]

# 2nd action
# [-0.01296126,  1.7586157 , -1.5670338 , -0.64176947,  0.10169099, 0.89535266]

# 3rd action
# [ 0.00373928,  1.7602636 , -1.5617038 , -0.6458767 ,  0.09015466, 0.89725024]

# then get stuck at above position

ros2 topic pub /joint_command sensor_msgs/msg/JointState "
header:
  stamp: {sec: 0, nanosec: 0}
  frame_id: ''
name:
  ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
position:
  [ 0.0037, 1.7, -1.5, 0.9, -1.57, 0.0]
" -1



