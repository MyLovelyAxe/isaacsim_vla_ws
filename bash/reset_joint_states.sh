#!/bin/bash

# ====================================
# Give target state to /joint_command
# ====================================

ros2 topic pub /joint_command sensor_msgs/msg/JointState "
header:
  stamp: {sec: 0, nanosec: 0}
  frame_id: ''
name:
  ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
position:
  [0.0009, 0.0258, 0.0, 0.0, 0.0, -0.0065]
" -1



