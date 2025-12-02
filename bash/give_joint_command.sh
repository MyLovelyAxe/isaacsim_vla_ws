#!/bin/bash

# ====================================
# Give target state to /joint_command
# ====================================

ros2 topic pub /joint_command sensor_msgs/msg/JointState "
header:
  stamp: {sec: 0, nanosec: 0}
  frame_id: ''
name:
  ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7','panda_finger_joint1','panda_finger_joint2']
position:
  [0.0, -1.16, 0.0, -2.3, 0.0, 1.6, 1.1, 0.4, 0.4]
" -1



