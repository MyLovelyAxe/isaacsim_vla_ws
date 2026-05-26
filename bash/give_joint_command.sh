#!/bin/bash

# ====================================
# Give target state to /joint_command
# ====================================

new_calib=(0.0 -0.875 0.845 0.83 0.05 -0.17)
old_calib=(0.0 0.8 -0.8 0.5 0.05 0.0)

calib=("${new_calib[@]}")

while [[ $# -gt 0 ]]; do
  case $1 in
    --new-calib) calib=("${new_calib[@]}"); shift ;;
    --old-calib) calib=("${old_calib[@]}"); shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

ros2 topic pub /joint_command sensor_msgs/msg/JointState "
header:
  stamp: {sec: 0, nanosec: 0}
  frame_id: ''
name:
  ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
position:
  [${calib[0]}, ${calib[1]}, ${calib[2]}, ${calib[3]}, ${calib[4]}, ${calib[5]}]
" -1

