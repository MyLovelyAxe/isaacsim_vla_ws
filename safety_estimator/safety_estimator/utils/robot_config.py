"""
This file defines configurations of robot model SO100 in Isaac Sim.
"""

# the angular range of each joint of robot SO100, in radian
SO100_JOINTS_LIMITS_RADIAN = {
    'shoulder_pan': {'min': -2.0, 'max': 2.0},
    'shoulder_lift': {'min': 0.0, 'max': 3.5},
    'elbow_flex': {'min': -3.142, 'max': 0.0},
    'wrist_flex': {'min': -2.5, 'max': 1.2},
    'wrist_roll': {'min': -3.142, 'max': 3.142},
    'gripper': {'min': -0.2, 'max': 2.0},
}