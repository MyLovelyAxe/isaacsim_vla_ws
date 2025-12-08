import numpy as np

###### SO100 ######

SO100_JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]

# for calibration

SO100_SIGNS = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)

SO100_OFFSETS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)