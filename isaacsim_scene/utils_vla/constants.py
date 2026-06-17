from enum import Enum
from pathlib import Path

###### Paths ######

ISAACSIM_VLA_WS_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_USD_PATH = ISAACSIM_VLA_WS_ROOT / "config/vla_so101_new_calib_2cam.usd"


###### Prim Paths ######

WORLD_PRIM = "/World"
SO101_NEW_CALIB_PRIM = f"{WORLD_PRIM}/so101_new_calib"
SO101_NEW_CALIB_JOINTS_PRIM = f"{SO101_NEW_CALIB_PRIM}/joints"


###### Cameras ######

DEV_V4L_BY_PATH = "/dev/v4l/by-path"
LAPTOP_LEFT_USB_1 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:1:1.0-video-index0"
LAPTOP_RIGHT_USB_1 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2:1.0-video-index0"
# usb hub: connect to laptop right usb 1
# NOTE: socket ID 1-4 starting from the indicator light on the usb hub
LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_1 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2.1:1.0-video-index0"
LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_2 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2.2:1.0-video-index0"
LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_3 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2.3:1.0-video-index0"
LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_4 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2.4:1.0-video-index0"

class So101Camera(str, Enum):
    """The usb path of camera on usb hub for SO101 robot arm system."""
    wrist_cam = LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_3
    side_cam = LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_4


###### Calibration Names ######

SO101_NEW_CALIB = "so101_new_calib"
SO101_OLD_CALIB = "so101_old_calib"
SO101_FOLLOWER_NEW_CALIB = "so101_follower_new_calib"
SO101_FOLLOWER_OLD_CALIB = "so101_follower_old_calib"
SO101_LEADER_NEW_CALIB = "so101_leader_new_calib"
SO101_LEADER_OLD_CALIB = "so101_leader_old_calib"


###### Hardware ######

# Controlling
HOME_MOVE_HZ = 50
HOME_SPEED = 8.0  # normalized position units per second; tune lower/slower
HOME_TOL = 0.5

# Robot ports
SO101_FOLLOWER_PORT_ID  = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF218449-if00"
SO101_LEADER_PORT_ID    = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF219896-if00"

# Joints
JOINT_NAME_ORDER = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]


###### Default Joint States ######

# Before starting inference
HOME_JOINT_STATE = {
    SO101_OLD_CALIB: {
        'shoulder_pan.pos': 0.0,
        'shoulder_lift.pos': 0.0,
        'elbow_flex.pos': 0.0,
        'wrist_flex.pos': 40.0,
        'wrist_roll.pos': 0.0,
        'gripper.pos': 0.0,
    },
    SO101_NEW_CALIB: {
        'shoulder_pan.pos': 0.0,
        'shoulder_lift.pos': 0.0,
        'elbow_flex.pos': 0.0,
        'wrist_flex.pos': 0.0,
        'wrist_roll.pos': 0.0,
        'gripper.pos': 0.0,
    },
}

# After finishing inference and shut down the robot
RETURN_JOINT_STATE = {
    SO101_OLD_CALIB: dict(), # TODO
    SO101_NEW_CALIB: {
        'shoulder_pan.pos': -4.0,
        'shoulder_lift.pos': -99.0,
        'elbow_flex.pos': 98.0,
        'wrist_flex.pos': -80.0,
        'wrist_roll.pos': -55.0,
        'gripper.pos': 0.0,
    },
}


###### Inference Model ######

MODEL_ID = "lerobot/smolvla_base"


###### ZeroMQ Sockets ######

INPUT_OBSERVATION_SOCKET = "tcp://localhost:5556"
OUTPUT_ACTION_SOCKET = "tcp://127.0.0.1:5555"
EMPTY_SIGNAL_SOCKET = "tcp://localhost:5558" # signal to empty action chunk queue


###### Dataset Features ######

# Feature structure of output action
ACTION_FEATURES = {
    'action': {
        'dtype': 'float32',
        'shape': (6,),
        'names': [
            'shoulder_pan.pos', 
            'shoulder_lift.pos', 
            'elbow_flex.pos', 
            'wrist_flex.pos', 
            'wrist_roll.pos', 
            'gripper.pos',
        ],
    },
}

# Feature structure of intput observation
OBS_FEATURES = {
    'observation.state': {
        'dtype': 'float32',
        'shape': (6,),
        'names': [
            'shoulder_pan.pos', 
            'shoulder_lift.pos', 
            'elbow_flex.pos', 
            'wrist_flex.pos', 
            'wrist_roll.pos', 
            'gripper.pos',
        ],
    },
    'observation.images.camera1': {
        'dtype': 'video',
        'shape': (480, 640, 3),
        'names': [
            'height', 
            'width', 
            'channels',
        ],
    },
    'observation.images.camera2': {
        'dtype': 'video',
        'shape': (480, 640, 3),
        'names': [
            'height', 
            'width', 
            'channels',
        ],
    },
}

DATASET_FEATURES = {**ACTION_FEATURES, **OBS_FEATURES}


###### Simulation ######

# radian range of so100 joints in Isaac Sim
SIMULATION_RANGE = {
    SO101_OLD_CALIB: {
        'shoulder_pan.pos'  : {'sim_min': -2.0,     'sim_max': 2.0  },
        'shoulder_lift.pos' : {'sim_min': 0.0,      'sim_max': 3.5  },
        'elbow_flex.pos'    : {'sim_min': -3.142,   'sim_max': 0.0  },
        'wrist_flex.pos'    : {'sim_min': -2.5,     'sim_max': 1.2  },
        'wrist_roll.pos'    : {'sim_min': -3.142,   'sim_max': 3.142},
        'gripper.pos'       : {'sim_min': -0.2,     'sim_max': 2.0  },
    },
    SO101_NEW_CALIB: {
        'shoulder_pan.pos'  : {'sim_min': -1.92,    'sim_max': 1.92 },
        'shoulder_lift.pos' : {'sim_min': -1.75,    'sim_max': 1.75 },
        'elbow_flex.pos'    : {'sim_min': -1.57,    'sim_max': 1.57 },
        'wrist_flex.pos'    : {'sim_min': -1.66,    'sim_max': 1.66 },
        'wrist_roll.pos'    : {'sim_min': -2.74,    'sim_max': 2.84 },
        'gripper.pos'       : {'sim_min': -0.17,    'sim_max': 1.74 },
    },
}