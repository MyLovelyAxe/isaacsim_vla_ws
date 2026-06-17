"""
This script launches Isaac Sim full 5.1 with existing USD and temporary maximum joint velocities.
Do not store in the Isaac Sim GUI.
"""

from isaacsim.core.utils.stage import open_stage
import omni.usd
from pxr import PhysxSchema

# NOTE: add these to make sure the root path of ~/isaacsim_vla_ws can be found by isaac sim python
from pathlib import Path
import sys
THIS_DIR = Path(__file__).resolve().parent
ISAACSIM_VLA_WS_ROOT = THIS_DIR.parent
sys.path.insert(0, str(ISAACSIM_VLA_WS_ROOT))

from utils_vla.constants import (
    JOINT_NAME_ORDER,
    SO101_NEW_CALIB_JOINTS_PRIM, 
    DEFAULT_USD_PATH,
)

TMP_MAX_JOINT_VEL = 300.0


print(f"DEFAULT_USD_PATH: {DEFAULT_USD_PATH}")
open_stage(str(DEFAULT_USD_PATH))

stage = omni.usd.get_context().get_stage()

for joint_name in JOINT_NAME_ORDER:

    joint_prim_path = f"{SO101_NEW_CALIB_JOINTS_PRIM}/{joint_name}"
    joint_prim = stage.GetPrimAtPath(joint_prim_path)

    # Get or apply PhysX joint API
    physx_joint = PhysxSchema.PhysxJointAPI.Get(stage, joint_prim_path)
    if not physx_joint:
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
    attr = physx_joint.GetMaxJointVelocityAttr()

    print(f"Attribue maximum joint velocity of ioint {joint_name}: ")
    print(f"Original value: {attr.Get()}")
    attr.Set(TMP_MAX_JOINT_VEL)
    print(f"Updated temporary value: {attr.Get()}")