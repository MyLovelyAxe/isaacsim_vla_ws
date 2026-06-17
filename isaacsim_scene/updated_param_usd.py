from isaacsim.core.utils.stage import open_stage
import omni.usd
from pxr import PhysxSchema

USD_PATH = "/home/hardli/isaacsim_vla_ws/config/vla_so101_new_calib_2cam.usd"

open_stage(USD_PATH)

stage = omni.usd.get_context().get_stage()

joint_path = "/World/so101_new_calib/joints/wrist_flex"
joint_prim = stage.GetPrimAtPath(joint_path)

# Get or apply PhysX joint API
physx_joint = PhysxSchema.PhysxJointAPI.Get(stage, joint_path)
if not physx_joint:
    physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)

attr = physx_joint.GetMaxJointVelocityAttr()

print("Old value:", attr.Get())

attr.Set(75.0)

print("New value:", attr.Get())

# Do NOT save stage