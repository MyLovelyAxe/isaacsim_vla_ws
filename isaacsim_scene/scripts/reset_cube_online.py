"""
Launch Isaac Sim Full 5.1, open existing USD, temporarily update joint max velocity,
and reset cube to initial pose when pressing R.

Run with:
cd ~/isaacsim
./isaac-sim.sh --exec /home/hardli/isaacsim_vla_ws/isaacsim_scene/scripts/test_reset.py
"""

from pathlib import Path
import sys

import carb.input
import omni.appwindow
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import open_stage
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

# Local imports
THIS_DIR = Path(__file__).resolve().parent
ISAACSIM_SCENE_ROOT = THIS_DIR.parent
sys.path.insert(0, str(ISAACSIM_SCENE_ROOT))

from utils_vla.constants import (
    DEFAULT_USD_PATH,
    JOINT_NAME_ORDER,
    SO101_NEW_CALIB_JOINTS_PRIM,
)

TMP_MAX_JOINT_VEL = 300.0

CUBE_PRIM_PATH = "/World/Cube"
CUBE_INITIAL_TRANSLATION = Gf.Vec3d(0.8, 0.05, 0.1)
CUBE_INITIAL_ORIENTATION = Gf.Quatd(1.0, 0.0, 0.0, 0.0)

print(f"Opening USD: {DEFAULT_USD_PATH}")
open_stage(str(DEFAULT_USD_PATH))

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("Failed to open USD stage")

# Temporarily update joint max velocity
for joint_name in JOINT_NAME_ORDER:
    joint_prim_path = f"{SO101_NEW_CALIB_JOINTS_PRIM}/{joint_name}"
    joint_prim = stage.GetPrimAtPath(joint_prim_path)

    if not joint_prim.IsValid():
        print(f"[WARN] Joint prim not found: {joint_prim_path}")
        continue

    physx_joint = PhysxSchema.PhysxJointAPI.Get(stage, joint_prim_path)
    if not physx_joint:
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)

    attr = physx_joint.GetMaxJointVelocityAttr()

    print(f"Joint {joint_name}:")
    print(f"  Original max velocity: {attr.Get()}")
    attr.Set(TMP_MAX_JOINT_VEL)
    print(f"  Temporary max velocity: {attr.Get()}")

# Cube xform setup
cube_prim = stage.GetPrimAtPath(CUBE_PRIM_PATH)
if not cube_prim.IsValid():
    raise RuntimeError(f"Cube prim not found: {CUBE_PRIM_PATH}")

xformable = UsdGeom.Xformable(cube_prim)

translate_op = None
orient_op = None

for op in xformable.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
        translate_op = op
    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
        orient_op = op

if translate_op is None:
    translate_op = xformable.AddTranslateOp()

if orient_op is None:
    orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

timeline = omni.timeline.get_timeline_interface()


def reset_cube():
    print("Resetting cube...")

    timeline.pause()

    # Do NOT clear xform op order; preserve scale and existing transform structure.
    translate_op.Set(CUBE_INITIAL_TRANSLATION)
    orient_op.Set(CUBE_INITIAL_ORIENTATION)

    rb_api = UsdPhysics.RigidBodyAPI.Get(stage, CUBE_PRIM_PATH)
    if rb_api:
        rb_api.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rb_api.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    omni.kit.app.get_app().update()
    timeline.play()

    print("Cube reset done")


# Keyboard callback
_input = carb.input.acquire_input_interface()
_appwindow = omni.appwindow.get_default_app_window()
_keyboard = _appwindow.get_keyboard()


def on_keyboard_event(event, *args):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.R:
            print("R pressed")
            reset_cube()
    return True


# Keep this global
_keyboard_sub = _input.subscribe_to_keyboard_events(_keyboard, on_keyboard_event)

# NOTE: 
# this starts the scene at the beginning of launching the app, 
# corresponding to manually clicking on "play" button
# timeline.play() 

print("Scene running. Click viewport, then press R to reset cube.")