import omni.usd

stage = omni.usd.get_context().get_stage()
joint_name = "wrist_roll"
prim = stage.GetPrimAtPath(f"/World/so101_new_calib/joints/{joint_name}")

print()
print(f"joint: {joint_name}")
print(prim.GetAppliedSchemas())

for attr in prim.GetAttributes():
    print(attr.GetName(), attr.GetTypeName(), attr.Get())
