import omni.usd

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/Cube")

print()

for attr in prim.GetAttributes():
    print(attr.GetName(), attr.GetTypeName(), attr.Get())

