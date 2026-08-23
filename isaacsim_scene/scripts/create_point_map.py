import numpy as np
import omni.usd
from pxr import UsdGeom, UsdLux, Gf

stage = omni.usd.get_context().get_stage()

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(3000.0)
dome.CreateColorAttr(Gf.Vec3f(0.1, 0.1, 0.1))

data: dict = np.load("/home/hardli/Projects/vggt/input_images_20260823_104514_694305/predictions.npz")

# # use point cloud
# points = data["world_points"]
# conf   = data["world_points_conf"]
# images = data["images"]

# use both depth map
points = data["world_points_from_depth"]
conf = data.get("depth_conf", np.ones_like(points[..., 0]))
images = data["images"]


# Convert [N, 3, H, W] -> [N, H, W, 3]
if images.ndim == 4 and images.shape[1] == 3:
    images = np.transpose(images, (0, 2, 3, 1))

# Example confidence threshold — tune this
# mask = conf > 1.0
conf_threshold = np.percentile(conf, 50.0) # 50% confidence value
print(f"Select 50% percentile confidence threshold: {conf_threshold}")
mask = (conf >= conf_threshold) & (conf > 1e-5)

points = points[mask]

# If images are [N,H,W,3]
colors = images[mask]

# Convert uint8 RGB -> [0,1]
if colors.max() > 3.0:
    colors = colors.astype(np.float32) / 255.0

# IMPORTANT for first test:
# reduce number of points so Isaac doesn't get flooded
points = points[::5]
colors = colors[::5]

stage = omni.usd.get_context().get_stage()

# Parent prim: lets you transform the whole reconstructed scene
root = UsdGeom.Xform.Define(stage, "/World/VGGTScene")

# Point cloud prim
pc = UsdGeom.Points.Define(stage, "/World/VGGTScene/PointCloud")

pc.CreatePointsAttr(
).Set([
    Gf.Vec3f(float(x), float(y), float(z))
    for x, y, z in points
])

# Physical/rendered diameter of each point
pc.CreateWidthsAttr().Set([0.003] * len(points))

# Per-point RGB
color_primvar = pc.CreateDisplayColorPrimvar(
    interpolation="vertex"
)

color_primvar.Set([
    Gf.Vec3f(float(r), float(g), float(b))
    for r, g, b in colors
])