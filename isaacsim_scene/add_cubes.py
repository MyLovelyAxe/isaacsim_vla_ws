import numpy as np

from isaacsim.simulation_app import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

app = SimulationApp({"headless": False})

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

world.scene.add(
    DynamicCuboid(
        prim_path="/World/red_cube",
        name="cube",
        position=np.array([0.0, 0.0, 2.0]),
        size=0.5,
        color=np.array([1.0, 0.0, 0.0]),  # use 0~1, not 0~255
    )
)

world.reset()

while app.is_running():
    world.step(render=True)

app.close()