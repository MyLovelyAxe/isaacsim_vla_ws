
Run `add_cubes.py` with:

```bash
cd ~/isaacsim
./python.sh /home/hardli/isaacsim_vla_ws/isaacsim_scene/add_cubes.py
```

Run `tmp_max_joint_vel.py` with:

```bash
cd ~/isaacsim
./isaac-sim.sh --exec ~/isaacsim_vla_ws/isaacsim_scene/scripts/tmp_max_joint_vel.py
```

Type R key on keyboard in the viewpoint window of Isaac Sim, to reset the cube's position without restarting the scene:

```bash
cd ~/isaacsim
./isaac-sim.sh --exec ~/isaacsim_vla_ws/isaacsim_scene/scripts/reset_cube_online.py
```