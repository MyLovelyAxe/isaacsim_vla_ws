#!/bin/bash

# ================================
# Resolve home directory paths
# ================================

USD_PATH="$HOME/isaacsim_vla_ws/config/vla_franka_2cam.usd"
LAYOUT_PATH="$HOME/isaacsim_vla_ws/config/franka_action_graph_2cam_layout.json"

# ================================
# Start Isaac Sim with USD + layout
# ================================

cd "$HOME/isaacsim"

./isaac-sim.sh --exec "open_stage.py $USD_PATH"

