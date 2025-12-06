# isaacsim_vla_ws: ROS2 workspace for VLA model with Isaac Sim

This ROS2 workspace provides packages to implement message communication between Isaac Sim and VLA model, including images, current and target joint states of robot arm. It also contains some utility commands for Isaac Sim.

<!-- ![Description](media/isaacsim_vla_pipeline_demo.gif) -->

## Table of Contents

- [Requirements](#requirements)
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)

---

## Requirements

This package is tested on the following environment configuration:

- Ubuntu22.04
- [ROS2 humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- System python 3.10
- [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)

---

## About

This workspace works for 2 aspects.

#### 1. Utility commands for Isaac Sim

This workspace also provides helpful commands to:

1. Start Isaac Sim GUI with specified robot arm USD;

2. Test whether image topics receive images from simulated camera, by visualization in Rviz2;

3. Quickly test controller of robot arm, by giving one-time target state;

#### 2. Message exchange

The following diagram describes how package `vla_center` exchange messages, including images, joint states of robot arm, between Isaac Sim and VLA model, which in this project is [Lerobot SmolVLA](git@github.com:MyLovelyAxe/lerobot.git):

<!-- <img src="media/isaacsim_vla_msg_diagram.drawio.svg" width="800"/> -->

1. the topic `/camera1_rgb` receives RGB images from simulated camera in Isaac Sim, and sends them to **obs ZMQ socket**;

2. the topic `/joint_states` receives current joint states from simulated robot arm [SO100](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html) in Isaac Sim, and sends them to **obs ZMQ socket**;

3. the **obs ZMQ socket** sends the RGB images and current joint states to VLA model, i.e. [SmolVLA](git@github.com:MyLovelyAxe/lerobot.git);

4. the **action ZMQ socket** receives the result action values for each joint of robot arm from VLA model, and sends them to topic `/joint_commmand`, as target state in Isaac Sim;

---

## Installation

Please make sure the following setup is already done before installing this package:

- [ROS2 humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) installed in system environment on Ubuntu22

- [Isaac Sim GUI 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)

```bash
# Clone the workspace
cd
git clone git@github.com:MyLovelyAxe/isaacsim_vla_ws.git

# Build package
cd .. # return to workspace
colcon build --packages-select vla_center --symlink-install
```

## Usage

#### 1. Utility commands for Isaac Sim

**Terminal 1**: Start Isaac Sim GUI

```bash
cd ~/isaacsim_vla_ws/bash
source setup_isaacsim.sh
./start_isaac_sim.sh
```

This starts Isaac Sim GUI with pre-defined USD for robot arm model SO100, including action graphs to publish current joint states and camera images;

**Terminal 2**: Test image topics 

```bash
cd ~/isaacsim_vla_ws/bash
source setup_systemros.sh
./test_image_topic_rviz.sh
```

This starts Rviz2 with pre-defined `.rviz` config, which visualizes images from image topics defined in Isaac Sim;

**Terminal 3**: Test robot arm controller

```bash
cd ~/isaacsim_vla_ws/bash
source setup_systemros.sh
./give_joint_command.sh
```

This quickly test the controller node of action graph for the robot arm model, by giving one-time target state, which is specified for robot SO100;

#### 2. Message exchange

TODO: start vla_center node

TODO: start smolvla according to forked smolvla repo