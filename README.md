# isaacsim_vla_ws: ROS2 workspace for VLA model with Isaac Sim and Jetson Orin Nano

This ROS2 workspace provides packages to implement message communication between Isaac Sim and VLA model (which can be deployed either on host machine or Jetson Orin Nano), including images, current and target joint states of robot arm. It also contains some utility commands for Isaac Sim.

![Description](media/safety_estimator_live_demo.gif)

## Table of Contents

- [Requirements](#requirements)
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Open tasks](#open_tasks)

---

## Requirements

This package is tested on the following environment configuration:

On laptop or workstation:

- Ubuntu22.04
- [ROS2 humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- System python 3.10
- [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
- Anaconda3
- RTX3080Ti with driver 575.57.08
- CUDA version 12.9

On Jetson Orin Nano:

- Jetson Orin Nano [8GB developer kit version]
- Jetpack 6.2.1
- [ROS2 humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- Docker

---

## About

This workspace works for 3 aspects:

#### 1. Utility commands for Isaac Sim

This workspace provides helpful commands to:

1. Start Isaac Sim GUI with specified robot arm USD;

2. Visulize the image topics messages from simulated camera in Rviz2;

3. Quickly test controller of robot arm, by giving one-time target state;

#### 2. Message exchange

The following diagram describes how messages are exchanged between Isaac Sim and VLA model ([LeRobot SmolVLA](https://huggingface.co/blog/smolvla) in this project) with management of package `vla_center`, including images, joint states of robot arm, between Isaac Sim and VLA model:

<img src="media/vla_center_safety_estimator_diagram.drawio.svg" width="800"/>

1. the topica `/camera1_rgb` and `/camera2_rgb` receives RGB images from simulated camera in Isaac Sim, and sends them to **obs ZMQ socket**;

2. the topic `/joint_states` receives current joint states from simulated robot arm [SO100](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html) in Isaac Sim, and sends them to **obs ZMQ socket**;

3. the **obs ZMQ socket** sends the RGB images and current joint states to VLA model, i.e. [SmolVLA](git@github.com:MyLovelyAxe/lerobot.git);

4. the **act ZMQ socket** receives the result action values for each joint of robot arm from VLA model, and sends them to topic `/joint_commmand`, as target state in Isaac Sim;

5. the buffers in action sender node stores a short history from topic `/joint_states` and returned actions from SmolVLA, used for control action generation of SmolVLA by a safety estiamtor;

#### 3. Safety estimation

The VLA model this repository uses, i.e. [LeRobot SmolVLA](https://huggingface.co/blog/smolvla), generates an action chunk every time when it receives new observation. And only when the entire action chunk is returned, will it generate a new chunk. However, the chunk of actions based on the observation from one single timestamp might lead to noisy actions, accumulated execution error, and unsafe behavior within the chunk.

This repository provides a neural network-based safety estimator, which estimates a risk score based on a short history to decide whether it is safe to take a new proposed action from VLA model. 

The details about training and testing can be found under `~/isaacsim_vla_ws/safety_estimator`. 

> **Note 1**:
> The `vla_center` package and VLA model can be deployed on both the same host machine with Isaac Sim, or Jetson Orin Nano.

> **Note 2**:
> The ZMQ sockets involved in this project:
> tcp://127.0.0.1:5555: action commands from VLA
> tcp://127.0.0.1:5556: observation (images + joint states) from Isaac Sim
> tcp://127.0.0.1:5557: history (joint states + executed actions) for safety estimator
> tcp://127.0.0.1:5558: signal to empty action queue of VLA

---

## Installation

Either deploy **VLA model** on the host machine (e.g. laptop or workstation) or Jetson Orin Nano, make sure [Isaac Sim GUI 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) is installed on the **host machine**.

<details>
<summary>If deploy VLA model on host machine:</summary>

#### 1. ROS2

Install **ROS2 humble** on host machine with Ubuntu22 in the system path, following ROS2 [official installation instruction](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

#### 2. Clone and build this repository

```bash
# Clone the workspace
cd
git clone git@github.com:MyLovelyAxe/isaacsim_vla_ws.git

# Build vla_center package
cd ~/isaacsim_vla_ws
colcon build --packages-select vla_center --symlink-install
```

#### 3. Install LeRobot for VLA model

Refer to the `README` of [branch `camera/zmq_socket` of this fork lerobot repository](https://github.com/MyLovelyAxe/lerobot/tree/camera/zmq_socket), follow the installation steps of host machine.

</details>

<details>
<summary>If deploy VLA model on Jetson Orin Nano:</summary>

#### 1. ROS2

Firstly, flash [Jetpack 6.2](https://www.jetson-ai-lab.com/tutorials/initial-setup-jetson-orin-nano/) on Jetson Orin Nano.

Then, install **ROS2 humble** on Jetson Orin Nano with Jetson Linux in the system path (putside any docker containers), following ROS2 [official installation instruction](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

#### 2. Clone and build this repository

```bash
# Clone the workspace
cd
git clone git@github.com:MyLovelyAxe/isaacsim_vla_ws.git

# Build vla_center package
cd ~/isaacsim_vla_ws
colcon build --packages-select vla_center --symlink-install
```

#### 3. Install LeRobot for VLA model

Refer to the `README` of [branch `camera/zmq_socket` of this fork lerobot repository](https://github.com/MyLovelyAxe/lerobot/tree/camera/zmq_socket), follow the installation steps of Jetson Orin Nano.

</details>


## Usage

#### 1. Test Isaac Sim

> **Attention:**
> The `config/vla_so101_2cam.usd` for managing all robot, cameras and action graphs is based on a **local** reference of so100 robot. It refers to `config/Collected_so100` for the so100 prim, which is also included in this repo. You can also collect this Asset in Isaac Sim GUI, refer to [Isaac Sim instruction](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots.html).


**Terminal 1**: Start Isaac Sim GUI (always on host machine)

```bash
cd ~/isaacsim_vla_ws/bash
source setup_isaacsim.sh
./start_isaac_sim.sh
```

This starts Isaac Sim GUI with pre-defined USD for robot arm model SO100, including action graphs to publish current joint states and camera images;

> **Remember:**
> Press **PLAY** button in Isaac Sim to start simulation!


**Terminal 2**: Test image topics (on either host machine or Jetson)

```bash
cd ~/isaacsim_vla_ws/bash
source setup_systemros.sh
./test_image_topic_rviz.sh
```

This starts Rviz2 with pre-defined `.rviz` config, which visualizes images from image topics defined in Isaac Sim;

**Terminal 3**: Test robot arm controller (on either host machine or Jetson)

```bash
cd ~/isaacsim_vla_ws/bash
source setup_systemros.sh
./give_joint_command.sh # give new target state, --new-calib gives target under new calibration, similar logic for --old-calib. New calib by default. 
./reset_joint_states.sh # reset to initial state, --new-calib gives target under new calibration, similar logic for --old-calib. New calib by default. 
```

This quickly test the controller node of action graph for the robot arm model, by giving one-time target state or reset to initial state, which is specified for robot SO100;

#### 2. Run VLA pipeline

**Terminal 1**: Start Isaac Sim

Always start Isaac Sim GUI on the host machine, and press **PLAY** button to start simulation:

```bash
cd ~/isaacsim_vla_ws/bash
source setup_isaacsim.sh
./start_isaac_sim_so101_new_calib.sh
```

<details>
<summary>If run VLA model on the host machine:</summary>

</br>

**Terminal 2**: Start ROS2 nodes for message exchange (On host machine)

```bash
cd ~/isaacsim_vla_ws/
source bash/setup_systemros.sh
source install/setup.bash
ros2 launch vla_center send_obs_get_act.launch.py
```

**Terminal 3**: Start VLA model (On host machine)

```bash
conda activate smolvla
cd ~/lerobot/isaacsim_sim2real/scripts
python so101_follower_smolvla.py
```

</details>

<details>
<summary>If run VLA model on Jetson Orin Nano:</summary>

</br>

**Terminal 2**: Start ROS2 nodes for message exchange

On Jetson Orin Nano, in the host **outside** the container:

```bash
cd ~/isaacsim_vla_ws/
source bash/setup_systemros.sh
source install/setup.bash
ros2 launch vla_center send_obs_get_act.launch.py
```

**Terminal 3**: Start VLA model

On Jetson Orin Nano, **inside** the container:

```bash
docker start -ai smolvla_pytorch27_container
cd /opt/lerobot/examples/tutorial/smolvla
python so101_follower_sim2real.py
```

</details>


#### 3. Safety estimation

The following functions are for safety estimation, which also relies on Isaac Sim. So firstly start Isaac Sim in one terminal, then follow the steps for different usages. 

For now, the safety estimatoronly only runs on host machine, not on Jetson Orin Nano

**Terminal 1**: start isaac sim (always on host machine)

```bash
cd ~/isaacsim_vla_ws/bash
source setup_isaacsim.sh
./start_isaac_sim_so101_new_calib.sh
```

Pres **Play** button.

<details>
<summary>Func 1: Record trajectories as dataset to train the safety estimator</summary>

</br>

**Terminal 2**: Start VLA model (On host machine)

```bash
conda activate smolvla
cd ~/lerobot/examples/tutorial/smolvla
python smolvla_zmq.py
```

**Terminal 3**: start recording node

```bash
cd ~/isaacsim_vla_ws/
source bash/setup_systemros.sh
source install/setup.bash
ros2 launch vla_center record_trajectory.launch.py
```

Check the Isaac Sim window for the robot's behavior, manually stop the process when you think the recording is done. The recorded `.npy` will be stored under `~/isaacsim_vla_ws/record`.

</details>

<details>
<summary>Func 2: Replay a recorded trajectory</summary>

</br>

**Terminal 2**: Start safety estimator

```bash
conda activate smolvla
cd ~/isaacsim_vla_ws/safety_estimator
python run_safety_estimator.py
```

**Terminal 3**: Select a recoreded `.npy` under `~/isaacsim_vla_ws/record`

```bash
cd ~/isaacsim_vla_ws/
source bash/setup_systemros.sh
source install/setup.bash
ros2 launch vla_center replay_record.launch.py 'npy_name:=20260109_154629.npy'
```

Manually stop the process when the recording is finished replaying.


</details>

<details>
<summary>Func 3: Online inference with trained safety estimator</summary>

</br>

**Terminal 2**: Start nodes to send history to estimator

```bash
cd ~/isaacsim_vla_ws/
source bash/setup_systemros.sh
source install/setup.bash
ros2 launch vla_center estimate_safety.launch.py
```

**Terminal 3**: Start VLA model

```bash
conda activate smolvla
cd ~/lerobot/isaacsim_sim2real/scripts
python so101_follower_smolvla.py
```

**Terminal 4**: Start safety estimator

```bash
conda activate smolvla
cd ~/isaacsim_vla_ws/safety_estimator
python run_safety_estimator.py
```

</details>


#### 4. sim2real synchronization

The following functions are for synchronization between simulated and real robot.

**Terminal 1**: start isaac sim

```bash
cd ~/isaacsim_vla_ws/bash
source setup_isaacsim.sh
./start_isaac_sim_so101_new_calib.sh
```

**Terminal 2**: Start node only for joint states

```bash
cd ~/isaacsim_vla_ws/
source bash/setup_systemros.sh
source install/setup.bash
ros2 launch vla_center exchange_joint_state.launch.py
```

<details>
<summary>Case 1: Execute a fixed trajectory</summary>

</br>

**Terminal 3**: Start the process to give a fixed trajectory:

```bash
conda activate smolvla
cd ~/lerobot/isaacsim_sim2real/scripts
python so101_follower_fix_traj.py --sim --real
```

</details>

<details>
<summary>Case 2: Use leader arm to teleoperate</summary>

</br>

**Terminal 3**: Start the process to use leader arm guide both real follower arm and simulated robot arm:

```bash
conda activate smolvla
cd ~/lerobot/isaacsim_sim2real/scripts
python so101_teleoperate.py --sim --real
```

</details>


## Open tasks

For now the perception-action loop with Isaac Sim and VLA model is setup, but only zero-shot SmolVLA is tested, the performance needs to be improved by fine-tuning SmolVLA. Therefore the on-going open tasks of this project include:

1. Build a pipeline to generate synthetic dataset which fits lerobot format

2. Fine-tune SmolVLA for some manipulation tasks
