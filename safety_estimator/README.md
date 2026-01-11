
build `/safety_estimator` as a importable package:

```bash
conda activate smolvla
cd ~/isaacsim_vla_ws/safety_estimator
pip install -e .
```

training

```bash
conda activate smolvla
cd ~/isaacsim_vla_ws/safety_estimator
python train.py
```


online inference

```bash
conda activate smolvla
cd ~/isaacsim_vla_ws/safety_estimator
python run_estimator.py
```

problems to improve:

1. the dataset is not idealy synchronized, i.e. in the pair [Q(t), A(t)], Q(t) should be result from previous action A(t-1), not the current action A(t), since execution of target action needs time, but that would be extra work to deal with this, so be it for now

2. proposed action and executed action can be different, but for now to simplify, just make them the same

3. dataset splits only trajectories now, not the timetamps. If the number of .npy trajectories is less than 10, valid or test set might be empty.

4. whether exploring where to reach the object should be considered as unsafe, or it is only safe under a tolerance of exploration, e.g. explore for 20 timestamps, if still doesn't get the right path, then should be unsafe

5. risk rule 3, i.e. stuck, can detect stuck, but can't detect unreasonable proposed action. If the proposed action is messy but the robot arm can still follow without getting stuck, then it is still labeled as safe. Replay this one for example: 20260110_091746

6. risk rule 3 for now can't distinguish whether the large change comes from warm-up or getting stuck. I just manually exculde the first 30 timestamps for warm-up, since it always happens but not fixed before 30 timestamps in every trajectory.

7. risk rule 3 can't detect moveable obstacle, e.g. if robot arm touches the cube, moves it but doesn't pick it up, there might be slight large change for difference between Q and A, but also might be considered as safe as well.