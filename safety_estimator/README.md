
#### Build package

Build `/safety_estimator` as an importable package:

```bash
conda activate smolvla
cd ~/isaacsim_vla_ws/safety_estimator
pip install -e .
```

#### Train

Train safety estimator network:

```bash
conda activate smolvla
cd ~/isaacsim_vla_ws/safety_estimator
python train.py
```

The trained checkpoint `.pt` and a log of training and validation process in `.json` will be stored under `~/isaacsim_vla_ws/safety_estimator/safety_estimator/checkpoints`.

#### Test

Test a pretrained checkpoint:

```bash
conda activate smolvla
cd ~/isaacsim_vla_ws/safety_estimator
python train.py --test --load_model_pt <checkpoint_path.pt>
```

#### Online inference

Receive live-stream history window of joint states and executed actions from zmq socket, and estimate a risk score with a pretrained checkpoint:

```bash
conda activate smolvla
cd ~/isaacsim_vla_ws/safety_estimator
python run_safety_estimator.py
```
