

# PS: add a sigmoid on direct output of network, since loss function BCEWithLogitsLoss
import os
import io
import zmq
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from safety_estimator.utils.tools import (
    load_model,
    load_model_hyperparam,
    load_normalizer,
)
from safety_estimator.utils.metrics import sigmoid
from safety_estimator.model.safety_estimator_network import SafetyEstimatorNetwork


INPUT_HISTORY_SOCKET = "tcp://localhost:5557"
EMPTY_SIGNAL_SOCKET = "tcp://127.0.0.1:5558"


parser = argparse.ArgumentParser(description='Safety estimator')
parser.add_argument(
    '--load_model_pt', 
    type=Path, 
    default=Path("/home/hardli/isaacsim_vla_ws/safety_estimator/safety_estimator/checkpoints/exp-20260111_234524-N10-M10-D16-WL30-MST0.8-E20.pt"),
    help='The path of .pt of a pretrained model for testing.',
)
parser.add_argument(
    '--device_choice', 
    type=str, 
    default="cuda:0", 
    help='The device choice for training or inference, i.e. cpu or gpu.',
)


#### source of history input ###

history_context = zmq.Context()
history_socket = history_context.socket(zmq.SUB)
history_socket.connect(INPUT_HISTORY_SOCKET)
history_socket.setsockopt(zmq.SUBSCRIBE, b"")


### signal for emtpy VLA action chunk ###

risk_label_buffer = deque(maxlen=10)
signal_context = zmq.Context()
signal_socket = signal_context.socket(zmq.PUB)
signal_socket.bind(EMPTY_SIGNAL_SOCKET)


### online risk score indicator ###

plt.ion()  # interactive mode ON
fig, ax = plt.subplots(figsize=(1, 6))
bar = ax.bar([0], [0.0], width=0.2)[0]
ax.axhline(0.5, color="black", linestyle="--", linewidth=2) # threshold
ax.set_ylim(0.0, 1.0)
ax.set_title("Risk Score", fontsize=20)
plt.show()

def risk_to_color(risk: float):
    """Update the color of the safety estimator bar."""

    if risk < 0.4:
        return "green"
    elif risk < 0.7:
        return "orange"
    else:
        return "red"


def update_risk_bar(risk_value):
    """
    risk_value: float in [0, 1]
    """
    bar.set_height(risk_value)
    bar.set_color(risk_to_color(risk_value))
    fig.canvas.draw()
    fig.canvas.flush_events()

update_risk_bar(0.0) # warm-up


time.sleep(2) # give subscribers a short time to connect

if __name__ == '__main__':

    args = parser.parse_args()

    # select device
    # TODO: move this to utils/tools.py
    if args.device_choice == "cuda:0":
        if torch.cuda.is_available():
            print(f"GPU available, choose {args.device_choice}.")
            device = torch.device(args.device_choice)
        else:
            print("No cuda GPU available, force to use CPU.")
            device = torch.device("cpu")
    elif args.device_choice == "cpu":
        print("Choose to use CPU.")
        device = torch.device("cpu")

    # create model
    history_len, encoded_dim = load_model_hyperparam(
        load_model_pt=args.load_model_pt,
    )
    model = SafetyEstimatorNetwork(
        history_len=history_len,
        encoded_dim=encoded_dim,
    )
    model = load_model(
        model=model,
        device=device,
        load_model_pt=args.load_model_pt,
    )

    # prepare normalizers
    q_normalizer, a_normalizer = load_normalizer(
        load_model_pt=args.load_model_pt,
        device=torch.device("cpu"), # normalize input on cpu first, then infer on gpu
    )


    while True:

        ### Get live-stream history data ###

        payload = history_socket.recv()   # one npz blob
        buf = io.BytesIO(payload)
        history = np.load(buf)
        # ts = float(history["ts"][0])
        q_history = torch.tensor(history["joint_states_history"], dtype=torch.float32)
        a_history = torch.tensor(history["executed_actions_history"], dtype=torch.float32)
        prop_next_a = torch.tensor(history["prop_next_actioin"], dtype=torch.float32)

        ### Prepare input for network ###

        # noramlize input
        # N = history_len loaded from .pt
        norm_q_history = q_normalizer(q_history) # shape (N, 6)
        norm_a_history = a_normalizer(a_history) # shape (N, 6)
        norm_history = torch.concat([norm_q_history, norm_a_history], dim=1) # shape (N, 12)
        norm_prop_act = a_normalizer(prop_next_a) # shape (6,)

        # create batches, i.e. batch_size = 1
        history_batch = norm_history.unsqueeze(0) # shape (1, N, 12)
        prop_act_batch = norm_prop_act.unsqueeze(0) # shape (1, 6)

        # move input to device
        history_batch = history_batch.to(device)
        prop_act_batch = prop_act_batch.to(device)

        ### Estimate risk score ###

        risk_score = model(history_batch, prop_act_batch)
        logits = risk_score.detach().cpu().numpy().reshape(-1)
        # probability of risk or not, high: risk, low: not risk
        risk_prob = sigmoid(logits)
        # label of risk, 1: risk, 0: not risk
        risk_label = (risk_prob >= 0.5).astype(np.int32)
        # add risk label into buffer
        risk_label_buffer.append(risk_label)
        print(f"risk prob: {risk_prob}, risk label: {risk_label}")
        # visualize the risk score as a bar
        update_risk_bar(float(risk_prob[0]))

        ### Signal for VLA: empty action chunk or not ###

        if sum(risk_label_buffer) >= 8:
            # if at least 8 proposed actions are unsafe
            # empty the current action queue, 
            # let VLA model togenerate a new chunk
            empty = np.array([1.0],dtype=np.float32)
        else:
            # otherwise keep the current chunk
            empty = np.array([0.0],dtype=np.float32)

        # send emtpy signal to VLA model
        signal_socket.send(empty.tobytes())
