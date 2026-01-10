import os
import uuid
import torch
import argparse
from pathlib import Path
from datetime import datetime
from safety_estimator.data.dataloader import TrajectoryDataLoader
from safety_estimator.experiment.exp_safety_estimator import SafetyEsimatiorExp

parser = argparse.ArgumentParser(description='Safety estimator')

parser.add_argument(
    '--dataset_path', 
    type=Path, 
    default=Path("~/isaacsim_vla_ws/record").expanduser(), 
    help='The folder that contains the recorded trajectories.',
)
parser.add_argument(
    '--history_len', 
    type=int, 
    default=10, 
    help='The window length of history of joint states Q and executed actions A, i.e. t-N, ..., t.',
)
parser.add_argument(
    '--future_len', 
    type=int, 
    default=10, 
    help='The horizon length of future of joint states Q and executed actions A, i.e. t, ..., t+M.',
)
parser.add_argument(
    '--train_set_ratio', 
    type=float, 
    default=0.8, 
    help='Ratio of training set in the whole dataset.',
)
parser.add_argument(
    '--valid_set_ratio', 
    type=float, 
    default=0.1, 
    help='Ratio of validation set in the whole dataset.',
)
parser.add_argument(
    '--test_set_ratio', 
    type=float, 
    default=0.1, 
    help='Ratio of testing set in the whole dataset.',
)
parser.add_argument(
    '--batch_size', 
    type=int, 
    default=32, 
    help='Number of samples for one single batch.',
)
parser.add_argument(
    '--train_epoch', 
    type=int, 
    default=20, 
    help='Number of epochs to train the network.',
)
parser.add_argument(
    '--device_choice', 
    type=str, 
    default="cuda:0", 
    help='The device choice for training or inference, i.e. cpu or gpu.',
)
parser.add_argument(
    '--learning_rate', 
    type=float, 
    default=3e-4, 
    help='Learning rate for optimizer to update gradients.',
)
parser.add_argument(
    '--output_path', 
    type=Path, 
    default=Path("~/isaacsim_vla_ws/safety_estimator/safety_estimator/checkpoints").expanduser(), 
    help='The path to store the trained checkpoints.',
)
parser.add_argument(
    '--log_id', 
    type=str, 
    help='Unique ID for one single experiment.',
)
parser.add_argument(
    '--examine_mode', 
    type=bool, 
    default=False, 
    help='In examine mode, the returned samples are not yet normalized.',
)
parser.add_argument(
    '--verbose', 
    type=bool, 
    default=False, 
    help='If True: log the process of creating dataloader.',
)


if __name__ == '__main__':

    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    args = parser.parse_args()

    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_id = f"exp-{time}-N{args.history_len}-M{args.future_len}-E{args.train_epoch}-{uuid.uuid4()}"

    exp = SafetyEsimatiorExp(
        dataset_path=Path("~/isaacsim_vla_ws/record").expanduser(),
        history_len=args.history_len,
        future_len=args.future_len,
        train_set_ratio=args.train_set_ratio,
        valid_set_ratio=args.valid_set_ratio,
        test_set_ratio=args.test_set_ratio,
        batch_size=args.batch_size,
        train_epoch=args.train_epoch,
        device_choice=args.device_choice,
        learning_rate=args.learning_rate,
        output_path=args.output_path,
        log_id=args.log_id,
        examine_mode=args.examine_mode,
        verbose=args.verbose,
    )

    before_train = datetime.now().timestamp()
    print("===================Normal-Start=========================")
    exp.train()
    after_train = datetime.now().timestamp()
    print(f'Training took {(after_train - before_train) / 60} minutes')
    print("===================Normal-End=========================")
