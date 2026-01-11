import os
import torch
import argparse
from pathlib import Path
from datetime import datetime
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
    '--encoded_dim', 
    type=int, 
    default=16, 
    help='the dimension of encoded input for both history and proposed action.',
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
    '--warmup_len', 
    type=int, 
    default=30, 
    help='The number of timestamp at beginning of each trajectory to warm-up, difference beween Q and A is large but still considered as safe.',
)
parser.add_argument(
    '--motion_stuck_risk_thr', 
    type=float, 
    default=0.8, 
    help='Threshold for determining risk label based on motion stuck.',
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
    default=1e-3,
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
    '--test', 
    type=bool, 
    default=False,
    # default=True,
    help='Whether test a pretrained model.',
)
parser.add_argument(
    '--load_model_pt', 
    type=Path, 
    default=Path("/home/hardli/isaacsim_vla_ws/safety_estimator/safety_estimator/checkpoints/exp-20260111_220710-N10-M10-D16-WL30-MST0.8-E20.pt"),
    help='The path of .pt of a pretrained model for testing.',
)
parser.add_argument(
    '--log_process', 
    type=bool, 
    default=True, 
    help='Whether save the log process of training and validation into .json.',
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
    args.log_id = f"exp-{time}-N{args.history_len}-M{args.future_len}-D{args.encoded_dim}-WL{args.warmup_len}-MST{args.motion_stuck_risk_thr}-E{args.train_epoch}"

    exp = SafetyEsimatiorExp(
        dataset_path=Path("~/isaacsim_vla_ws/record").expanduser(),
        history_len=args.history_len,
        future_len=args.future_len,
        encoded_dim=args.encoded_dim,
        train_set_ratio=args.train_set_ratio,
        valid_set_ratio=args.valid_set_ratio,
        test_set_ratio=args.test_set_ratio,
        batch_size=args.batch_size,
        warmup_len=args.warmup_len,
        motion_stuck_risk_thr=args.motion_stuck_risk_thr,
        train_epoch=args.train_epoch,
        device_choice=args.device_choice,
        learning_rate=args.learning_rate,
        output_path=args.output_path,
        log_id=args.log_id,
        load_model_pt=args.load_model_pt,
        log_process=args.log_process,
        examine_mode=args.examine_mode,
        verbose=args.verbose,
    )

    if args.test:

        print("Test the pretrained model: ")
        print(f"{args.load_model_pt}")
        if os.path.exists(args.load_model_pt):
            exp.test()
        else:
            print(f"The loaded .pt file doesn't exist.")

    else:

        print("Train a new model: ")
        before_train = datetime.now().timestamp()
        print("===================Start=========================")
        exp.train()
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("====================End==========================")
