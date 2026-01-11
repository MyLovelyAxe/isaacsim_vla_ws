import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from safety_estimator.utils.safety_rules import (
    joint_range_limit,
    velocity_limit,
    motion_stuck,
    DT,
)
from safety_estimator.utils.robot_config import (
    SO100_JOINT_LIMIT_RANGE, 
    SO100_JOINTS_NAMES,
)

DEFAULT_DATASET_PATH = Path(__file__).parent.parent.parent.resolve() / "record"



@dataclass
class Trajectory:
    """One single trajectory consists of executed actions and joint states."""

    joint_states: torch.Tensor
    """The resulted joint states of simulated robot arm, dimension (timestamp, joints), shape (T, 6)."""
    executed_actions: torch.Tensor
    """The actions from VLA model and executed for robot arm, dimension (timestamp, joints), shape (T, 6)."""
    sample_num: Optional[int] = None
    """The number of samples this trajectory offers."""
    general_start_idx: Optional[int] = None
    """[Closed side] The starting idx of this trajectory for a sample window in the whole trajectory dataset."""
    general_end_idx: Optional[int] = None
    """[Open side] The ending idx of this trajectory for a sample window in the whole trajectory dataset."""


    def __len__(self):
        """The length of trajectoy, i.e. the number of timestamps."""

        assert len(self.executed_actions) == len(self.joint_states), (
            "The length of executed_actions and joint_states are different."
        )
        return len(self.executed_actions)


@dataclass
class TrajectoryDataSet:
    """Dataset consisting of trajectories with training data statistics."""

    trajectories: Dict[str, Trajectory]
    """All trajectories under this dataset."""
    mean_q: Optional[torch.Tensor] = None
    """Mean of joint states of entire trajectory training set, shape (6,)"""
    std_q: Optional[torch.Tensor] = None
    """Stanard deviation of joint states of entire trajectory training set, shape (6,)"""
    mean_a: Optional[torch.Tensor] = None
    """Mean of executed actions of entire trajectory training set, shape (6,)"""
    std_a: Optional[torch.Tensor] = None
    """Stanard deviation of executed actions of entire trajectory training set, shape (6,)"""
    eps: torch.Tensor = torch.Tensor([1e-6])
    """Epsilon, in case there are 0s in standard deviation when noramlize."""
    history_len: int = 10
    """The window length of history of joint states Q and executed actions A, i.e. t-N, ..., t"""
    future_len: int = 10
    """The horizon length of future of joint states Q and executed actions A, i.e. t, ..., t+M"""
    batch_size: int = 32
    """Number of samples for one single batch."""
    warmup_len: int = 30 # TODO: this is a temporary compromise, remove it later
    """The number of timestamp at beginning of each trajectory to warm-up, difference beween Q and A is large but still considered as safe."""
    motion_stuck_risk_thr: float = 0.8
    """Threshold for determining risk label based on motion stuck."""
    examine_mode: bool = False
    """In examine mode, the returned samples are not yet normalized."""


    @property
    def N(self) -> int:
        """The window length of history."""
        return self.history_len

    @property
    def M(self) -> int:
        """The horizon length of future."""
        return self.future_len
    
    @property
    def window(self) -> int:
        """The entire window length of one sample."""
        return self.N + 1 + self.M

    @property
    def all_trajectory_names(self) -> List[str]:
        """The labels of all trajectory, i.e. the npy filenames"""
        return list(self.trajectories.keys())
    
    @property
    def general_start_end_idx(self) -> List[str]:
        """A look up table of general starting and ending index of all trajectories."""
        lut = dict()
        for name, trajectory in self.trajectories.items():
            lut[name] = {
                "start": trajectory.general_start_idx,
                "end": trajectory.general_end_idx,
            }
        return lut

    @property
    def batch_num(self) -> int:
        """The number of batches based on batch size."""
        return int(len(self) / self.batch_size)
    
    @property
    def joint_states_vel_limit(self) -> int:
        """The 99% percentile of velocities of joint states."""
        all_joint_states_lst = list()
        for _, trajectory in self.trajectories.items():
            all_joint_states_lst.append(trajectory.joint_states) # shape (T, 6)
        all_joint_states = torch.concatenate(all_joint_states_lst, dim=0) 
        velocity = (all_joint_states[1:] - all_joint_states[:-1]) / DT
        return torch.quantile(velocity.abs(), 0.99, dim=0)

    @property
    def all_risk_labels(self) -> torch.Tensor:
        """All the risk labels."""
        all_risk_labels_lst = list()
        for idx in range(len(self)):
            _, _, risk_label = self[idx]
            all_risk_labels_lst.append(risk_label)
        return torch.Tensor(all_risk_labels_lst)

    @property
    def num_safe(self) -> int:
        """Number of samples whose proposed action is labeled as safe, i.e. risk_label=0."""
        return (self.all_risk_labels == 0.0).sum()

    @property
    def num_risk(self) -> int:
        """Number of samples whose proposed action is labeled as risk, i.e. risk_label=1."""
        return (self.all_risk_labels == 1.0).sum()


    def __post_init__(self):
        # specify the general starting and ending index of samples for each trajectory
        indexing_num = 0
        for _, trajectory in self.trajectories.items():
            # how many samples does this trajectory offer
            sample_num = len(trajectory) - self.window + 1
            trajectory.sample_num = sample_num
            trajectory.general_start_idx = indexing_num # closed side
            trajectory.general_end_idx = indexing_num + sample_num # open side
            indexing_num = trajectory.general_end_idx


    def __len__(self):
        """Length of dataset, i.e. number of all samples"""

        num_sample = 0
        for _, trajectory in self.trajectories.items():
            num_sample += trajectory.sample_num
        return num_sample


    def __getitem__(self, idx):
        """Get one windowlized sample from this dataset."""

        for _, trajectory in self.trajectories.items():
            # if a general sample idx falls in the range of a specific trajectory
            if idx >= trajectory.general_start_idx and idx < trajectory.general_end_idx:
                # current timestamp t inside the current trajectory
                curr_time = idx - trajectory.general_start_idx + self.N
                # history
                joint_states_history=trajectory.joint_states[curr_time-self.N: curr_time]
                executed_actions_history=trajectory.executed_actions[curr_time-self.N: curr_time]
                # proposed next action
                prop_act = trajectory.executed_actions[curr_time]
                # future
                joint_states_future=trajectory.joint_states[curr_time+1: curr_time+self.M]
                executed_actions_future=trajectory.executed_actions[curr_time+1: curr_time+self.M]
                if self.examine_mode:
                    return (
                        torch.concat([joint_states_history, executed_actions_history], dim=1), # shape (N, 12)
                        prop_act, # shape (6,)
                        torch.concat([joint_states_future, executed_actions_future], dim=1), # shape (M, 12)
                    )
                else:
                    # normalize the input of network
                    norm_history = self.normalize_history(
                        joint_states_history=joint_states_history,
                        executed_actions_history=executed_actions_history,
                    )
                    norm_prop_act = self.normalize_prop_act(
                        prop_act=prop_act,
                    )
                    # compute risk score based on future
                    # PS: most of trajectories have warm-up stage for around 30 timestamps
                    # PS: where the differnce between Q and A is also large, but safe, 
                    # PS: so skip all sample window covering these areas for checking stuck
                    if curr_time > self.warmup_len + self.N:
                        check_stuck = True
                    else:
                        check_stuck = False
                    risk_label = self.compute_risk_label(
                        joint_states_future=joint_states_future,
                        executed_actions_future=executed_actions_future,
                        check_stuck=check_stuck,
                    )
                    return (
                        norm_history, # shape (N, 12)
                        norm_prop_act, # shape (6,)
                        risk_label, # shape (1,)
                    )


    def normalize_history(
        self, 
        joint_states_history: torch.Tensor, # shape (N, 6)
        executed_actions_history: torch.Tensor,# shape (N, 6)
    ) -> torch.Tensor: # shape (N, 12)
        """Normalize history sample with statistics."""

        norm_joint_states_history = (joint_states_history - self.mean_q) / (self.std_q + self.eps)
        norm_executed_actions_history = (executed_actions_history - self.mean_a) / (self.std_a + self.eps)
        return torch.concat([norm_joint_states_history, norm_executed_actions_history], dim=1)


    def normalize_prop_act(
        self,
        prop_act: torch.Tensor, # shape (6,)
    ):
        """Normalize proposed next action sample with statistics."""

        return (prop_act - self.mean_a) / (self.std_a + self.eps)


    def compute_risk_label(
        self, 
        joint_states_future: torch.Tensor, # shape (M, 6)
        executed_actions_future: torch.Tensor, # shape (M, 6)
        check_stuck: bool = True,
    ) -> torch.Tensor: # shape (1,)
        """Compute risk label based on near future.
        
        :param joint_states_future: joint states in near-future window, shape (M, 6)
        :param vel_thr: the threshold of velocity for each joint, shape (6,)
        :param check_stuck: whether apply risk rule for checking if getting stuck
        """

        risk_label_1 = joint_range_limit(joint_states_future=joint_states_future)
        risk_label_2 = velocity_limit(
            joint_states_future=joint_states_future,
            # 99% percentile of velocity magnitude, remove unusually large velocities
            vel_thr=self.joint_states_vel_limit, 
        )
        if check_stuck:
            risk_label_3 = motion_stuck(
                joint_states_future=joint_states_future,
                executed_actions_future=executed_actions_future,
                risk_thr=self.motion_stuck_risk_thr,
            )
        else:
            risk_label_3 = 0.0 # safe by default

        # any of risk label as 1 makes it unsafe
        if risk_label_1 == 1.0 or risk_label_2 == 1.0 or risk_label_3 == 1.0:
            return torch.tensor([1.0], dtype=joint_states_future.dtype)
        else:
            return torch.tensor([0.0], dtype=joint_states_future.dtype)


    def get_batch(self, batch_idx: int) -> torch.Tensor:
        """Create batch of samples for training."""

        assert batch_idx < self.batch_num, "required batch_idx exceeds number of all batches."
        norm_history_lst = list()
        norm_prop_act_lst = list()
        risk_label_lst = list()
        for offset_sample_idx in range(self.batch_size):
            # norm_history: dim (history_len, feature_len), shape (N, 12)
            # norm_prop_act: dim (joints_len), shape (6,)
            # risk_label: scalar, shape (1,)
            norm_history, norm_prop_act, risk_label = self[batch_idx * self.batch_size + offset_sample_idx]
            norm_history_lst.append(norm_history)
            norm_prop_act_lst.append(norm_prop_act)
            risk_label_lst.append(risk_label)
        norm_history_batch = torch.stack(norm_history_lst, dim=0) 
        norm_prop_act_batch = torch.stack(norm_prop_act_lst, dim=0) 
        risk_label_batch = torch.stack(risk_label_lst, dim=0) 
        return (
            norm_history_batch, # batch shape: (self.batch_size, N, 12)
            norm_prop_act_batch, # batch shape: (self.batch_size, 6)
            risk_label_batch, # batch shape: (self.batch_size, 1)
        )
        

    def load_sample(self, sample_idx: int):
        """Directly load the sample from dataset."""

        norm_history, norm_prop_act, risk_label = self[sample_idx]
        print(f"norm_history.shape: {norm_history.shape}")
        print(f"norm_prop_act.shape: {norm_prop_act.shape}")
        print(f"risk_label.shape: {risk_label.shape}")
        # only check the first 2 timestamps, and first 2 dim of both joint states and executed actions
        print(f"concatenated: part of joint states: {norm_history[0:2,0:2]}")
        print(f"concatenated: part of executed actions: {norm_history[0:2, 6:8]}")
        return norm_history
        

    def get_sub_sample(self, sample_idx: int):
        """Get the sample from its source trajectory."""

        for name, trajectory in self.trajectories.items():
            if sample_idx >= trajectory.general_start_idx and sample_idx < trajectory.general_end_idx:
                print(f"sub sample is from trajectory {name}")
                subsample_idx = sample_idx - trajectory.general_start_idx
                # only check the first 2 timestamps, and first 2 dim of features
                sub_q_part = trajectory.joint_states[subsample_idx:subsample_idx + 2, 0:2] # shape (time, feature) -> (T, 6)
                sub_a_part = trajectory.executed_actions[subsample_idx:subsample_idx + 2, 0:2] # shape (time, feature) -> (T, 6)
                print(f"isolated: part of joint states: {sub_q_part}")
                print(f"isolated: part of executed actions: {sub_a_part}")
                continue
        return sub_q_part, sub_a_part


    def verify_sample(self, sample_idx: int):
        """Verify the sample directly loaded from dataset equals to the sub-sample in the specific trajectory."""
        
        assert sample_idx < len(self), "sample_idx exceeds the length of dataset"
        assert self.examine_mode == True, "the dataset should be in examine mode to verify sample"
        norm_history = self.load_sample(sample_idx=sample_idx)
        sub_q_part, sub_a_part = self.get_sub_sample(sample_idx=sample_idx)
        assert torch.any(norm_history[0:2,0:2] - sub_q_part < 0.001), "joint states are different"
        assert torch.any(norm_history[0:2,6:8] - sub_a_part < 0.001), "executed actions are different"


    def visualize_qa_error(self, trajectory_name: str):
        """Visualize the error between joint states and executed actions."""

        trajectory = self.trajectories[trajectory_name]
        diff = trajectory.joint_states - trajectory.executed_actions

        row_num = (len(SO100_JOINTS_NAMES) + 1)
        colors = ["steelblue", "orange", "green", "dimgrey", "purple", "brown"]
        fig = plt.figure(figsize=(10, 4 * row_num))
        axes = []

        # all joints
        ax_all = fig.add_subplot(row_num, 1, 1)
        for joint_idx in range(diff.shape[1]):
            ax_all.plot(diff[:,joint_idx], c=colors[joint_idx], label=SO100_JOINTS_NAMES[joint_idx])
        plt.legend()
        axes.append(ax_all)

        # individual joint
        for joint_idx in range(len(SO100_JOINTS_NAMES)):
            ax = fig.add_subplot(row_num, 1, joint_idx+2)  # 7 rows, 1 column, position i
            ax.plot(diff[:,joint_idx], c=colors[joint_idx], label=SO100_JOINTS_NAMES[joint_idx])
            ax.hlines(y=SO100_JOINT_LIMIT_RANGE[joint_idx] * 0.1, xmin=0, xmax=len(diff), color="red")
            ax.hlines(y=-SO100_JOINT_LIMIT_RANGE[joint_idx] * 0.1, xmin=0, xmax=len(diff), color="red")
            plt.legend()
            axes.append(ax)

        axes[-1].set_xlabel("Timestamp")
        axes[0].set_title("Error between joint states and executed actions")
        plt.tight_layout()
        plt.show()


@dataclass
class TrajectoryDataLoader:
    """Dataloader of recorded trjectory to train the safety estimator.
    
    The trajectories are recorded based on Isaac Sim and Lerbot SmolVLA, consisting of:

        - executed actions: the actions generated from SmolVLA and executed on simulated robot in Isaac Sim
        - joint states: the real joint states of simulated robot arm

    The trajectories are stored in format of .npy, each of them has such structure:
        - time_idx
        - joint_states_time
        - joint_states
        - executed_actions_time
        - proposed_actions
        - executed_actions
    """

    dataset_path: Path = DEFAULT_DATASET_PATH
    """The folder that contains the recorded trajectories."""
    history_len: int = 10
    """The window length of history of joint states Q and executed actions A, i.e. t-N, ..., t"""
    future_len: int = 10
    """The horizon length of future of joint states Q and executed actions A, i.e. t, ..., t+M"""
    train_set_ratio: float = 0.8
    """Ratio of training set in the whole dataset"""
    valid_set_ratio: float = 0.1
    """Ratio of validation set in the whole dataset"""
    test_set_ratio: float = 0.1
    """Ratio of testing set in the whole dataset"""
    batch_size: int = 32
    """Number of samples for one single batch."""
    warmup_len: int = 30 # TODO: this is a temporary compromise, remove it later
    """The number of timestamp at beginning of each trajectory to warm-up, difference beween Q and A is large but still considered as safe."""
    motion_stuck_risk_thr: float = 0.8
    """Threshold for determining risk label based on motion stuck."""
    examine_mode: bool = False
    """In examine mode, the returned samples are not yet normalized."""
    verbose: bool = False
    """If True: log the process of creating dataloader."""


    @property
    def N(self) -> int:
        """The window length of history."""
        return self.history_len

    @property
    def M(self) -> int:
        """The horizon length of future."""
        return self.future_len
    
    @property
    def all_npy_filenames(self) -> List[str]:
        """All filenames of .npy file without surffix."""
        return list(npy_file.name.split(".")[0] for npy_file in self.all_npy_files)

    @property
    def train_npy_files(self) -> List[Path]:
        """All paths of .npy file of training set."""
        return self.all_npy_files[:int(len(self.all_npy_files) * self.train_set_ratio)]

    @property
    def train_npy_filenames(self) -> List[str]:
        """All filenames of .npy file without surffix of training set."""
        return list(npy_file.name.split(".")[0] for npy_file in self.train_npy_files)
    
    @property
    def valid_npy_files(self) -> List[Path]:
        """All paths of .npy file of validation set."""
        return self.all_npy_files[
            int(len(self.all_npy_files) * self.train_set_ratio): 
            int(len(self.all_npy_files) * (self.train_set_ratio + self.valid_set_ratio))
        ]

    @property
    def valid_npy_filenames(self) -> List[str]:
        """All filenames of .npy file without surffix of validation set."""
        return list(npy_file.name.split(".")[0] for npy_file in self.valid_npy_files)
    
    @property
    def test_npy_files(self) -> List[Path]:
        """All paths of .npy file of testing set."""
        return self.all_npy_files[
            int(len(self.all_npy_files) * (self.train_set_ratio + self.valid_set_ratio)): 
        ]

    @property
    def test_npy_filenames(self) -> List[str]:
        """All filenames of .npy file without surffix of testing set."""
        return list(npy_file.name.split(".")[0] for npy_file in self.test_npy_files)


    def __post_init__(self):

        print("Split trajectoris into train & valid & test set ...... ")

        # Load all trajectories before preprocessing
        self.all_npy_files = sorted(self.dataset_path.glob("*.npy"))

        # allocate the trajectories into different set
        train_trajectories = dict()
        valid_trajectories = dict()
        test_trajectories = dict()
        for npy_file in self.all_npy_files:
            loaded_npy = np.load(npy_file, allow_pickle=True).item()
            npy_filename = npy_file.name.split(".")[0]
            assert len(loaded_npy["joint_states_time"]) == len(loaded_npy["executed_actions"]), (
                f"The joint states and executed actions in {npy_file} have different length",
            )
            trajectory = Trajectory(
                    joint_states=torch.tensor(np.array(loaded_npy["joint_states"])),
                    executed_actions=torch.tensor(np.array(loaded_npy["executed_actions"])),
                )
            if npy_filename in self.train_npy_filenames:
                train_trajectories[npy_filename] = trajectory
            elif npy_filename in self.valid_npy_filenames:
                valid_trajectories[npy_filename] = trajectory
            elif npy_filename in self.test_npy_filenames:
                test_trajectories[npy_filename] = trajectory
                
        if self.verbose:
            print(train_trajectories.keys())
            print(f"Validation trajectories: ")
            print(valid_trajectories.keys())
            print(f"Testing trajectories: ")
            print(test_trajectories.keys())
        
        # compute normalization statistics of trajectoris from training set
        self.mean_q, self.std_q, self.mean_a, self.std_a = self.compute_norm_stats(
            trajectories=train_trajectories,
            verbose=self.verbose,
        )

        # create datasets
        self.train_set = TrajectoryDataSet(
            trajectories=train_trajectories, 
            mean_q=self.mean_q,
            std_q=self.std_q,
            mean_a=self.mean_a,
            std_a=self.std_a,
            history_len=self.N,
            future_len=self.M,
            batch_size=self.batch_size,
            warmup_len=self.warmup_len,
            motion_stuck_risk_thr=self.motion_stuck_risk_thr,
            examine_mode=self.examine_mode,
        )
        self.valid_set = TrajectoryDataSet(
            trajectories=valid_trajectories, 
            mean_q=self.mean_q,
            std_q=self.std_q,
            mean_a=self.mean_a,
            std_a=self.std_a,
            history_len=self.N,
            future_len=self.M,
            batch_size=self.batch_size,
            warmup_len=self.warmup_len,
            motion_stuck_risk_thr=self.motion_stuck_risk_thr,
            examine_mode=self.examine_mode,
        )
        self.test_set = TrajectoryDataSet(
            trajectories=test_trajectories, 
            mean_q=self.mean_q,
            std_q=self.std_q,
            mean_a=self.mean_a,
            std_a=self.std_a,
            history_len=self.N,
            future_len=self.M,
            batch_size=self.batch_size,
            warmup_len=self.warmup_len,
            motion_stuck_risk_thr=self.motion_stuck_risk_thr,
            examine_mode=self.examine_mode,
        )


    def compute_norm_stats(
        self, 
        trajectories: Dict[str, Trajectory],
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute normalization statistics, i.e mean and standard deviation.
        
        :param trajectories: usually should be the trajectories for training set
        """
        print("Compute normalization statistics ......")
        all_joint_states_lst = list()
        all_executed_actions_lst = list()
        for name, trajectory in trajectories.items():
            all_joint_states_lst.append(trajectory.joint_states) # shape (T, 6)
            all_executed_actions_lst.append(trajectory.executed_actions) # shape (T, 6)
            if verbose:
                print(f"trajectory {name} length: {len(trajectory.joint_states)}")
        # concatenate on timestampe dimention
        all_joint_states = torch.concatenate(all_joint_states_lst, dim=0) 
        all_executed_actions = torch.concatenate(all_executed_actions_lst, dim=0)
        if verbose:
            print(f"concatenated all_joint_states shape: {all_joint_states.shape}")
            print(f"concatenated all_executed_actions shape: {all_executed_actions.shape}")
        # compute noramlization statistics, on timestampe dimention
        mean_q = torch.mean(all_joint_states, dim=0)
        std_q = torch.std(all_joint_states, dim=0)
        mean_a = torch.mean(all_executed_actions, dim=0)
        std_a = torch.std(all_executed_actions, dim=0)
        if verbose:
            print(f"mean_q: {mean_q}")
            print(f"std_q: {std_q}")
            print(f"mean_a: {mean_a}")
            print(f"std_a: {std_a}")

        return mean_q, std_q, mean_a, std_a
