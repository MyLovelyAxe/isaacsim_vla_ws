"""
This file defines different rules to determine whether an action is safe to execute or not, 
based on near future window including joint states and executed actions from recorded trajectories.
"""
import torch
from typing import List
from safety_estimator.utils.robot_config import (
    SO100_JOINT_LIMIT_RANGE,
    SO100_JOINT_LIMIT_MIN,
    SO100_JOINT_LIMIT_MAX,
    SO100_JOINTS_NAMES,
)

DT = 0.1  # seconds (10 Hz), the frequency of recorded trajectories


def joint_range_limit(
    joint_states_future: torch.Tensor,
    margin_thr: float = 0.03,
) -> torch.Tensor:
    """Unsafe, if the joint states are too close to limit.

    :param joint_states_future: joint states in near-future window, shape (M, 6)
    :param margin_thr: the threshold of margin ratio to decide if close

    :output risk_label: if 1, unsafe; if 0, safe
    """
    data_type = joint_states_future.dtype
    limit_min = torch.tensor(SO100_JOINT_LIMIT_MIN, dtype=data_type)
    limit_max = torch.tensor(SO100_JOINT_LIMIT_MAX, dtype=data_type)
    joint_range = limit_max - limit_min

    # distance to nearest limit
    margin = torch.minimum(
        limit_max - joint_states_future,
        joint_states_future - limit_min,
    )

    # unsafe if margin too small
    margin_ratio = margin / joint_range
    risk_label = (margin_ratio < margin_thr).any().float()
    return risk_label


def velocity_limit(
    joint_states_future: torch.Tensor,
    vel_thr: torch.Tensor,
    dt: float = DT,
) -> torch.Tensor:
    """Unsafe, if joint velocities spike too high.

    :param joint_states_future: joint states in near-future window, shape (M, 6)
    :param vel_thr: the threshold of velocity for each joint, shape (6,)
    :param dt: time interval for computing velocity

    :output risk_label: scalar, if 1, unsafe; if 0, safe
    """
    # window doesn't contain at least 2 timestamps, can't compute velocity
    if joint_states_future.shape[0] < 2:
        return torch.tensor(0.0, dtype=joint_states_future.dtype)

    velocity = (joint_states_future[1:] - joint_states_future[:-1]) / dt
    risk_label = (velocity.abs() > vel_thr).any().float()

    return risk_label


def motion_stuck(
    joint_states_future: torch.Tensor,
    executed_actions_future: torch.Tensor,
    joint_limits: List[float] = list(j * 0.1 for j in SO100_JOINT_LIMIT_RANGE),
    risk_thr: float = 0.8,
) -> torch.Tensor:
    """Unsafe, if the robot arm gets stuck at somewhere.

    :param joint_states_future: joint states in near-future window, shape (M, 6)
    :param executed_actions_future: executed actions in near-future window, shape (M, 6)
    :param joint_limits: if difference between Q and A exceeds joint limit, stuck
    :param risk_thr: only when at least this many changes execeed the limits, label as unsafe

    :output risk_label: scalar, if 1, unsafe; if 0, safe
    """
    difference = joint_states_future - executed_actions_future

    # how many changes of Q and A exceed the limits, i.e. get stuck
    exceed = torch.abs(difference) >= torch.Tensor([joint_limits]) # shape (M, 6)

    # if any of joints have at least e.g. 50% timestamp which exceed the limits, unsafe
    risk_label = (torch.sum(exceed, dim=0) / exceed.shape[0] >= risk_thr).any().float()

    return risk_label



if __name__ == "__main__":

    sample_q = torch.rand([10,6], dtype=torch.float32)
    sample_a = torch.rand([10,6], dtype=torch.float32)
    sample_vel_thr = torch.rand([6], dtype=torch.float32)

    risk_label_1 = joint_range_limit(
        joint_states_future=sample_q,
    )
    print(f"risk label based on joint_range_limit: {risk_label_1}")

    risk_label_2 = velocity_limit(
        joint_states_future=sample_q,
        vel_thr=sample_vel_thr,
    )
    print(f"risk label based on joint_range_limit: {risk_label_1}")

    risk_label_3 = motion_stuck(
        joint_states_future=sample_q,
        executed_actions_future=sample_a,
    )
    print(f"risk label based on motion_stuck: {risk_label_3}")
