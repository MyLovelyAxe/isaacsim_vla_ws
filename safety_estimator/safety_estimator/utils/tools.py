"""
This file provides some tools for all processes.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Callable


def save_model(
    model: nn.Module, 
    epoch: int,
    best_auc: float,
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_a: torch.Tensor,
    std_a: torch.Tensor,
    eps: torch.Tensor,
    history_len: int,
    encoded_dim: int,
    output_path: Path, 
    log_id: str,
):
    """Save the trained weights of model with the best AUC.
    
    :param mean_q: mean of joint states of training set, shape (6,)
    :param std_q: standard deviation of joint states of training set, shape (6,)
    :param mean_a: mean of executed actions of training set, shape (6,)
    :param std_a: standard deviation of executed actions of training set, shape (6,)
    :param eps: Epsilon for normalization
    :param history_len: window length of history, used for construct model
    :param encoded_dim: The dimension of encoded input of network, used for construct model
    """

    os.makedirs(output_path, exist_ok=True)
    save_path = output_path / f"{log_id}.pt"
    print(f"save model in {save_path}......")
    torch.save({
        "epoch": epoch,
        "auc": best_auc,
        "mean_q": mean_q,
        "std_q": std_q,
        "mean_a": mean_a,
        "std_a": std_a,
        "eps": eps,
        "history_len": history_len,
        "encoded_dim": encoded_dim,
        "model_state_dict": model.state_dict(),
    }, save_path)


def load_model(
    model: nn.Module,  
    device: torch.device, 
    load_model_pt: Path,
) -> nn.Module:
    """Load a pretrained weights for a model.
    
    :param model: a new created model
    :param device: CPU or cuda
    :param load_path: The path of .pt of a pretrained model

    :output model: the model populated with pretrained weights
    """

    # load checkpoint
    print(f"Load model from {load_model_pt}......")
    checkpoint = torch.load(load_model_pt, map_location=device)

    # load model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def load_normalizer(
    load_model_pt: Path,
    device: torch.device,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Load the normalizers from a checkpoint.
    
    :param load_path: The path of .pt of a pretrained model
    :param device: CPU or cuda

    :output q_normalizer: normalizer for joint states history
    :output a_normalizer: normalizer for executed actions history
    """

    print(f"Load normalizers from {load_model_pt}......")
    checkpoint = torch.load(load_model_pt, map_location=device)

    q_normalizer = lambda x: (x-checkpoint["mean_q"]) / (checkpoint["std_q"] + checkpoint["eps"])
    a_normalizer = lambda x: (x-checkpoint["mean_a"]) / (checkpoint["std_a"] + checkpoint["eps"])
    
    return q_normalizer, a_normalizer


def load_model_hyperparam(
    load_model_pt: Path,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, int]:
    """Load the hyper-parameters needed to construct a model from a checkpoint.
    
    :param load_path: The path of .pt of a pretrained model
    :param device: CPU or cuda

    :output history_len: window length of history
    :output encoded_dim: The dimension of encoded input of network
    """

    print(f"Load normalizers from {load_model_pt}......")
    checkpoint = torch.load(load_model_pt, map_location=device)
    
    return checkpoint["history_len"], checkpoint["encoded_dim"]


def log_epoch(
    epoch_info: dict, 
    log_path: Path,
):
    """Save training and validation process into json."""

    with open(log_path, "a") as f:
        json.dump(epoch_info, f)
        f.write("\n")



if __name__ == "__main__":

    q_normalizer, a_normalizer = load_normalizer(
        load_model_pt=Path("/home/hardli/isaacsim_vla_ws/safety_estimator/safety_estimator/checkpoints/exp-20260111_220710-N10-M10-D16-WL30-MST0.8-E20.pt"),
        device=torch.device("cpu"),
    )

    rand_q = torch.rand((6,), dtype=torch.float32)
    print(f"Raw state joints: {rand_q}")

    norm_q = q_normalizer(rand_q)
    print(f"Normalized state joints: {norm_q}")