"""
This file provides some tools for all processes.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path


def save_model(
    model: nn.Module, 
    epoch: int,
    best_auc: float,
    output_path: Path, 
    log_id: str,
):
    """Save the trained weights of model with the best AUC."""

    os.makedirs(output_path, exist_ok=True)
    save_path = output_path / f"{log_id}.pt"
    print(f"save model in {save_path}......")
    torch.save({
        "epoch": epoch,
        "auc": best_auc,
        "model_state_dict": model.state_dict(),
    }, save_path)


def load_model(
    model: nn.Module,  
    device: torch.device, 
    load_model_pt: Path,
):
    """Load a pretrained weights for a model.
    
    :param model: a new created model
    :param device: CPU or cuda
    :param load_path: The path of .pt of a pretrained model
    """

    print(f"Load model from {load_model_pt}......")
    checkpoint = torch.load(load_model_pt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def log_epoch(
    epoch_info: dict, 
    log_path: Path,
):
    """Save training and validation process into json."""

    with open(log_path, "a") as f:
        json.dump(epoch_info, f)
        f.write("\n")
