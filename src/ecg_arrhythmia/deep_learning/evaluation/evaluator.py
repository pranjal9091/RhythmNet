"""
Deep learning model evaluation, visualization, and latency benchmarking module.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_deep_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates class predictions and softmax class probabilities for a PyTorch model.
    
    Parameters
    ----------
    model : nn.Module
        Evaluated PyTorch model.
    dataloader : DataLoader
        DataLoader for the dataset.
    device : torch.device
        Compute device.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (y_pred, y_prob) where y_pred has shape (N,) and y_prob has shape (N, 5).
    """
    model.eval()
    model.to(device)

    all_probs = []

    for batch in dataloader:
        waveforms = batch["waveform"].to(device)
        logits = model(waveforms)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    return y_pred, y_prob


@torch.no_grad()
def measure_deep_inference_latency(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> Dict[str, float]:
    """
    Measures deep learning model inference latency and throughput.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    dataloader : DataLoader
        DataLoader for testing.
    device : torch.device
        Compute device.
    n_warmup : int, default=2
        Number of unmeasured warm-up runs.
    n_runs : int, default=5
        Number of timed measurement runs.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing total_time_sec, ms_per_beat, single_beat_ms, throughput_beats_per_sec.
    """
    model.eval()
    model.to(device)

    # Extract sample batch for batch latency
    sample_batch = next(iter(dataloader))["waveform"].to(device)
    single_sample = sample_batch[:1].to(device)
    total_beats = len(dataloader.dataset)

    # Warm-up batch inference
    for _ in range(n_warmup):
        _ = model(sample_batch)
        _ = model(single_sample)

    # Measure full dataset throughput
    start_time = time.perf_counter()
    for _ in range(n_runs):
        for batch in dataloader:
            waveforms = batch["waveform"].to(device)
            _ = model(waveforms)
    end_time = time.perf_counter()

    avg_total_time = (end_time - start_time) / n_runs
    ms_per_beat = (avg_total_time / total_beats) * 1000.0
    throughput = total_beats / avg_total_time

    # Measure single-beat latency
    t_single_start = time.perf_counter()
    for _ in range(100):
        _ = model(single_sample)
    t_single_end = time.perf_counter()
    single_beat_ms = ((t_single_end - t_single_start) / 100.0) * 1000.0

    return {
        "total_time_sec": float(avg_total_time),
        "ms_per_beat": float(ms_per_beat),
        "single_beat_ms": float(single_beat_ms),
        "throughput_beats_per_sec": float(throughput),
    }


def get_model_param_counts(model: nn.Module) -> Tuple[int, int]:
    """
    Counts total parameters and trainable parameters.
    
    Returns
    -------
    Tuple[int, int]
        (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plots training and validation loss, accuracy, and macro-F1 curves over epochs.
    
    Parameters
    ----------
    history_df : pd.DataFrame
        DataFrame containing columns [epoch, train_loss, val_loss, train_acc, val_acc, val_macro_f1].
    output_path : Path
        Target PNG filepath.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history_df["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss curve
    axes[0].plot(epochs, history_df["train_loss"], label="Train Loss", color="blue", linewidth=2)
    axes[0].plot(epochs, history_df["val_loss"], label="Val Loss", color="red", linestyle="--", linewidth=2)
    axes[0].set_title("Loss Trajectory", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("CrossEntropy Loss", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Accuracy curve
    axes[1].plot(epochs, history_df["train_acc"], label="Train Acc", color="blue", linewidth=2)
    axes[1].plot(epochs, history_df["val_acc"], label="Val Acc", color="green", linestyle="--", linewidth=2)
    axes[1].set_title("Accuracy Trajectory", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Validation Macro-F1 curve
    axes[2].plot(epochs, history_df["val_macro_f1"], label="Val Macro-F1", color="purple", linewidth=2)
    axes[2].set_title("Validation Macro-F1 (Early Stopping Metric)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].set_ylabel("Macro-F1 Score", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
