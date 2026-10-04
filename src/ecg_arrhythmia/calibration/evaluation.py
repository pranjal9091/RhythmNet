"""
Calibration evaluation and reliability diagram visualization module.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes binned accuracy, confidence, and counts for reliability diagrams.
    
    Parameters
    ----------
    y_true : np.ndarray
        True integer labels.
    y_prob : np.ndarray
        Predicted probabilities of shape (N, K).
    n_bins : int, default=10
        Number of confidence bins.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (bin_accuracies, bin_confidences, bin_counts)
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        count = np.sum(in_bin)
        if count > 0:
            acc = np.mean(accuracies[in_bin])
            conf = np.mean(confidences[in_bin])
        else:
            acc = 0.0
            conf = (bin_lower + bin_upper) / 2.0

        bin_accs.append(acc)
        bin_confs.append(conf)
        bin_counts.append(count)

    return np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_uncal: np.ndarray,
    y_prob_cal: np.ndarray,
    model_name: str,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """
    Generates side-by-side or overlay reliability diagrams showing uncalibrated vs
    temperature-scaled calibration curves.
    
    Parameters
    ----------
    y_true : np.ndarray
        True integer class labels.
    y_prob_uncal : np.ndarray
        Uncalibrated class probabilities.
    y_prob_cal : np.ndarray
        Temperature-scaled class probabilities.
    model_name : str
        Model name string.
    output_path : Path
        Target PNG filepath.
    n_bins : int, default=10
        Number of bins.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    accs_uncal, confs_uncal, counts_uncal = compute_reliability_curve(y_true, y_prob_uncal, n_bins=n_bins)
    accs_cal, confs_cal, counts_cal = compute_reliability_curve(y_true, y_prob_cal, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bin_centers = np.linspace(1.0 / (2 * n_bins), 1.0 - 1.0 / (2 * n_bins), n_bins)
    width = 1.0 / n_bins

    # Uncalibrated Plot
    axes[0].bar(bin_centers, accs_uncal, width=width * 0.8, alpha=0.7, color="skyblue", edgecolor="blue", label="Outputs")
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Calibration")
    axes[0].set_title(f"{model_name.replace('_', ' ').title()}\n(Uncalibrated)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Confidence", fontsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=10)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left")

    # Calibrated Plot
    axes[1].bar(bin_centers, accs_cal, width=width * 0.8, alpha=0.7, color="lightgreen", edgecolor="green", label="Outputs")
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Calibration")
    axes[1].set_title(f"{model_name.replace('_', ' ').title()}\n(Temperature Scaled)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Confidence", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
