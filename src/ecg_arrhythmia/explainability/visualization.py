"""
Attribution visualization module for 1D ECG beat explainability.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_beat_attribution(
    waveform: np.ndarray,
    attributions: Dict[str, np.ndarray],
    example_meta: Dict[str, Any],
    output_path: Path,
    center_sample: int = 72,
    sampling_rate: float = 360.0,
) -> None:
    """
    Plots ECG beat waveform and aligned attributions (Input Gradient, Integrated Gradients, Grad-CAM).
    
    Parameters
    ----------
    waveform : np.ndarray
        ECG beat waveform array of shape (216,).
    attributions : Dict[str, np.ndarray]
        Dictionary mapping method name ('input_gradient', 'integrated_gradients', 'gradcam') to (216,) array.
    example_meta : Dict[str, Any]
        Metadata dictionary for title/caption.
    output_path : Path
        Target PNG filepath.
    center_sample : int, default=72
        Annotated R-peak center sample index.
    sampling_rate : float, default=360.0
        ECG sampling rate in Hz.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = len(waveform)
    time_ms = (np.arange(n_samples) - center_sample) / sampling_rate * 1000.0

    n_methods = len(attributions)
    fig, axes = plt.subplots(1 + n_methods, 1, figsize=(10, 2.5 * (1 + n_methods)), sharex=True)

    # Top Panel: ECG Waveform
    axes[0].plot(time_ms, waveform, color="black", linewidth=1.8, label="ECG Beat Waveform")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Annotated Beat Center (t=0 ms)")
    axes[0].set_ylabel("Normalized Amplitude", fontsize=10)
    title_str = (
        f"ECG Beat Attribution | Record {example_meta.get('record_id', '')} "
        f"Beat {example_meta.get('beat_id', '')} | "
        f"True: {example_meta.get('true_class', '')}, Pred: {example_meta.get('predicted_class', '')} "
        f"({example_meta.get('confidence', 0.0)*100:.1f}%)"
    )
    axes[0].set_title(title_str, fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right")

    # Lower Panels: Method Attributions
    colors = {"input_gradient": "tab:blue", "integrated_gradients": "tab:purple", "gradcam": "tab:orange"}
    labels = {"input_gradient": "Input Gradient", "integrated_gradients": "Integrated Gradients", "gradcam": "1D Grad-CAM"}

    for idx, (method_name, attr_array) in enumerate(attributions.items(), start=1):
        c = colors.get(method_name, "tab:green")
        l = labels.get(method_name, method_name)

        if method_name == "gradcam":
            axes[idx].fill_between(time_ms, 0, attr_array, color=c, alpha=0.5, label=l)
            axes[idx].plot(time_ms, attr_array, color=c, linewidth=1.5)
        else:
            axes[idx].plot(time_ms, attr_array, color=c, linewidth=1.5, label=l)
            axes[idx].axhline(0, color="gray", linestyle=":", alpha=0.6)

        axes[idx].axvline(0, color="red", linestyle="--", linewidth=1.2)
        axes[idx].set_ylabel("Attribution", fontsize=10)
        axes[idx].grid(True, linestyle="--", alpha=0.4)
        axes[idx].legend(loc="upper right")

    axes[-1].set_xlabel("Time Relative to Beat Center (ms)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_aggregate_attribution(
    aggregate_dict: Dict[str, np.ndarray],
    output_path: Path,
    center_sample: int = 72,
    sampling_rate: float = 360.0,
) -> None:
    """
    Plots aggregate mean absolute attributions across AAMI classes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    n_samples = 216
    time_ms = (np.arange(n_samples) - center_sample) / sampling_rate * 1000.0

    colors = {"N": "blue", "S": "orange", "V": "green", "F": "red"}
    for cls, mean_attr in aggregate_dict.items():
        if cls in colors and np.sum(mean_attr) > 0:
            ax.plot(time_ms, mean_attr, label=f"Class {cls}", color=colors[cls], linewidth=2)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Annotated Center (t=0 ms)")
    ax.set_title("Aggregate Mean Absolute Integrated Gradients Attribution by Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Relative to Beat Center (ms)", fontsize=10)
    ax.set_ylabel("Mean Absolute Attribution", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
