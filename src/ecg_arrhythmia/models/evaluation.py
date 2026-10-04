"""
Evaluation and visualization module for Classical ML Baselines.

Handles inference latency measurement, prediction artifact generation,
confusion matrix plotting, and comparison reporting.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ecg_arrhythmia.models.metrics import (
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_confusion_matrices,
    DEFAULT_CLASS_NAMES,
)


def measure_inference_latency(
    model: Any,
    X: np.ndarray,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> Tuple[float, float]:
    """
    Measures total inference time and average milliseconds per beat.
    
    Parameters
    ----------
    model : Any
        Fitted model instance.
    X : np.ndarray
        Feature matrix of shape (N, D).
    n_warmup : int, default=2
        Number of unmeasured warm-up iterations.
    n_runs : int, default=5
        Number of timed evaluation iterations.
        
    Returns
    -------
    Tuple[float, float]
        (total_time_seconds, ms_per_beat)
    """
    # Warm-up runs
    for _ in range(n_warmup):
        _ = model.predict_proba(X)
        
    start_time = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict_proba(X)
    end_time = time.perf_counter()
    
    avg_total_time = (end_time - start_time) / n_runs
    ms_per_beat = (avg_total_time / len(X)) * 1000.0
    
    return float(avg_total_time), float(ms_per_beat)


def generate_predictions_dataframe(
    metadata_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = DEFAULT_CLASS_NAMES,
) -> pd.DataFrame:
    """
    Generates detailed prediction DataFrame containing beat identifiers,
    true/predicted labels, confidence, and per-class probabilities.
    
    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata DataFrame associated with the partition.
    y_true : np.ndarray
        True integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    y_prob : np.ndarray
        Predicted class probabilities of shape (N, 5).
    class_names : List[str]
        List of class names ['N', 'S', 'V', 'F', 'Q'].
        
    Returns
    -------
    pd.DataFrame
        Prediction DataFrame.
    """
    reverse_mapping = {i: c for i, c in enumerate(class_names)}
    
    df = pd.DataFrame()
    
    if "beat_id" in metadata_df.columns:
        df["beat_id"] = metadata_df["beat_id"].values
    else:
        df["beat_id"] = np.arange(len(metadata_df))
        
    if "record_id" in metadata_df.columns:
        df["record_id"] = metadata_df["record_id"].values
    else:
        df["record_id"] = ""
        
    if "sample_index" in metadata_df.columns:
        df["sample_index"] = metadata_df["sample_index"].values
    else:
        df["sample_index"] = 0
        
    df["true_class"] = [reverse_mapping[int(y)] for y in y_true]
    df["predicted_class"] = [reverse_mapping[int(y)] for y in y_pred]
    df["confidence"] = np.max(y_prob, axis=1)
    
    for i, c in enumerate(class_names):
        df[f"probability_{c}"] = y_prob[:, i]
        
    return df


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    output_path: Path,
    fmt: str = "d",
    cmap: str = "Blues",
) -> None:
    """
    Generates and saves a confusion matrix heatmap plot.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix array of shape (5, 5).
    class_names : List[str]
        Ordered list of class names ['N', 'S', 'V', 'F', 'Q'].
    title : str
        Plot title.
    output_path : Path
        Filepath to save the PNG plot.
    fmt : str, default="d"
        String formatting specification ("d" for raw integer counts, ".2f" for normalized).
    cmap : str, default="Blues"
        Color map name.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("True Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_class_f1_comparison(
    per_class_df: pd.DataFrame,
    partition_name: str,
    output_path: Path,
) -> None:
    """
    Generates a grouped bar chart comparing per-class F1-scores across models.
    
    Parameters
    ----------
    per_class_df : pd.DataFrame
        DataFrame containing columns [model, partition, class, f1].
    partition_name : str
        Partition name ('val' or 'test').
    output_path : Path
        Target PNG filepath.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df = per_class_df[per_class_df["partition"] == partition_name].copy()
    
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=sub_df,
        x="class",
        y="f1",
        hue="model",
        palette="viridis",
        order=DEFAULT_CLASS_NAMES,
    )
    
    plt.title(f"Per-Class F1-Score Comparison ({partition_name.upper()} Partition)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("AAMI EC57 Class", fontsize=11, labelpad=8)
    plt.ylabel("F1-Score", fontsize=11, labelpad=8)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Model", frameon=True)
    
    # Add value annotations above bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
