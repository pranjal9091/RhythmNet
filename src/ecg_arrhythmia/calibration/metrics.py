"""
Calibration metrics module for ECG Arrhythmia Classification.

Provides computation routines for:
- Negative Log-Likelihood (NLL)
- Multiclass Brier Score
- Expected Calibration Error (ECE)
- Full calibration summary report
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from ecg_arrhythmia.models.metrics import compute_ece


def compute_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Computes Negative Log-Likelihood (NLL) / Log Loss for multi-class probabilities.
    
    Parameters
    ----------
    y_true : np.ndarray
        True integer labels of shape (N,).
    y_prob : np.ndarray
        Predicted class probabilities of shape (N, K).
    eps : float, default=1e-15
        Epsilon for numerical stability.
        
    Returns
    -------
    float
        NLL score.
    """
    probs_clipped = np.clip(y_prob, eps, 1.0 - eps)
    N = len(y_true)
    if N == 0:
        return 0.0
    correct_probs = probs_clipped[np.arange(N), y_true]
    nll = -np.mean(np.log(correct_probs))
    return float(nll)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 5) -> float:
    """
    Computes Multiclass Brier Score.
    
    Brier = (1 / N) * sum_i sum_k (p_{ik} - y_{ik})^2
    
    Parameters
    ----------
    y_true : np.ndarray
        True integer labels of shape (N,).
    y_prob : np.ndarray
        Predicted probabilities of shape (N, K).
    num_classes : int, default=5
        Number of target classes.
        
    Returns
    -------
    float
        Brier score in range [0, 2].
    """
    N = len(y_true)
    if N == 0:
        return 0.0
    y_onehot = np.eye(num_classes)[y_true]
    brier = np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1))
    return float(brier)


def compute_calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    labels: list = [0, 1, 2, 3, 4],
) -> Dict[str, float]:
    """
    Computes full evaluation report combining classification and calibration metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True integer labels.
    y_prob : np.ndarray
        Predicted probabilities of shape (N, 5).
    n_bins : int, default=10
        Number of bins for ECE calculation.
    labels : list, default=[0, 1, 2, 3, 4]
        Class integer labels.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of all metrics.
    """
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else acc
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    nll = compute_nll(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob, num_classes=len(labels))
    ece = compute_ece(y_true, y_prob, n_bins=n_bins)

    return {
        "nll": float(nll),
        "ece": float(ece),
        "brier": float(brier),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
