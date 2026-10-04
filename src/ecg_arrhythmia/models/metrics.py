"""
Metrics computation module for ECG Arrhythmia Classification.

Provides overall, per-class, calibration (ECE), and confusion matrix calculations
with explicit support for zero-support minority classes and deterministic 5-class AAMI mapping.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

DEFAULT_LABEL_MAPPING = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
DEFAULT_CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-class predictions.
    
    ECE measures the difference between model confidence and empirical accuracy
    across binned confidence intervals.
    
    Parameters
    ----------
    y_true : np.ndarray
        True integer class labels of shape (N,).
    y_prob : np.ndarray
        Predicted class probability matrix of shape (N, K).
    n_bins : int, default=10
        Number of equal-width confidence bins in [0, 1].
        
    Returns
    -------
    float
        ECE score in range [0, 1].
    """
    if len(y_true) == 0:
        return 0.0
        
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def compute_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    labels: List[int] = [0, 1, 2, 3, 4],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Computes overall summary metrics for multi-class classification.
    
    Parameters
    ----------
    y_true : np.ndarray
        True integer class labels.
    y_pred : np.ndarray
        Predicted integer class labels.
    y_prob : np.ndarray
        Predicted probability matrix of shape (N, K).
    labels : List[int], default=[0, 1, 2, 3, 4]
        List of integer class labels.
    n_bins : int, default=10
        Number of bins for ECE calculation.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of overall metric scores.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    macro_prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    
    # Compute One-vs-Rest PR-AUC and ROC-AUC per class
    pr_aucs = []
    roc_aucs = []
    
    for cls in labels:
        y_binary = (y_true == cls).astype(int)
        pos_count = np.sum(y_binary)
        
        if pos_count > 0 and pos_count < len(y_true):
            pr_auc = average_precision_score(y_binary, y_prob[:, cls])
            roc_auc = roc_auc_score(y_binary, y_prob[:, cls])
            pr_aucs.append(pr_auc)
            roc_aucs.append(roc_auc)
        elif pos_count > 0:
            pr_auc = average_precision_score(y_binary, y_prob[:, cls])
            pr_aucs.append(pr_auc)
            
    macro_pr_auc = float(np.mean(pr_aucs)) if len(pr_aucs) > 0 else np.nan
    macro_roc_auc = float(np.mean(roc_aucs)) if len(roc_aucs) > 0 else np.nan
    
    ece = compute_ece(y_true, y_prob, n_bins=n_bins)
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_pr_auc": macro_pr_auc,
        "macro_roc_auc": macro_roc_auc,
        "ece": float(ece),
    }


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = DEFAULT_CLASS_NAMES,
    label_mapping: Dict[str, int] = DEFAULT_LABEL_MAPPING,
) -> pd.DataFrame:
    """
    Computes per-class metrics including precision, recall, F1, support, and PR-AUC.
    
    Handles zero-support classes safely by setting precision, recall, F1 to 0.0
    and PR-AUC to NaN.
    
    Parameters
    ----------
    y_true : np.ndarray
        True integer class labels.
    y_pred : np.ndarray
        Predicted integer class labels.
    y_prob : np.ndarray
        Predicted probability matrix of shape (N, K).
    class_names : List[str]
        List of class names in order.
    label_mapping : Dict[str, int]
        Mapping from class name to integer label.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns [class, precision, recall, f1, support, pr_auc].
    """
    rows = []
    
    precisions = precision_score(y_true, y_pred, labels=[label_mapping[c] for c in class_names], average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=[label_mapping[c] for c in class_names], average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=[label_mapping[c] for c in class_names], average=None, zero_division=0)
    
    for i, cls in enumerate(class_names):
        cls_idx = label_mapping[cls]
        y_binary = (y_true == cls_idx).astype(int)
        support = int(np.sum(y_binary))
        
        if support > 0:
            pr_auc = float(average_precision_score(y_binary, y_prob[:, cls_idx]))
        else:
            pr_auc = np.nan
            
        rows.append({
            "class": cls,
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": support,
            "pr_auc": pr_auc,
        })
        
    return pd.DataFrame(rows)


def compute_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int] = [0, 1, 2, 3, 4],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes raw and true-class-normalized confusion matrices.
    
    Parameters
    ----------
    y_true : np.ndarray
        True class integer labels.
    y_pred : np.ndarray
        Predicted class integer labels.
    labels : List[int]
        Ordered list of integer labels.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (raw_confusion_matrix, normalized_confusion_matrix)
    """
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm_raw.astype(float) / row_sums, 0.0)
    return cm_raw, cm_norm
