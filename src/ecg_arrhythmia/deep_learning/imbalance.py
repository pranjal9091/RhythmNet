"""
Class imbalance handling module.

Provides train-only class weight computation (inverse frequency, sqrt-inverse frequency),
WeightedRandomSampler creation for oversampling, and auditing reports.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def compute_inverse_class_weights(
    y_train: np.ndarray,
    num_classes: int = 5,
) -> torch.Tensor:
    """
    Computes inverse-frequency class weights strictly from training set labels.
    
    w_c = N_total / (num_classes * N_c)
    
    Parameters
    ----------
    y_train : np.ndarray
        Integer class labels of training set.
    num_classes : int, default=5
        Total number of target classes.
        
    Returns
    -------
    torch.Tensor
        Float tensor of shape (num_classes,) containing class weights.
    """
    N_total = len(y_train)
    weights = np.zeros(num_classes, dtype=np.float32)
    
    for c in range(num_classes):
        N_c = np.sum(y_train == c)
        if N_c > 0:
            weights[c] = N_total / (num_classes * N_c)
        else:
            weights[c] = 1.0
            
    return torch.tensor(weights, dtype=torch.float32)


def compute_sqrt_inverse_class_weights(
    y_train: np.ndarray,
    num_classes: int = 5,
) -> torch.Tensor:
    """
    Computes square-root inverse-frequency class weights strictly from training set labels.
    
    w_c = sqrt(N_total / (num_classes * N_c))
    
    Parameters
    ----------
    y_train : np.ndarray
        Integer class labels of training set.
    num_classes : int, default=5
        Total number of target classes.
        
    Returns
    -------
    torch.Tensor
        Float tensor of shape (num_classes,) containing class weights.
    """
    inv_weights = compute_inverse_class_weights(y_train, num_classes=num_classes).numpy()
    sqrt_weights = np.sqrt(inv_weights)
    return torch.tensor(sqrt_weights, dtype=torch.float32)


def create_weighted_sampler(
    y_train: np.ndarray,
    num_classes: int = 5,
    seed: int = 42,
) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler for training set oversampling based on inverse class frequency.
    
    Parameters
    ----------
    y_train : np.ndarray
        Integer class labels of training set.
    num_classes : int, default=5
        Total number of target classes.
    seed : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    WeightedRandomSampler
        PyTorch WeightedRandomSampler instance.
    """
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    
    sample_weights = class_weights[y_train]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    
    # Set seed generator if supported or use deterministic sampler
    generator = torch.Generator().manual_seed(seed)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(y_train),
        replacement=True,
        generator=generator,
    )
    return sampler


def get_effective_sampling_stats(
    y_train: np.ndarray,
    class_names: Tuple[str, ...] = ("N", "S", "V", "F", "Q"),
) -> pd.DataFrame:
    """
    Generates an audit DataFrame reporting training class distribution, class weights,
    and expected sampling probabilities.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training labels.
    class_names : Tuple[str, ...]
        Class name tuple.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns [class, count, inverse_weight, sqrt_inverse_weight,
        sampling_weight, expected_sampling_prob, effective_samples_per_epoch].
    """
    num_classes = len(class_names)
    inv_w = compute_inverse_class_weights(y_train, num_classes).numpy()
    sqrt_w = compute_sqrt_inverse_class_weights(y_train, num_classes).numpy()
    
    counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    
    # Under uniform class sampling, each class weight = 1 / count
    # Sum of sample weights across all samples of class c = count * (1 / count) = 1.0
    # Therefore, expected sampling probability for each class = 1.0 / num_classes = 0.20
    sampler_class_weight = 1.0 / np.maximum(counts, 1)
    total_weight = np.sum(counts * sampler_class_weight)  # = num_classes (5.0)
    expected_prob = (counts * sampler_class_weight) / total_weight
    effective_samples = expected_prob * total_samples
    
    rows = []
    for c in range(num_classes):
        rows.append({
            "class": class_names[c],
            "count": int(counts[c]),
            "inverse_frequency_weight": float(inv_w[c]),
            "sqrt_inverse_frequency_weight": float(sqrt_w[c]),
            "sampling_weight": float(sampler_class_weight[c]),
            "expected_sampling_probability": float(expected_prob[c]),
            "effective_samples_per_epoch": float(effective_samples[c]),
        })
        
    return pd.DataFrame(rows)
