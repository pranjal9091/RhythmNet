"""
Deep Learning package for ECG Arrhythmia Classification.
"""

from ecg_arrhythmia.deep_learning.dataset import ECGBeatDataset
from ecg_arrhythmia.deep_learning.models import ECG1DCNN
from ecg_arrhythmia.deep_learning.losses import FocalLoss
from ecg_arrhythmia.deep_learning.imbalance import (
    compute_inverse_class_weights,
    compute_sqrt_inverse_class_weights,
    create_weighted_sampler,
    get_effective_sampling_stats,
)
from ecg_arrhythmia.deep_learning.training import (
    train_one_epoch,
    evaluate_epoch,
    train_deep_model,
)
from ecg_arrhythmia.deep_learning.evaluation import (
    predict_deep_model,
    measure_deep_inference_latency,
    get_model_param_counts,
    plot_training_curves,
)

__all__ = [
    "ECGBeatDataset",
    "ECG1DCNN",
    "FocalLoss",
    "compute_inverse_class_weights",
    "compute_sqrt_inverse_class_weights",
    "create_weighted_sampler",
    "get_effective_sampling_stats",
    "train_one_epoch",
    "evaluate_epoch",
    "train_deep_model",
    "predict_deep_model",
    "measure_deep_inference_latency",
    "get_model_param_counts",
    "plot_training_curves",
]
