"""
Deep learning evaluation subpackage.
"""

from ecg_arrhythmia.deep_learning.evaluation.evaluator import (
    predict_deep_model,
    measure_deep_inference_latency,
    get_model_param_counts,
    plot_training_curves,
)

__all__ = [
    "predict_deep_model",
    "measure_deep_inference_latency",
    "get_model_param_counts",
    "plot_training_curves",
]
