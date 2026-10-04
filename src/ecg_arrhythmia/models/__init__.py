"""
Models package for ECG Arrhythmia Classification.
"""

from ecg_arrhythmia.models.classical import (
    build_classical_model,
    train_model,
    predict_model,
)
from ecg_arrhythmia.models.metrics import (
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_confusion_matrices,
    compute_ece,
    DEFAULT_LABEL_MAPPING,
    DEFAULT_CLASS_NAMES,
)
from ecg_arrhythmia.models.serialization import (
    save_model,
    load_model,
    save_label_mapping,
    load_label_mapping,
    get_model_size_mb,
)
from ecg_arrhythmia.models.evaluation import (
    measure_inference_latency,
    generate_predictions_dataframe,
    plot_confusion_matrix,
    plot_per_class_f1_comparison,
)

__all__ = [
    "build_classical_model",
    "train_model",
    "predict_model",
    "compute_overall_metrics",
    "compute_per_class_metrics",
    "compute_confusion_matrices",
    "compute_ece",
    "save_model",
    "load_model",
    "save_label_mapping",
    "load_label_mapping",
    "get_model_size_mb",
    "measure_inference_latency",
    "generate_predictions_dataframe",
    "plot_confusion_matrix",
    "plot_per_class_f1_comparison",
    "DEFAULT_LABEL_MAPPING",
    "DEFAULT_CLASS_NAMES",
]
