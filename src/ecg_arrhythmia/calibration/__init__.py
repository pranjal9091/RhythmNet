"""
Calibration package for ECG Arrhythmia Classification.
"""

from ecg_arrhythmia.calibration.metrics import (
    compute_nll,
    compute_brier_score,
    compute_calibration_report,
)
from ecg_arrhythmia.calibration.temperature_scaling import (
    ModelWithTemperature,
    fit_temperature_scaling,
    save_temperature_artifact,
    load_temperature_artifact,
)
from ecg_arrhythmia.calibration.evaluation import (
    compute_reliability_curve,
    plot_reliability_diagram,
)

__all__ = [
    "compute_nll",
    "compute_brier_score",
    "compute_calibration_report",
    "ModelWithTemperature",
    "fit_temperature_scaling",
    "save_temperature_artifact",
    "load_temperature_artifact",
    "compute_reliability_curve",
    "plot_reliability_diagram",
]
