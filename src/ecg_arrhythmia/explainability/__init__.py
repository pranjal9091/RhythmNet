"""
Explainability package for ECG Arrhythmia Classification.
"""

from ecg_arrhythmia.explainability.gradients import compute_input_gradients
from ecg_arrhythmia.explainability.integrated_gradients import compute_integrated_gradients
from ecg_arrhythmia.explainability.gradcam import GradCAM1D
from ecg_arrhythmia.explainability.examples import select_explainability_examples
from ecg_arrhythmia.explainability.aggregation import compute_aggregate_attributions
from ecg_arrhythmia.explainability.visualization import (
    plot_beat_attribution,
    plot_aggregate_attribution,
)

__all__ = [
    "compute_input_gradients",
    "compute_integrated_gradients",
    "GradCAM1D",
    "select_explainability_examples",
    "compute_aggregate_attributions",
    "plot_beat_attribution",
    "plot_aggregate_attribution",
]
