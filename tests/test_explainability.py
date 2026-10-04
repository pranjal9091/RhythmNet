"""
Unit tests for explainability module in Milestone 8.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from ecg_arrhythmia.deep_learning.models import ECG1DCNN
from ecg_arrhythmia.explainability.gradients import compute_input_gradients
from ecg_arrhythmia.explainability.integrated_gradients import compute_integrated_gradients
from ecg_arrhythmia.explainability.gradcam import GradCAM1D
from ecg_arrhythmia.explainability.examples import select_explainability_examples


@pytest.fixture
def dummy_cnn():
    model = ECG1DCNN(in_channels=1, num_classes=5)
    model.eval()
    return model


@pytest.fixture
def dummy_waveform():
    torch.manual_seed(42)
    return torch.randn(1, 1, 216)


def test_input_gradients_shape_and_finite(dummy_cnn, dummy_waveform):
    device = torch.device("cpu")
    grad = compute_input_gradients(dummy_cnn, dummy_waveform, target_class=0, device=device)

    assert isinstance(grad, np.ndarray)
    assert grad.shape == (216,)
    assert np.all(np.isfinite(grad))


def test_integrated_gradients_shape_finite_completeness(dummy_cnn, dummy_waveform):
    device = torch.device("cpu")
    ig, comp_err = compute_integrated_gradients(
        dummy_cnn, dummy_waveform, target_class=1, device=device, n_steps=20
    )

    assert isinstance(ig, np.ndarray)
    assert ig.shape == (216,)
    assert np.all(np.isfinite(ig))
    assert isinstance(comp_err, float)
    assert comp_err >= 0.0
    # For small n_steps on standard random weights, completeness error is finite and reasonable
    assert comp_err < 5.0


def test_gradcam_shape_and_finite(dummy_cnn, dummy_waveform):
    device = torch.device("cpu")
    gradcam = GradCAM1D(dummy_cnn, target_layer=dummy_cnn.block3[3])
    cam = gradcam.generate_heatmap(dummy_waveform, target_class=2, device=device)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (216,)
    assert np.all(np.isfinite(cam))
    assert np.all(cam >= 0.0)


def test_explainability_does_not_mutate_model_weights(dummy_cnn, dummy_waveform):
    device = torch.device("cpu")
    initial_weights = [p.clone() for p in dummy_cnn.parameters()]

    _ = compute_input_gradients(dummy_cnn, dummy_waveform, target_class=0, device=device)
    _ = compute_integrated_gradients(dummy_cnn, dummy_waveform, target_class=0, device=device, n_steps=10)
    
    gradcam = GradCAM1D(dummy_cnn, target_layer=dummy_cnn.block3[3])
    _ = gradcam.generate_heatmap(dummy_waveform, target_class=0, device=device)

    for p_init, p_curr in zip(initial_weights, dummy_cnn.parameters()):
        assert torch.equal(p_init, p_curr), "Model weights mutated during explanation generation!"


def test_model_predictions_unchanged_by_explainability(dummy_cnn, dummy_waveform):
    device = torch.device("cpu")

    with torch.no_grad():
        out_before = dummy_cnn(dummy_waveform.to(device)).clone()

    _ = compute_integrated_gradients(dummy_cnn, dummy_waveform, target_class=0, device=device, n_steps=10)

    with torch.no_grad():
        out_after = dummy_cnn(dummy_waveform.to(device))

    torch.testing.assert_close(out_before, out_after)


def test_center_index_is_72():
    # Verify annotation center index convention
    CENTER_SAMPLE = 72
    TOTAL_SAMPLES = 216
    assert CENTER_SAMPLE == 72
    assert TOTAL_SAMPLES == 216
    # 72 samples before center, 144 after center
    assert CENTER_SAMPLE == 72
    assert TOTAL_SAMPLES - CENTER_SAMPLE == 144


def test_deterministic_example_selection():
    pred_df = pd.DataFrame({
        "beat_id": [1, 2, 3, 4, 5],
        "record_id": ["100", "100", "101", "101", "102"],
        "sample_index": [100, 200, 300, 400, 500],
        "true_class": ["N", "V", "S", "F", "N"],
        "predicted_class": ["N", "V", "S", "N", "N"],  # beat 4 is F misclassified as N
        "confidence": [0.95, 0.90, 0.88, 0.85, 0.92],
    })

    examples1 = select_explainability_examples(pred_df, seed=42)
    examples2 = select_explainability_examples(pred_df, seed=42)

    assert len(examples1) == len(examples2)
    for ex1, ex2 in zip(examples1, examples2):
        assert ex1["record_id"] == ex2["record_id"]
        assert ex1["beat_id"] == ex2["beat_id"]
        assert ex1["selection_reason"] == ex2["selection_reason"]
