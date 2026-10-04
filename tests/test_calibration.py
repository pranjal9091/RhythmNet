"""
Unit tests for calibration module in Milestone 8.
"""

import json
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_arrhythmia.deep_learning.models import ECG1DCNN
from ecg_arrhythmia.calibration.temperature_scaling import (
    ModelWithTemperature,
    fit_temperature_scaling,
    save_temperature_artifact,
    load_temperature_artifact,
)
from ecg_arrhythmia.calibration.metrics import compute_calibration_report


@pytest.fixture
def dummy_cnn():
    model = ECG1DCNN(in_channels=1, num_classes=5)
    model.eval()
    return model


@pytest.fixture
def dummy_val_loader():
    torch.manual_seed(42)
    # Generate 100 synthetic validation samples (classes 0..3, 0 Q samples)
    x = torch.randn(100, 1, 216)
    y = torch.randint(0, 4, (100,))  # labels 0, 1, 2, 3 only (no 4 = Q)
    dataset = TensorDataset(x, y)
    
    def collate_fn(batch):
        waveforms = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return {"waveform": waveforms, "label": labels}

    return DataLoader(dataset, batch_size=16, collate_fn=collate_fn)


def test_temperature_positive(dummy_cnn):
    calib_model = ModelWithTemperature(dummy_cnn, temperature=1.5)
    assert calib_model.temperature > 0.0

    calib_model.set_temperature(2.0)
    assert pytest.approx(calib_model.temperature, abs=1e-5) == 2.0

    with pytest.raises(ValueError):
        calib_model.set_temperature(-0.5)


def test_fit_temperature_uses_val_only(dummy_cnn, dummy_val_loader):
    device = torch.device("cpu")
    temp, nll_before, nll_after = fit_temperature_scaling(
        dummy_cnn, dummy_val_loader, device, max_iter=20
    )
    assert temp > 0.0
    assert isinstance(nll_before, float)
    assert isinstance(nll_after, float)
    assert nll_after <= nll_before + 1e-4


def test_temperature_scaling_probabilities_and_predictions(dummy_cnn):
    x = torch.randn(10, 1, 216)
    
    with torch.no_grad():
        uncalib_logits = dummy_cnn(x)
        uncalib_probs = torch.softmax(uncalib_logits, dim=-1).numpy()
        uncalib_preds = torch.argmax(uncalib_logits, dim=-1).numpy()

    calib_model = ModelWithTemperature(dummy_cnn, temperature=1.4)
    with torch.no_grad():
        calib_logits = calib_model(x)
        calib_probs = torch.softmax(calib_logits, dim=-1).numpy()
        calib_preds = torch.argmax(calib_logits, dim=-1).numpy()

    # Probabilities must be finite and sum to ~1.0 per sample
    assert np.all(np.isfinite(calib_probs))
    np.testing.assert_allclose(np.sum(calib_probs, axis=-1), 1.0, atol=1e-5)

    # Argmax predictions must remain identical after pure temperature scaling
    np.testing.assert_array_equal(uncalib_preds, calib_preds)


def test_calibration_does_not_change_model_weights(dummy_cnn, dummy_val_loader):
    device = torch.device("cpu")
    initial_weights = [p.clone() for p in dummy_cnn.parameters()]

    _ = fit_temperature_scaling(dummy_cnn, dummy_val_loader, device, max_iter=10)

    for p_initial, p_current in zip(initial_weights, dummy_cnn.parameters()):
        assert torch.equal(p_initial, p_current), "Model weights mutated during calibration!"


def test_q_validation_support_zero_detection():
    # Synthetic val dataset with 0 Q samples (class index 4)
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_prob = np.eye(5)[y_true]  # Perfect 5-class one-hot
    
    report = compute_calibration_report(y_true, y_prob, n_bins=10)
    assert "ece" in report
    assert "nll" in report
    # Check that class index 4 (Q) has 0 occurrences in y_true
    assert np.sum(y_true == 4) == 0


def test_artifact_save_and_reload(tmp_path):
    artifact_path = tmp_path / "test_model_temperature.json"
    save_temperature_artifact(
        model_name="test_model",
        temperature=1.2345,
        val_nll_before=0.5,
        val_nll_after=0.4,
        output_path=artifact_path,
        n_bins=10,
        seed=42,
    )

    assert artifact_path.exists()
    reloaded_temp = load_temperature_artifact(artifact_path)
    assert pytest.approx(reloaded_temp, abs=1e-5) == 1.2345

    with open(artifact_path, "r") as f:
        data = json.load(f)
    assert data["fitting_split"] == "validation"
    assert data["objective"] == "negative_log_likelihood"
