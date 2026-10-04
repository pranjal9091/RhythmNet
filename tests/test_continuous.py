"""
Unit and integration tests for Milestone 11 Continuous ECG Processing Pipeline.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ecg_arrhythmia.deployment.continuous import (
    detect_r_peaks,
    segment_continuous_ecg,
    ContinuousInferencePipeline,
)
from ecg_arrhythmia.deployment.api import app
from ecg_arrhythmia.deployment.service import ONNXInferenceService


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with lifespan context initializing ONNX service."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def synthetic_continuous_ecg():
    """Generates a 360 Hz synthetic 5-second ECG signal with 5 distinct QRS peaks."""
    fs = 360
    duration_sec = 5.0
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    
    # Baseline sinusoid + P/T waves
    ecg = 0.1 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.cos(2 * np.pi * 3.0 * t)
    
    # Add QRS peaks every 1.0 second (sample indices: 360, 720, 1080, 1440)
    for peak_t in [1.0, 2.0, 3.0, 4.0]:
        peak_idx = int(peak_t * fs)
        # Sharp QRS pulse
        width = 15
        qrs = 1.5 * np.exp(-0.5 * ((np.arange(len(t)) - peak_idx) / width) ** 2)
        ecg += qrs

    return ecg.astype(np.float32)


# ==============================================================================
# 1. R-PEAK DETECTION TESTS
# ==============================================================================

def test_detect_r_peaks_synthetic(synthetic_continuous_ecg):
    """Test Pan-Tompkins detector on synthetic 360 Hz ECG with known peak locations."""
    peaks = detect_r_peaks(synthetic_continuous_ecg, fs=360.0)
    assert isinstance(peaks, np.ndarray)
    assert len(peaks) >= 4

    # Expected peak indices near 360, 720, 1080, 1440
    expected_peaks = [360, 720, 1080, 1440]
    for exp in expected_peaks:
        close_matches = [p for p in peaks if abs(p - exp) <= 15]
        assert len(close_matches) == 1, f"Expected R-peak near sample {exp}, got {peaks}"


def test_detect_r_peaks_flatline():
    """Test R-peak detector on constant flatline signal."""
    flatline = np.ones(1000, dtype=np.float32) * 0.5
    peaks = detect_r_peaks(flatline, fs=360.0)
    assert len(peaks) == 0


def test_detect_r_peaks_too_short():
    """Test R-peak detector on signal shorter than 216 samples."""
    short_sig = np.random.randn(150).astype(np.float32)
    peaks = detect_r_peaks(short_sig, fs=360.0)
    assert len(peaks) == 0


def test_detect_r_peaks_invalid_inputs():
    """Test R-peak detector exception handling for NaN/Inf and invalid dimensions."""
    # 2D array
    with pytest.raises(ValueError, match="1-dimensional"):
        detect_r_peaks(np.zeros((10, 10)))

    # NaN in signal
    nan_sig = np.zeros(500)
    nan_sig[250] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        detect_r_peaks(nan_sig)

    # Inf in signal
    inf_sig = np.zeros(500)
    inf_sig[100] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        detect_r_peaks(inf_sig)


# ==============================================================================
# 2. BEAT SEGMENTATION & NORMALIZATION TESTS
# ==============================================================================

def test_segment_continuous_ecg_boundary_handling(synthetic_continuous_ecg):
    """Test 216-sample beat segmentation and boundary peak rejection."""
    # Include an intentional boundary peak at index 30 (< 72 samples from start)
    # and at index len-20 (> len-144 from end)
    r_peaks = np.array([30, 360, 720, 1080, len(synthetic_continuous_ecg) - 20])
    
    beats_norm, valid_r_peaks, unclassified = segment_continuous_ecg(
        synthetic_continuous_ecg, r_peaks, pre_samples=72, post_samples=144
    )

    # Rejection checks
    assert len(unclassified) == 2
    reasons = [u["reason"] for u in unclassified]
    assert all(r == "boundary_clipping" for r in reasons)
    rejected_peaks = [u["r_peak_sample"] for u in unclassified]
    assert 30 in rejected_peaks
    assert (len(synthetic_continuous_ecg) - 20) in rejected_peaks

    # Valid beat checks
    assert len(valid_r_peaks) == 3
    assert beats_norm.shape == (3, 216)
    
    # Check per-beat z-score normalization
    for b in beats_norm:
        assert pytest.approx(np.mean(b), abs=1e-4) == 0.0
        assert pytest.approx(np.std(b), abs=1e-4) == 1.0


# ==============================================================================
# 3. CONTINUOUS INFERENCE PIPELINE TESTS
# ==============================================================================

def test_continuous_inference_pipeline_end_to_end(synthetic_continuous_ecg):
    """Test end-to-end ContinuousInferencePipeline on synthetic ECG signal."""
    service = ONNXInferenceService()
    pipeline = ContinuousInferencePipeline(service=service)

    res = pipeline.process_signal(synthetic_continuous_ecg, sampling_rate=360.0)

    assert res["status"] == "success"
    assert res["signal_length"] == len(synthetic_continuous_ecg)
    assert res["sampling_rate"] == 360.0
    assert res["num_detected_peaks"] >= 4
    assert res["num_processed_beats"] >= 4

    assert len(res["predictions"]) == res["num_processed_beats"]
    for pred in res["predictions"]:
        assert "r_peak_sample" in pred
        assert "class_label" in pred
        assert pred["class_label"] in ["N", "S", "V", "F", "Q"]
        assert pytest.approx(sum(pred["probabilities"].values()), abs=1e-4) == 1.0


def test_continuous_inference_pipeline_validation():
    """Test ContinuousInferencePipeline input validation errors."""
    pipeline = ContinuousInferencePipeline()

    # Unsupported sampling rate
    with pytest.raises(ValueError, match="Sampling rate 250.0 Hz is not supported"):
        pipeline.process_signal([0.0] * 500, sampling_rate=250.0)

    # Signal too short
    with pytest.raises(ValueError, match="too short"):
        pipeline.process_signal([0.0] * 200, sampling_rate=360.0)

    # NaN input
    nan_signal = [0.0] * 300
    nan_signal[150] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.process_signal(nan_signal, sampling_rate=360.0)


# ==============================================================================
# 4. REST API POST /predict/signal TESTS
# ==============================================================================

def test_api_predict_signal_valid(client, synthetic_continuous_ecg):
    """Test REST API POST /predict/signal with valid 360 Hz continuous signal."""
    payload = {
        "signal": synthetic_continuous_ecg.tolist(),
        "sampling_rate": 360.0,
    }
    response = client.post("/predict/signal", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["model"] == "cnn_focal"
    assert data["runtime"] == "onnx_int8"
    assert data["signal_length"] == len(synthetic_continuous_ecg)
    assert data["sampling_rate"] == 360.0
    assert data["num_detected_peaks"] >= 4
    assert data["num_processed_beats"] >= 4
    assert len(data["predictions"]) == data["num_processed_beats"]

    for pred in data["predictions"]:
        assert "r_peak_sample" in pred
        assert pred["class_label"] in ["N", "S", "V", "F", "Q"]


def test_api_predict_signal_invalid_sampling_rate(client, synthetic_continuous_ecg):
    """Test REST API POST /predict/signal rejection for invalid sampling rate."""
    payload = {
        "signal": synthetic_continuous_ecg.tolist(),
        "sampling_rate": 500.0,
    }
    response = client.post("/predict/signal", json=payload)
    assert response.status_code == 422
    assert "360 Hz" in str(response.json())


def test_api_predict_signal_too_short(client):
    """Test REST API POST /predict/signal rejection for signal shorter than 216 samples."""
    payload = {
        "signal": [0.0] * 200,
        "sampling_rate": 360.0,
    }
    response = client.post("/predict/signal", json=payload)
    assert response.status_code == 422
    assert "short" in str(response.json()) or "216" in str(response.json())


def test_api_predict_signal_nan_rejected(client):
    """Test REST API POST /predict/signal rejection for NaN values."""
    payload = {
        "signal": [0.0] * 100 + ["NaN"] + [0.0] * 200,
        "sampling_rate": 360.0,
    }
    response = client.post("/predict/signal", json=payload)
    assert response.status_code == 422
    assert "NaN" in str(response.json())
