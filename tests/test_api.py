"""
Unit and integration tests for FastAPI REST API deployment layer in Milestone 10B.
"""

from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ecg_arrhythmia.deployment.api import app, get_inference_service
from ecg_arrhythmia.deployment.service import ONNXInferenceService
from ecg_arrhythmia.utils import get_project_root


@pytest.fixture(scope="module")
def client():
    """Module-scoped FastAPI TestClient initializing lifespan ONNX model session."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dummy_valid_beat():
    np.random.seed(42)
    return np.random.randn(216).tolist()


def test_health_endpoint_returns_200_and_metadata(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["model"] == "cnn_focal"
    assert data["runtime"] == "onnxruntime"
    assert data["quantization"] == "int8_static_qdq"
    assert data["execution_provider"] == "CPUExecutionProvider"
    assert data["input_shape"] == [1, 1, 216]
    assert data["num_classes"] == 5
    assert pytest.approx(data["temperature"], abs=1e-4) == 0.9854227900505066
    assert data["classes"] == {"0": "N", "1": "S", "2": "V", "3": "F", "4": "Q"}


def test_ready_endpoint_returns_200_when_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["is_ready"] is True
    assert data["model"] == "cnn_focal"


def test_predict_beat_valid_returns_200_and_prediction(client, dummy_valid_beat):
    response = client.post("/predict/beat", json={"signal": dummy_valid_beat})
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    pred = data["prediction"]
    assert "class_index" in pred
    assert pred["class_index"] in [0, 1, 2, 3, 4]
    assert pred["class_label"] in ["N", "S", "V", "F", "Q"]
    assert "confidence" in pred
    assert 0.0 <= pred["confidence"] <= 1.0

    assert "probabilities" in data
    probs = data["probabilities"]
    assert len(probs) == 5
    assert set(probs.keys()) == {"N", "S", "V", "F", "Q"}

    total_prob = sum(probs.values())
    assert pytest.approx(total_prob, abs=1e-4) == 1.0
    assert pytest.approx(pred["confidence"], abs=1e-5) == probs[pred["class_label"]]


def test_predict_beat_wrong_signal_length_rejected(client):
    short_signal = [0.0] * 200
    res_short = client.post("/predict/beat", json={"signal": short_signal})
    assert res_short.status_code == 422
    assert "216" in str(res_short.json())

    long_signal = [0.0] * 220
    res_long = client.post("/predict/beat", json={"signal": long_signal})
    assert res_long.status_code == 422


def test_predict_beat_nan_rejected(client):
    nan_signal = [0.0] * 100 + ["NaN"] + [0.0] * 115
    response = client.post("/predict/beat", json={"signal": nan_signal})
    assert response.status_code == 422
    assert "NaN" in str(response.json())


def test_predict_beat_inf_rejected(client):
    inf_signal = [0.0] * 50 + ["inf"] + [0.0] * 165
    response = client.post("/predict/beat", json={"signal": inf_signal})
    assert response.status_code == 422
    assert "Infinite" in str(response.json()) or "inf" in str(response.json())


def test_predict_batch_valid_returns_200(client, dummy_valid_beat):
    batch_signals = [dummy_valid_beat, dummy_valid_beat, dummy_valid_beat]
    response = client.post("/predict/batch", json={"signals": batch_signals})
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["num_beats"] == 3
    assert len(data["predictions"]) == 3

    for idx, item in enumerate(data["predictions"]):
        assert item["beat_index"] == idx
        assert item["class_index"] in [0, 1, 2, 3, 4]
        assert item["class_label"] in ["N", "S", "V", "F", "Q"]
        assert len(item["probabilities"]) == 5
        assert pytest.approx(sum(item["probabilities"].values()), abs=1e-4) == 1.0


def test_predict_batch_invalid_empty(client):
    response = client.post("/predict/batch", json={"signals": []})
    assert response.status_code == 422
    assert "empty" in str(response.json())


def test_predict_batch_invalid_too_large(client, dummy_valid_beat):
    large_batch = [dummy_valid_beat] * 1001
    response = client.post("/predict/batch", json={"signals": large_batch})
    assert response.status_code == 422
    assert "1000" in str(response.json())


def test_predict_batch_invalid_beat_length(client, dummy_valid_beat):
    invalid_batch = [dummy_valid_beat, [0.0] * 200]
    response = client.post("/predict/batch", json={"signals": invalid_batch})
    assert response.status_code == 422
    assert "216" in str(response.json())


def test_predict_batch_nan_or_inf_rejected(client, dummy_valid_beat):
    nan_batch = [dummy_valid_beat, [0.0] * 50 + ["NaN"] + [0.0] * 165]
    res_nan = client.post("/predict/batch", json={"signals": nan_batch})
    assert res_nan.status_code == 422
    assert "NaN" in str(res_nan.json())

    inf_batch = [[0.0] * 50 + ["inf"] + [0.0] * 165, dummy_valid_beat]
    res_inf = client.post("/predict/batch", json={"signals": inf_batch})
    assert res_inf.status_code == 422
    assert "Infinite" in str(res_inf.json()) or "inf" in str(res_inf.json())


def test_model_session_reused_across_requests(client, dummy_valid_beat):
    service1 = get_inference_service()
    client.post("/predict/beat", json={"signal": dummy_valid_beat})

    service2 = get_inference_service()
    client.post("/predict/beat", json={"signal": dummy_valid_beat})

    assert service1 is service2
    assert service1.session is service2.session


def test_inference_failure_converted_to_clean_api_error(client, dummy_valid_beat):
    service = get_inference_service()

    with patch.object(service.session, "run", side_effect=RuntimeError("Simulated ONNX error")):
        response = client.post("/predict/beat", json={"signal": dummy_valid_beat})

    assert response.status_code == 500
    data = response.json()
    assert data["status"] == "error"
    assert data["error_type"] == "InferenceError"
    assert "Simulated ONNX error" not in data["message"]


def test_model_initialization_failure():
    with pytest.raises(FileNotFoundError):
        ONNXInferenceService(model_path="non_existent_path.onnx")


def test_output_shape_and_finite_validation(dummy_valid_beat):
    service = ONNXInferenceService()
    assert service.is_ready() is True

    # Test output non-finite detection
    with patch.object(service.session, "run", return_value=[np.array([[np.nan, 0.1, 0.2, 0.3, 0.4]])]):
        with pytest.raises(RuntimeError) as exc_info:
            service.predict_beat(dummy_valid_beat)
        assert "non-finite" in str(exc_info.value)

    # Test output wrong shape detection (5 classes expected, got 4)
    with patch.object(service.session, "run", return_value=[np.array([[0.1, 0.2, 0.3, 0.4]])]):
        with pytest.raises(RuntimeError) as exc_info:
            service.predict_beat(dummy_valid_beat)
        assert "shape mismatch" in str(exc_info.value)


def test_real_test_beat_inference():
    project_root = get_project_root()
    test_beats_path = project_root / "data" / "processed" / "test" / "beats.npy"

    if test_beats_path.exists():
        beats = np.load(test_beats_path)
        real_beat = beats[0].tolist()

        with TestClient(app) as c:
            response = c.post("/predict/beat", json={"signal": real_beat})

        assert response.status_code == 200
        data = response.json()
        
        assert data["prediction"]["class_label"] in ["N", "S", "V", "F", "Q"]
        assert len(data["probabilities"]) == 5
        assert pytest.approx(sum(data["probabilities"].values()), abs=1e-4) == 1.0
        assert 0.0 <= data["prediction"]["confidence"] <= 1.0
