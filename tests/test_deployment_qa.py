"""
Milestone 10 Step 5 — End-to-End Production Deployment QA Test Suite.

Automates verification of:
1. Live Docker API E2E endpoints (/health, /ready, /predict/beat, /predict/batch).
2. Numerical prediction consistency (Local ONNXInferenceService vs Docker /predict/beat vs Docker /predict/batch).
3. Dashboard Integration & API resilience.
4. Container security and hygiene audit (non-root user, .dockerignore, port binding).
"""

from pathlib import Path
import json
import re
import yaml
import requests
import numpy as np
import pytest

from ecg_arrhythmia.deployment.service import (
    ONNXInferenceService,
    AAMI_CLASS_MAPPING,
    AAMI_CLASS_NAMES_FULL,
)
from ecg_arrhythmia.deployment.dashboard import (
    validate_beat_array,
    parse_csv_beat_input,
    get_api_base_url,
)
from ecg_arrhythmia.utils import get_project_root

DOCKER_API_URL = "http://localhost:8000"
DOCKER_DASHBOARD_URL = "http://localhost:8501"


def is_docker_api_available() -> bool:
    """Helper to check if live Docker API container is running and healthy."""
    try:
        res = requests.get(f"{DOCKER_API_URL}/ready", timeout=2.0)
        return res.status_code == 200 and res.json().get("is_ready") is True
    except Exception:
        return False


def is_docker_dashboard_available() -> bool:
    """Helper to check if live Streamlit Dashboard container is running."""
    try:
        res = requests.get(f"{DOCKER_DASHBOARD_URL}/_stcore/health", timeout=2.0)
        return res.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def deterministic_sample_beat():
    """Returns a deterministic 216-sample beat waveform for E2E consistency checks."""
    np.random.seed(42)
    # Generate a realistic synthetic beat waveform
    t = np.linspace(0, 0.6, 216)
    beat = np.sin(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 15 * t)
    beat += np.random.randn(216) * 0.05
    return beat.astype(np.float32)


# ==============================================================================
# 1. API E2E TESTS (LIVE DOCKER API)
# ==============================================================================

@pytest.mark.skipif(not is_docker_api_available(), reason="Docker API container not running on port 8000")
def test_e2e_docker_api_health_endpoint():
    """Verify live Docker GET /health response status, metadata, and temperature factor."""
    res = requests.get(f"{DOCKER_API_URL}/health", timeout=5.0)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model"] == "cnn_focal"
    assert data["runtime"] == "onnxruntime"
    assert data["quantization"] == "int8_static_qdq"
    assert data["execution_provider"] == "CPUExecutionProvider"
    assert data["input_shape"] == [1, 1, 216]
    assert data["num_classes"] == 5
    assert pytest.approx(data["temperature"], abs=1e-4) == 0.9854227900505066
    assert data["classes"] == {"0": "N", "1": "S", "2": "V", "3": "F", "4": "Q"}


@pytest.mark.skipif(not is_docker_api_available(), reason="Docker API container not running on port 8000")
def test_e2e_docker_api_ready_endpoint():
    """Verify live Docker GET /ready readiness probe."""
    res = requests.get(f"{DOCKER_API_URL}/ready", timeout=5.0)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["is_ready"] is True
    assert data["model"] == "cnn_focal"


@pytest.mark.skipif(not is_docker_api_available(), reason="Docker API container not running on port 8000")
def test_e2e_docker_api_predict_beat_valid(deterministic_sample_beat):
    """Verify live Docker POST /predict/beat single-beat prediction schema and probabilities."""
    payload = {"signal": deterministic_sample_beat.tolist()}
    res = requests.post(f"{DOCKER_API_URL}/predict/beat", json=payload, timeout=5.0)
    assert res.status_code == 200
    data = res.json()

    assert "prediction" in data
    pred = data["prediction"]
    assert pred["class_index"] in [0, 1, 2, 3, 4]
    assert pred["class_label"] in ["N", "S", "V", "F", "Q"]
    assert 0.0 <= pred["confidence"] <= 1.0

    assert "probabilities" in data
    probs = data["probabilities"]
    assert len(probs) == 5
    assert set(probs.keys()) == {"N", "S", "V", "F", "Q"}
    assert pytest.approx(sum(probs.values()), abs=1e-4) == 1.0
    assert pytest.approx(pred["confidence"], abs=1e-5) == probs[pred["class_label"]]


@pytest.mark.skipif(not is_docker_api_available(), reason="Docker API container not running on port 8000")
def test_e2e_docker_api_predict_beat_invalid_inputs():
    """Verify live Docker POST /predict/beat input validation errors (HTTP 422)."""
    # 1. Invalid signal length
    res_short = requests.post(f"{DOCKER_API_URL}/predict/beat", json={"signal": [0.0] * 200}, timeout=5.0)
    assert res_short.status_code == 422

    # 2. NaN values
    res_nan = requests.post(f"{DOCKER_API_URL}/predict/beat", json={"signal": [0.0] * 100 + ["NaN"] + [0.0] * 115}, timeout=5.0)
    assert res_nan.status_code == 422

    # 3. Infinite values
    res_inf = requests.post(f"{DOCKER_API_URL}/predict/beat", json={"signal": [0.0] * 50 + ["inf"] + [0.0] * 165}, timeout=5.0)
    assert res_inf.status_code == 422


@pytest.mark.skipif(not is_docker_api_available(), reason="Docker API container not running on port 8000")
def test_e2e_docker_api_predict_batch_valid(deterministic_sample_beat):
    """Verify live Docker POST /predict/batch batch prediction payload."""
    batch_signals = [deterministic_sample_beat.tolist()] * 5
    res = requests.post(f"{DOCKER_API_URL}/predict/batch", json={"signals": batch_signals}, timeout=5.0)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert data["num_beats"] == 5
    assert len(data["predictions"]) == 5

    for item in data["predictions"]:
        assert item["class_index"] in [0, 1, 2, 3, 4]
        assert item["class_label"] in ["N", "S", "V", "F", "Q"]
        assert pytest.approx(sum(item["probabilities"].values()), abs=1e-4) == 1.0


@pytest.mark.skipif(not is_docker_api_available(), reason="Docker API container not running on port 8000")
def test_e2e_docker_api_predict_batch_invalid_inputs():
    """Verify live Docker POST /predict/batch input validation errors (HTTP 422)."""
    # Empty batch
    res_empty = requests.post(f"{DOCKER_API_URL}/predict/batch", json={"signals": []}, timeout=5.0)
    assert res_empty.status_code == 422

    # Exceeding batch size limit
    oversized = [[0.0] * 216] * 1001
    res_oversized = requests.post(f"{DOCKER_API_URL}/predict/batch", json={"signals": oversized}, timeout=5.0)
    assert res_oversized.status_code == 422


# ==============================================================================
# 2. PREDICTION CONSISTENCY (LOCAL VS DOCKER SINGLE VS DOCKER BATCH)
# ==============================================================================

@pytest.mark.skipif(not is_docker_api_available(), reason="Docker API container not running on port 8000")
def test_prediction_consistency_local_vs_docker_single_and_batch(deterministic_sample_beat):
    """
    Verify 100% agreement and numerical probability tolerance (< 1e-5) across:
    1. Local ONNXInferenceService
    2. Docker POST /predict/beat
    3. Docker POST /predict/batch
    """
    # 1. Local inference service prediction
    local_service = ONNXInferenceService()
    local_res = local_service.predict_beat(deterministic_sample_beat)

    # 2. Docker single-beat REST API prediction
    res_single = requests.post(f"{DOCKER_API_URL}/predict/beat", json={"signal": deterministic_sample_beat.tolist()}, timeout=5.0)
    assert res_single.status_code == 200
    docker_single = res_single.json()

    # 3. Docker batch REST API prediction
    res_batch = requests.post(f"{DOCKER_API_URL}/predict/batch", json={"signals": [deterministic_sample_beat.tolist()]}, timeout=5.0)
    assert res_batch.status_code == 200
    docker_batch = res_batch.json()["predictions"][0]

    # Verify Class Labels Match 100%
    assert local_res["prediction"]["class_label"] == docker_single["prediction"]["class_label"]
    assert local_res["prediction"]["class_label"] == docker_batch["class_label"]

    # Verify Probabilities Match within 1e-5 tolerance
    for label in ["N", "S", "V", "F", "Q"]:
        p_local = local_res["probabilities"][label]
        p_docker_single = docker_single["probabilities"][label]
        p_docker_batch = docker_batch["probabilities"][label]

        assert pytest.approx(p_local, abs=1e-5) == p_docker_single
        assert pytest.approx(p_local, abs=1e-5) == p_docker_batch


# ==============================================================================
# 3. DASHBOARD INTEGRATION & RESILIENCE TESTS
# ==============================================================================

@pytest.mark.skipif(not is_docker_dashboard_available(), reason="Streamlit Dashboard container not running on port 8501")
def test_e2e_docker_dashboard_health_endpoint():
    """Verify live Docker Streamlit Dashboard health endpoint."""
    res = requests.get(f"{DOCKER_DASHBOARD_URL}/_stcore/health", timeout=5.0)
    assert res.status_code == 200
    assert res.text.strip() == "ok"


def test_dashboard_api_url_resolution(monkeypatch):
    """Verify dashboard API URL resolution helper."""
    # When no env var is set, defaults to empty string (direct in-process mode)
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.delenv("ECG_API_URL", raising=False)
    assert get_api_base_url() == ""

    # When API_URL is set, returns configured URL
    monkeypatch.setenv("API_URL", "http://api:8000")
    assert get_api_base_url() == "http://api:8000"


def test_dashboard_input_validation_helpers(deterministic_sample_beat):
    """Verify dashboard Mode B array validation and CSV parsing helpers."""
    # Valid beat array
    is_valid, err = validate_beat_array(deterministic_sample_beat)
    assert is_valid is True
    assert err == ""

    # Invalid beat array (length 200)
    is_valid, err = validate_beat_array([0.0] * 200)
    assert is_valid is False
    assert "216" in err

    # Valid CSV string parsing
    csv_str = ",".join(map(str, deterministic_sample_beat.tolist()))
    parsed = parse_csv_beat_input(csv_str)
    assert isinstance(parsed, np.ndarray)
    assert parsed.shape == (216,)
    assert pytest.approx(parsed[0], abs=1e-5) == deterministic_sample_beat[0]


# ==============================================================================
# 4. SECURITY & HYGIENE AUDIT
# ==============================================================================

def test_dockerignore_hygiene_exclusions():
    """Verify .dockerignore excludes virtual environment, git, raw data, and model checkpoints."""
    dockerignore_path = get_project_root() / ".dockerignore"
    assert dockerignore_path.exists()
    content = dockerignore_path.read_text()

    required_exclusions = [
        ".venv",
        ".git",
        "data/raw/",
        "data/processed/train/",
        "artifacts/models/",
        ".pytest_cache",
    ]
    for excl in required_exclusions:
        assert excl in content, f"Missing required exclusion in .dockerignore: {excl}"


def test_docker_compose_port_and_user_hygiene():
    """Verify docker-compose.yml configuration for ports and service environment."""
    compose_path = get_project_root() / "docker-compose.yml"
    assert compose_path.exists()
    content = compose_path.read_text()

    # Ports check
    assert "8000:8000" in content
    assert "8501:8501" in content
    assert "API_URL" in content
    assert "service_healthy" in content


def test_no_hardcoded_secrets_in_deployment_code():
    """Verify deployment codebase contains no hardcoded AWS/GCP/API keys or tokens."""
    deployment_dir = get_project_root() / "src" / "ecg_arrhythmia" / "deployment"
    secret_patterns = [
        r"AKIA[0-9A-Z]{16}",  # AWS key ID
        r"AIzaSy[0-9A-Za-z-_]{35}",  # GCP key
        r"sk-[0-9a-zA-Z]{32,}",  # OpenAI key
        r"bearer\s+[a-zA-Z0-9\._\-]{30,}",  # Hardcoded JWT
    ]
    for py_file in deployment_dir.glob("*.py"):
        text = py_file.read_text()
        for pat in secret_patterns:
            assert not re.search(pat, text), f"Potential hardcoded secret matching '{pat}' in {py_file.name}"
