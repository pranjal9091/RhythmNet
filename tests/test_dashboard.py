"""
Unit and integration tests for Dashboard helper functions and ONNX service integration in Milestone 10.
"""

from pathlib import Path
import numpy as np
import pytest
import plotly.graph_objects as go

from ecg_arrhythmia.deployment.dashboard import (
    validate_beat_array,
    parse_csv_beat_input,
    load_test_dataset_sample,
    format_ground_truth_comparison,
    create_ecg_plot,
    DISCLAIMER_TEXT,
    EXPLAINABILITY_TEXT,
)
from ecg_arrhythmia.deployment.service import ONNXInferenceService
from ecg_arrhythmia.utils import get_project_root


@pytest.fixture
def sample_216_beat():
    np.random.seed(42)
    return np.random.randn(216)


def test_validate_beat_array_valid(sample_216_beat):
    is_valid, msg = validate_beat_array(sample_216_beat)
    assert is_valid is True
    assert msg == ""


def test_validate_beat_array_invalid_length():
    short_signal = np.zeros(200)
    is_valid, msg = validate_beat_array(short_signal)
    assert is_valid is False
    assert "216" in msg

    long_signal = np.zeros(220)
    is_valid, msg = validate_beat_array(long_signal)
    assert is_valid is False
    assert "216" in msg


def test_validate_beat_array_nan_and_inf():
    nan_sig = np.zeros(216)
    nan_sig[10] = np.nan
    is_valid, msg = validate_beat_array(nan_sig)
    assert is_valid is False
    assert "NaN" in msg

    inf_sig = np.zeros(216)
    inf_sig[50] = np.inf
    is_valid, msg = validate_beat_array(inf_sig)
    assert is_valid is False
    assert "Infinite" in msg


def test_validate_beat_array_none_or_2d():
    is_valid, msg = validate_beat_array(None)
    assert is_valid is False
    assert "No signal data" in msg

    mat = np.zeros((2, 216))
    is_valid, msg = validate_beat_array(mat)
    assert is_valid is False
    assert "1-dimensional" in msg


def test_parse_csv_beat_input_valid_comma(sample_216_beat):
    csv_text = ",".join(map(str, sample_216_beat))
    parsed = parse_csv_beat_input(csv_text)
    assert isinstance(parsed, np.ndarray)
    assert parsed.shape == (216,)
    assert np.allclose(parsed, sample_216_beat)


def test_parse_csv_beat_input_valid_newline(sample_216_beat):
    csv_text = "\n".join(map(str, sample_216_beat))
    parsed = parse_csv_beat_input(csv_text)
    assert isinstance(parsed, np.ndarray)
    assert parsed.shape == (216,)
    assert np.allclose(parsed, sample_216_beat)


def test_parse_csv_beat_input_with_header(sample_216_beat):
    csv_text = "amplitude\n" + "\n".join(map(str, sample_216_beat))
    parsed = parse_csv_beat_input(csv_text)
    assert isinstance(parsed, np.ndarray)
    assert parsed.shape == (216,)
    assert np.allclose(parsed, sample_216_beat)


def test_parse_csv_beat_input_invalid_length():
    csv_text = ",".join(map(str, range(100)))
    with pytest.raises(ValueError) as exc_info:
        parse_csv_beat_input(csv_text)
    assert "100" in str(exc_info.value)
    assert "216" in str(exc_info.value)


def test_load_test_dataset_sample():
    project_root = get_project_root()
    beats_path = project_root / "data" / "processed" / "test" / "beats.npy"
    meta_path = project_root / "data" / "processed" / "test" / "metadata.csv"

    if beats_path.exists() and meta_path.exists():
        sample = load_test_dataset_sample(beats_path, meta_path, index=0)
        assert "beat_signal" in sample
        assert sample["beat_signal"].shape == (216,)
        assert "record_id" in sample
        assert "ground_truth" in sample
        assert sample["ground_truth"] in ["N", "S", "V", "F", "Q"]
        assert sample["num_total_beats"] > 0

        # Out of bounds test
        with pytest.raises(IndexError):
            load_test_dataset_sample(beats_path, meta_path, index=999999)


def test_format_ground_truth_comparison():
    # Match case
    res_match = format_ground_truth_comparison("V", "V")
    assert res_match["is_match"] is True
    assert res_match["result_text"] == "Match"
    assert res_match["ground_truth"] == "V"
    assert res_match["prediction"] == "V"

    # Mismatch case
    res_mismatch = format_ground_truth_comparison("V", "N")
    assert res_mismatch["is_match"] is False
    assert res_mismatch["result_text"] == "Mismatch"

    # Missing case
    res_none = format_ground_truth_comparison(None, "N")
    assert res_none["is_match"] is None
    assert res_none["result_text"] == "Ground truth not available"


def test_create_ecg_plot(sample_216_beat):
    fig = create_ecg_plot(sample_216_beat, title="Test Plot")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].x.shape[0] == 216
    assert fig.data[0].y.shape[0] == 216


def test_end_to_end_real_test_beat_flow():
    """Verify flow: dataset sample -> parse/validate -> ONNXInferenceService -> prediction format."""
    project_root = get_project_root()
    beats_path = project_root / "data" / "processed" / "test" / "beats.npy"
    meta_path = project_root / "data" / "processed" / "test" / "metadata.csv"

    if beats_path.exists() and meta_path.exists():
        sample_info = load_test_dataset_sample(beats_path, meta_path, index=0)
        signal = sample_info["beat_signal"]

        is_valid, err_msg = validate_beat_array(signal)
        assert is_valid is True
        assert err_msg == ""

        service = ONNXInferenceService()
        result = service.predict_beat(signal)

        assert "prediction" in result
        assert result["prediction"]["class_label"] in ["N", "S", "V", "F", "Q"]
        assert len(result["probabilities"]) == 5
        assert pytest.approx(sum(result["probabilities"].values()), abs=1e-4) == 1.0

        comp = format_ground_truth_comparison(sample_info["ground_truth"], result["prediction"]["class_label"])
        assert comp["is_match"] in [True, False]


def test_disclaimer_texts():
    assert "not for clinical diagnosis" in DISCLAIMER_TEXT
    assert "Production inference currently uses the frozen INT8 ONNX model" in EXPLAINABILITY_TEXT
