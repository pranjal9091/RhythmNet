"""
Unit tests for Milestone 12 Continuous R-Peak Detection Evaluation & Matching Protocol.
"""

import numpy as np
import pytest

from ecg_arrhythmia.evaluation.continuous_eval import evaluate_r_peak_detection


def test_evaluate_r_peak_detection_exact_match():
    """Test matching protocol when detected peaks exactly match reference annotations."""
    annotated = np.array([360, 720, 1080, 1440], dtype=np.int64)
    detected = np.array([360, 720, 1080, 1440], dtype=np.int64)

    metrics = evaluate_r_peak_detection(detected, annotated, fs=360.0, tolerance_sec=0.150)

    assert metrics["tp"] == 4
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["sensitivity"] == 1.0
    assert metrics["ppv"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["mean_abs_timing_error_ms"] == 0.0
    assert metrics["median_timing_error_ms"] == 0.0


def test_evaluate_r_peak_detection_with_timing_offset():
    """Test matching protocol with small timing offsets within tolerance."""
    annotated = np.array([360, 720, 1080, 1440], dtype=np.int64)
    # Offsets: +2 samples (+5.56 ms), -3 samples (-8.33 ms), +1 sample (+2.78 ms), 0 samples
    detected = np.array([362, 717, 1081, 1440], dtype=np.int64)

    metrics = evaluate_r_peak_detection(detected, annotated, fs=360.0, tolerance_sec=0.150)

    assert metrics["tp"] == 4
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["sensitivity"] == 1.0
    assert metrics["ppv"] == 1.0
    assert metrics["f1_score"] == 1.0

    # Expected timing errors in ms: [5.5555..., -8.3333..., 2.7777..., 0.0]
    expected_mae_ms = np.mean([2, 3, 1, 0]) / 360.0 * 1000.0
    assert pytest.approx(metrics["mean_abs_timing_error_ms"], abs=1e-3) == expected_mae_ms


def test_evaluate_r_peak_detection_false_positives_and_false_negatives():
    """Test TP, FP, FN accounting when extra detections and missed peaks occur."""
    annotated = np.array([360, 720, 1080, 1440], dtype=np.int64)
    # Detected: match 360, match 720, false positive at 500, missed 1080, match 1440, false positive at 1800
    detected = np.array([360, 500, 720, 1440, 1800], dtype=np.int64)

    metrics = evaluate_r_peak_detection(detected, annotated, fs=360.0, tolerance_sec=0.150)

    assert metrics["tp"] == 3  # (360, 720, 1440)
    assert metrics["fp"] == 2  # (500, 1800)
    assert metrics["fn"] == 1  # (1080 missed)

    assert pytest.approx(metrics["sensitivity"], abs=1e-4) == 3 / 4  # 0.75
    assert pytest.approx(metrics["ppv"], abs=1e-4) == 3 / 5  # 0.60
    assert pytest.approx(metrics["f1_score"], abs=1e-4) == (2 * 0.75 * 0.60) / (0.75 + 0.60)


def test_evaluate_r_peak_detection_tolerance_boundary():
    """Test tolerance threshold boundary (e.g. ±50 ms = ±18 samples at 360 Hz)."""
    annotated = np.array([360], dtype=np.int64)
    
    # 18 samples offset = 50 ms (within 50 ms tolerance)
    metrics_inside = evaluate_r_peak_detection([378], annotated, fs=360.0, tolerance_sec=0.050)
    assert metrics_inside["tp"] == 1
    assert metrics_inside["fp"] == 0

    # 25 samples offset = 69.4 ms (exceeds 50 ms tolerance)
    metrics_outside = evaluate_r_peak_detection([385], annotated, fs=360.0, tolerance_sec=0.050)
    assert metrics_outside["tp"] == 0
    assert metrics_outside["fp"] == 1
    assert metrics_outside["fn"] == 1


def test_evaluate_r_peak_detection_empty_inputs():
    """Test edge cases with empty detection or annotation arrays."""
    # Empty detected peaks
    m1 = evaluate_r_peak_detection([], [360, 720], fs=360.0)
    assert m1["tp"] == 0
    assert m1["fp"] == 0
    assert m1["fn"] == 2
    assert m1["sensitivity"] == 0.0
    assert m1["ppv"] == 0.0

    # Empty annotated peaks
    m2 = evaluate_r_peak_detection([360, 720], [], fs=360.0)
    assert m2["tp"] == 0
    assert m2["fp"] == 2
    assert m2["fn"] == 0
    assert m2["sensitivity"] == 0.0
    assert m2["ppv"] == 0.0


def test_evaluate_r_peak_detection_duplicate_detections():
    """Test handling of duplicate/clustered detections near a single annotated peak."""
    annotated = np.array([360], dtype=np.int64)
    # Two detections near sample 360 (358 and 362)
    detected = np.array([358, 362], dtype=np.int64)

    metrics = evaluate_r_peak_detection(detected, annotated, fs=360.0, tolerance_sec=0.150)

    # Greedily matches nearest (358: diff 2), remaining (362) becomes FP
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
