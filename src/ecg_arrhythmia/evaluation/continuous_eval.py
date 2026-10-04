"""
Continuous ECG R-Peak Detection & Pipeline Evaluation Module.

Provides deterministic matching of deployment-time R-peak detections against PhysioNet ground-truth
annotations, computing standard ANSI/AAMI EC57 metrics (Sensitivity/Recall, PPV/Precision, F1,
MAE timing error, Median timing error, TP, FP, FN).

Disclaimer:
Ground-truth annotations are strictly used post-hoc to evaluate R-peak detection accuracy.
Annotations are NEVER fed into the detector or inference pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

from ecg_arrhythmia.preprocessing.filtering import select_ecg_lead, filter_ecg
from ecg_arrhythmia.data.annotations import AAMI_MAPPING, is_heartbeat
from ecg_arrhythmia.deployment.continuous import (
    detect_r_peaks,
    segment_continuous_ecg,
    ContinuousInferencePipeline,
)
from ecg_arrhythmia.deployment.service import ONNXInferenceService
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger("continuous_eval")


def evaluate_r_peak_detection(
    detected_peaks: Union[List[int], np.ndarray],
    annotated_peaks: Union[List[int], np.ndarray],
    fs: float = 360.0,
    tolerance_sec: float = 0.150,
) -> Dict[str, Any]:
    """
    Evaluates R-peak detection accuracy against reference annotations using greedy bipartite matching.

    Parameters
    ----------
    detected_peaks : Union[List[int], np.ndarray]
        Chronologically sorted list or array of detected R-peak sample indices.
    annotated_peaks : Union[List[int], np.ndarray]
        Chronologically sorted list or array of ground-truth beat annotation sample indices.
    fs : float
        ECG sampling frequency in Hz (default 360.0 Hz).
    tolerance_sec : float
        Matching window tolerance in seconds (default 0.150 s / ±150 ms per ANSI/AAMI EC57).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing TP, FP, FN, sensitivity, ppv, f1_score,
        mean_abs_timing_error_ms, median_timing_error_ms, timing_errors_ms, matched_pairs.
    """
    d_peaks = np.asarray(detected_peaks, dtype=np.int64)
    a_peaks = np.asarray(annotated_peaks, dtype=np.int64)

    tolerance_samples = int(round(tolerance_sec * fs))

    if len(d_peaks) == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(a_peaks),
            "sensitivity": 0.0,
            "ppv": 0.0,
            "f1_score": 0.0,
            "mean_abs_timing_error_ms": 0.0,
            "median_timing_error_ms": 0.0,
            "timing_errors_ms": [],
            "matched_pairs": [],
        }

    if len(a_peaks) == 0:
        return {
            "tp": 0,
            "fp": len(d_peaks),
            "fn": 0,
            "sensitivity": 0.0,
            "ppv": 0.0,
            "f1_score": 0.0,
            "mean_abs_timing_error_ms": 0.0,
            "median_timing_error_ms": 0.0,
            "timing_errors_ms": [],
            "matched_pairs": [],
        }

    # Generate all candidate matching pairs within tolerance
    candidate_pairs = []
    for d_idx, d_val in enumerate(d_peaks):
        # Search for annotations within tolerance
        min_search = d_val - tolerance_samples
        max_search = d_val + tolerance_samples

        # Binary search bounds
        left = np.searchsorted(a_peaks, min_search, side="left")
        right = np.searchsorted(a_peaks, max_search, side="right")

        for a_idx in range(left, right):
            diff = abs(d_val - a_peaks[a_idx])
            if diff <= tolerance_samples:
                candidate_pairs.append((diff, d_idx, a_idx))

    # Sort candidate pairs by distance ascending
    candidate_pairs.sort(key=lambda x: x[0])

    matched_d = set()
    matched_a = set()
    matched_pairs = []
    timing_errors_samples = []

    for dist, d_idx, a_idx in candidate_pairs:
        if d_idx not in matched_d and a_idx not in matched_a:
            matched_d.add(d_idx)
            matched_a.add(a_idx)
            err = int(d_peaks[d_idx]) - int(a_peaks[a_idx])
            matched_pairs.append((int(d_peaks[d_idx]), int(a_peaks[a_idx])))
            timing_errors_samples.append(err)

    tp = len(matched_pairs)
    fp = len(d_peaks) - tp
    fn = len(a_peaks) - tp

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * sensitivity * ppv / (sensitivity + ppv)) if (sensitivity + ppv) > 0 else 0.0

    if tp > 0:
        timing_errors_ms = np.array(timing_errors_samples, dtype=np.float64) / fs * 1000.0
        mae_ms = float(np.mean(np.abs(timing_errors_ms)))
        median_err_ms = float(np.median(timing_errors_ms))
    else:
        timing_errors_ms = np.array([], dtype=np.float64)
        mae_ms = 0.0
        median_err_ms = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(sensitivity),
        "ppv": float(ppv),
        "f1_score": float(f1),
        "mean_abs_timing_error_ms": mae_ms,
        "median_timing_error_ms": median_err_ms,
        "timing_errors_ms": timing_errors_ms.tolist(),
        "matched_pairs": matched_pairs,
    }


def evaluate_record_continuous_pipeline(
    record_name: str,
    data_dir: Union[str, Path],
    service: Optional[ONNXInferenceService] = None,
    tolerance_sec: float = 0.150,
) -> Dict[str, Any]:
    """
    Evaluates continuous R-peak detection, beat window segmentation, and frozen ONNX INT8 inference
    on a single MIT-BIH record.

    Parameters
    ----------
    record_name : str
        MIT-BIH record identifier (e.g. '100', '200').
    data_dir : Union[str, Path]
        Path to directory containing raw MIT-BIH record files (.hea, .dat, .atr).
    service : Optional[ONNXInferenceService]
        ONNXInferenceService instance (initialized if None).
    tolerance_sec : float
        R-peak matching window tolerance in seconds (default 0.150 s / ±150 ms).

    Returns
    -------
    Dict[str, Any]
        Record evaluation summary dictionary containing detector metrics, segmentation audit,
        and end-to-end inference prediction distributions.
    """
    data_dir = Path(data_dir)
    rec_path = str(data_dir / str(record_name))

    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")

    # Filter beat annotations (exclude non-beat annotations like rhythm changes)
    valid_ann_indices = []
    valid_ann_symbols = []
    for idx, (samp, sym) in enumerate(zip(ann.sample, ann.symbol)):
        if is_heartbeat(sym):
            valid_ann_indices.append(int(samp))
            valid_ann_symbols.append(sym)

    valid_ann_indices = np.array(valid_ann_indices, dtype=np.int64)

    # Extract primary MLII lead signal
    raw_sig, lead_name, lead_idx = select_ecg_lead(record, primary_lead="MLII")

    # Instantiate continuous pipeline
    pipeline = ContinuousInferencePipeline(service=service)

    # Run continuous pipeline on raw unsegmented signal
    pipeline_res = pipeline.process_signal(raw_sig, sampling_rate=float(record.fs))

    # Evaluate R-peak detection against ground-truth beat annotations
    detector_eval = evaluate_r_peak_detection(
        detected_peaks=np.array([p["r_peak_sample"] for p in pipeline_res["predictions"]] + 
                                [u["r_peak_sample"] for u in pipeline_res["unclassified_peaks"]]),
        annotated_peaks=valid_ann_indices,
        fs=float(record.fs),
        tolerance_sec=tolerance_sec,
    )

    # Compute class distribution of predictions
    preds = pipeline_res["predictions"]
    class_counts = {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}
    for p in preds:
        lbl = p["class_label"]
        if lbl in class_counts:
            class_counts[lbl] += 1

    return {
        "record_name": str(record_name),
        "lead_name": lead_name,
        "signal_length": len(raw_sig),
        "num_annotated_beats": len(valid_ann_indices),
        "num_detected_peaks": pipeline_res["num_detected_peaks"],
        "num_processed_beats": pipeline_res["num_processed_beats"],
        "num_boundary_rejected": pipeline_res["num_boundary_rejected"],
        "segmentation_success_rate_pct": (
            (pipeline_res["num_processed_beats"] / pipeline_res["num_detected_peaks"] * 100.0)
            if pipeline_res["num_detected_peaks"] > 0 else 0.0
        ),
        "detector_tp": detector_eval["tp"],
        "detector_fp": detector_eval["fp"],
        "detector_fn": detector_eval["fn"],
        "detector_sensitivity": detector_eval["sensitivity"],
        "detector_ppv": detector_eval["ppv"],
        "detector_f1": detector_eval["f1_score"],
        "detector_mae_ms": detector_eval["mean_abs_timing_error_ms"],
        "detector_median_err_ms": detector_eval["median_timing_error_ms"],
        "predicted_class_counts": class_counts,
        "timing_errors_ms": detector_eval["timing_errors_ms"],
    }
