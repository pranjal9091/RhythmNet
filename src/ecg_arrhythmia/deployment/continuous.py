"""
Deployment-Only Continuous ECG Signal Processing Pipeline.

Provides deterministic Pan-Tompkins QRS / R-peak detection, beat segmentation (216-sample window:
72 pre-R-peak, 144 post-R-peak), per-beat z-score normalization, and continuous batch inference
using the frozen ONNX INT8 model.

Disclaimer:
This continuous pipeline is a deployment/demonstration utility for unsegmented ECG signals
and is NOT certified for primary clinical diagnosis or standalone patient monitoring.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import math
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, convolve

from ecg_arrhythmia.preprocessing.filtering import filter_ecg
from ecg_arrhythmia.preprocessing.normalization import normalize_beats_zscore
from ecg_arrhythmia.deployment.service import ONNXInferenceService
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger("continuous_pipeline")

CLINICAL_DISCLAIMER = (
    "Research prototype for ECG arrhythmia classification according to AAMI EC57 standards. "
    "Not certified for clinical diagnostic use or primary patient monitoring."
)


def detect_r_peaks(
    signal: np.ndarray,
    fs: float = 360.0,
    refractory_sec: float = 0.200,
    search_window_sec: float = 0.075,
) -> np.ndarray:
    """
    Deterministic Pan-Tompkins-style QRS / R-peak detector for 1D ECG signals sampled at 360 Hz.

    Algorithm steps:
    1. 5–15 Hz bandpass filtering to enhance QRS energy and suppress baseline wander/T-waves.
    2. 5-point 1st derivative operator.
    3. Pointwise squaring of derivative signal.
    4. Moving window integration (~150 ms window).
    5. Adaptive thresholding and peak detection with refractory period (~200 ms).
    6. Local R-peak refinement: finds exact amplitude maximum in raw signal within ±75 ms of energy peak.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG signal array.
    fs : float
        Sampling frequency in Hz (default 360 Hz).
    refractory_sec : float
        Minimum time separation between successive R-peaks (default 0.200 s / 200 ms).
    search_window_sec : float
        Half-width of local search window for R-peak refinement around energy peak (default 0.075 s / 75 ms).

    Returns
    -------
    np.ndarray
        1D array of integer R-peak sample indices, sorted chronologically.
    """
    if signal is None or len(signal) == 0:
        return np.array([], dtype=int)

    arr = np.asarray(signal, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(f"Signal must be 1-dimensional, got shape {arr.shape}.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Signal contains non-finite values (NaN or Inf).")

    if len(arr) < 216:
        return np.array([], dtype=int)

    if np.std(arr) < 1e-6:
        return np.array([], dtype=int)

    fs_int = int(round(fs))
    nyquist = 0.5 * fs_int

    # Step 1: Bandpass filter 5–15 Hz for QRS enhancement
    low = 5.0 / nyquist
    high = 15.0 / nyquist
    b, a = butter(2, [low, high], btype="bandpass")
    filtered_qrs = filtfilt(b, a, arr)

    # Step 2: 5-point derivative operator
    kernel = np.array([-1, -2, 0, 2, 1], dtype=np.float64) / 8.0
    derivative = convolve(filtered_qrs, kernel, mode="same")

    # Step 3: Squaring
    squared = derivative ** 2

    # Step 4: Moving window integration (window ~150 ms = 54 samples)
    integration_window = max(1, int(round(0.150 * fs_int)))
    integrated = convolve(squared, np.ones(integration_window) / integration_window, mode="same")

    # Step 5: Adaptive Peak Detection
    min_dist_samples = max(1, int(round(refractory_sec * fs_int)))
    p95_val = np.percentile(integrated, 95)
    p50_val = np.percentile(integrated, 50)
    
    threshold = max(0.20 * p95_val, 1.8 * p50_val + 1e-8)

    peaks, _ = find_peaks(integrated, height=threshold, distance=min_dist_samples)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Step 6: R-peak Local Refinement
    half_win = max(1, int(round(search_window_sec * fs_int)))
    refined_peaks = []
    
    filtered_ecg = filter_ecg(arr, fs=fs_int, low_hz=0.5, high_hz=40.0)

    for p in peaks:
        start_idx = max(0, p - half_win)
        end_idx = min(len(arr), p + half_win + 1)
        sub_window = filtered_ecg[start_idx:end_idx]
        if len(sub_window) > 0:
            local_max_idx = start_idx + np.argmax(sub_window)
            refined_peaks.append(int(local_max_idx))

    if not refined_peaks:
        return np.array([], dtype=int)

    refined_peaks = sorted(list(set(refined_peaks)))
    
    final_peaks = []
    for r in refined_peaks:
        if not final_peaks or (r - final_peaks[-1]) >= min_dist_samples:
            final_peaks.append(r)

    return np.array(final_peaks, dtype=int)


def segment_continuous_ecg(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    pre_samples: int = 72,
    post_samples: int = 144,
) -> Tuple[np.ndarray, List[int], List[Dict[str, Any]]]:
    """
    Extracts 216-sample beat windows around detected R-peaks and applies per-beat z-score normalization.

    Parameters
    ----------
    signal : np.ndarray
        Filtered 1D ECG signal array.
    r_peaks : np.ndarray
        Array of integer R-peak sample indices.
    pre_samples : int
        Number of samples before R-peak (default 72 samples / 200 ms at 360 Hz).
    post_samples : int
        Number of samples after R-peak (default 144 samples / 400 ms at 360 Hz).

    Returns
    -------
    Tuple[np.ndarray, List[int], List[Dict[str, Any]]]
        - valid_beats_norm: 2D array of shape (N_valid, 216) with float32 per-beat z-score normalized samples.
        - valid_r_peaks: List of R-peak sample indices corresponding to valid_beats_norm rows.
        - unclassified_peaks: List of dicts for rejected peaks with boundary clipping reasons.
    """
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1D, got shape {signal.shape}")

    total_len = len(signal)
    total_window_len = pre_samples + post_samples  # 216

    valid_beats = []
    valid_r_peaks = []
    unclassified_peaks = []

    for idx, r in enumerate(r_peaks):
        r_int = int(r)
        start_idx = r_int - pre_samples
        end_idx = r_int + post_samples

        if start_idx < 0 or end_idx > total_len:
            unclassified_peaks.append({
                "beat_index": idx,
                "r_peak_sample": r_int,
                "reason": "boundary_clipping",
                "details": f"Beat window [{start_idx}:{end_idx}] exceeds signal boundary [0:{total_len}]."
            })
        else:
            beat_raw = signal[start_idx:end_idx]
            valid_beats.append(beat_raw)
            valid_r_peaks.append(r_int)

    if not valid_beats:
        valid_beats_norm = np.empty((0, total_window_len), dtype=np.float32)
    else:
        beats_arr = np.array(valid_beats, dtype=np.float64)
        valid_beats_norm = normalize_beats_zscore(beats_arr)

    return valid_beats_norm, valid_r_peaks, unclassified_peaks


class ContinuousInferencePipeline:
    """
    Continuous ECG Signal Inference Pipeline.
    
    Transforms unsegmented raw ECG signal at 360 Hz -> 0.5-40 Hz bandpass filtering ->
    Pan-Tompkins R-peak detection -> 216-sample beat segmentation & per-beat z-score normalization ->
    ONNX INT8 batch inference -> calibrated probabilities and class predictions.
    """

    def __init__(self, service: Optional[ONNXInferenceService] = None):
        """Initializes pipeline with ONNXInferenceService instance."""
        self.service = service if service is not None else ONNXInferenceService()

    def process_signal(
        self,
        signal: Union[List[Union[float, int, str]], np.ndarray],
        sampling_rate: float = 360.0,
    ) -> Dict[str, Any]:
        """
        Processes unsegmented continuous ECG signal and returns predictions for all valid detected beats.

        Parameters
        ----------
        signal : Union[List[Union[float, int, str]], np.ndarray]
            Continuous 1D ECG signal values.
        sampling_rate : float
            Sampling frequency in Hz (must be 360.0 Hz).

        Returns
        -------
        Dict[str, Any]
            Pipeline result dictionary containing detected peaks, processed beats count,
            per-beat predictions, unclassified peaks, and metadata.
        """
        if abs(sampling_rate - 360.0) > 1e-3:
            raise ValueError(
                f"Sampling rate {sampling_rate} Hz is not supported. "
                "The pipeline strictly requires 360 Hz sampling frequency."
            )

        if signal is None:
            raise ValueError("Input signal cannot be None.")

        try:
            arr = np.asarray(signal, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Signal contains elements that cannot be converted to float: {e}")

        if arr.ndim != 1:
            raise ValueError(f"Signal must be 1-dimensional, got shape {arr.shape}.")

        if len(arr) < 216:
            raise ValueError(
                f"Signal length ({len(arr)} samples) is too short. "
                "Minimum required length is 216 samples (600 ms at 360 Hz)."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError("Signal contains non-finite values (NaN or Inf).")

        # Step 1: Bandpass filter (0.5 - 40 Hz)
        filtered_ecg = filter_ecg(arr, fs=360, low_hz=0.5, high_hz=40.0)

        # Step 2: R-peak detection
        r_peaks = detect_r_peaks(filtered_ecg, fs=360.0)

        # Step 3: Beat window segmentation & per-beat z-score normalization
        valid_beats_norm, valid_r_peaks, unclassified_peaks = segment_continuous_ecg(
            filtered_ecg, r_peaks, pre_samples=72, post_samples=144
        )

        num_detected = len(r_peaks)
        num_processed = len(valid_r_peaks)

        # Step 4: ONNX INT8 Batch Inference (chunked in sub-batches of max 500 beats)
        predictions = []
        if num_processed > 0:
            chunk_size = 500
            for start_idx in range(0, num_processed, chunk_size):
                sub_beats = valid_beats_norm[start_idx : start_idx + chunk_size]
                sub_r_peaks = valid_r_peaks[start_idx : start_idx + chunk_size]
                batch_result = self.service.predict_batch(sub_beats)
                for idx_offset, (r_peak, pred_dict) in enumerate(zip(sub_r_peaks, batch_result["predictions"])):
                    pred_dict["beat_index"] = start_idx + idx_offset
                    pred_dict["r_peak_sample"] = int(r_peak)
                    predictions.append(pred_dict)

        return {
            "status": "success",
            "model": self.service.model_name,
            "runtime": "onnx_int8",
            "signal_length": len(arr),
            "sampling_rate": float(sampling_rate),
            "num_detected_peaks": num_detected,
            "num_processed_beats": num_processed,
            "num_boundary_rejected": len(unclassified_peaks),
            "predictions": predictions,
            "unclassified_peaks": unclassified_peaks,
            "disclaimer": CLINICAL_DISCLAIMER,
        }
