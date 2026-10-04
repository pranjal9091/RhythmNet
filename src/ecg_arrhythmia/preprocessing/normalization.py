"""Per-beat ECG signal normalization."""

import numpy as np

from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_beats_zscore(
    beats: np.ndarray, epsilon: float = 1.0e-8
) -> np.ndarray:
    """Apply per-beat z-score normalization independently to each beat waveform.

    Formula:
        x_norm = (x - mean(x)) / (std(x) + epsilon)

    Leakage Prevention Guarantee:
        Since each beat waveform is normalized strictly using its own amplitude statistics
        (mean and std of the 216 samples in that single beat), no global distribution statistics
        are computed or shared across records or partitions.

    Args:
        beats: 2D numpy array of shape (N_beats, samples) or 1D array (samples,).
        epsilon: Small scalar constant to prevent division by zero for flat lines.

    Returns:
        Normalized numpy array of float32 values of same shape as input.
    """
    if beats.size == 0:
        return beats.astype(np.float32)

    original_shape = beats.shape

    if beats.ndim == 1:
        x = beats.astype(np.float64)
        mean = np.mean(x)
        std = np.std(x)
        norm = (x - mean) / (std + epsilon)
        return norm.astype(np.float32)

    elif beats.ndim == 2:
        x = beats.astype(np.float64)
        means = np.mean(x, axis=1, keepdims=True)
        stds = np.std(x, axis=1, keepdims=True)
        norm = (x - means) / (stds + epsilon)
        return norm.astype(np.float32)

    else:
        raise ValueError(
            f"Beats array must be 1D or 2D, got shape {beats.shape} (ndim={beats.ndim})"
        )
