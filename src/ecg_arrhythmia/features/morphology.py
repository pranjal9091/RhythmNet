"""Feature Group A: Basic morphology, statistical amplitude descriptors, and derivative metrics."""

from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import kurtosis, skew

from ecg_arrhythmia.features.schema import FeatureSchema
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def extract_morphology_features(
    beats: np.ndarray,
    schema: FeatureSchema,
    percentiles: List[int] = [5, 10, 25, 50, 75, 90, 95],
) -> np.ndarray:
    """Extract basic statistical and amplitude morphology features for each beat waveform.

    Args:
        beats: 2D array of shape (N, 216).
        schema: FeatureSchema object to register feature definitions.
        percentiles: List of percentile thresholds (5..95).

    Returns:
        2D array of shape (N, D_morph) containing float32 feature values.
    """
    if beats.ndim != 2:
        raise ValueError(f"Beats array must be 2D, got shape {beats.shape}")

    n_beats, n_samples = beats.shape
    features_list: List[np.ndarray] = []

    # 1. Basic amplitude & statistical moments
    f_mean = np.mean(beats, axis=1)
    f_std = np.std(beats, axis=1)
    f_median = np.median(beats, axis=1)
    f_min = np.min(beats, axis=1)
    f_max = np.max(beats, axis=1)
    f_ptp = f_max - f_min
    f_rms = np.sqrt(np.mean(beats**2, axis=1))
    f_energy = np.sum(beats**2, axis=1)
    f_abs_mean = np.mean(np.abs(beats), axis=1)

    # Higher order moments: skewness & kurtosis (vectorized along axis=1)
    # Using scipy skew and kurtosis with nan_policy='omit'
    f_skew = skew(beats, axis=1, bias=False)
    f_kurt = kurtosis(beats, axis=1, bias=False)

    # Replace potential NaN/Inf from flat arrays with 0
    f_skew = np.nan_to_num(f_skew, nan=0.0, posinf=0.0, neginf=0.0)
    f_kurt = np.nan_to_num(f_kurt, nan=0.0, posinf=0.0, neginf=0.0)

    base_feats = [
        ("morph_mean", "Basic mean amplitude of beat", f_mean),
        ("morph_std", "Standard deviation of beat amplitude", f_std),
        ("morph_median", "Median amplitude of beat", f_median),
        ("morph_min", "Minimum amplitude of beat waveform", f_min),
        ("morph_max", "Maximum amplitude of beat waveform", f_max),
        ("morph_ptp", "Peak-to-peak amplitude (max - min)", f_ptp),
        ("morph_rms", "Root Mean Square amplitude", f_rms),
        ("morph_energy", "Total signal energy (sum of squared amplitudes)", f_energy),
        ("morph_abs_mean", "Mean of absolute amplitudes", f_abs_mean),
        ("morph_skewness", "Statistical skewness of amplitude distribution", f_skew),
        ("morph_kurtosis", "Statistical kurtosis of amplitude distribution", f_kurt),
    ]

    for name, desc, arr in base_feats:
        schema.add_feature(name=name, group="morphology", description=desc)
        features_list.append(arr.reshape(-1, 1))

    # 2. Percentile features
    for p in percentiles:
        name = f"morph_p{p:02d}"
        desc = f"{p}th percentile of amplitude distribution"
        p_val = np.percentile(beats, p, axis=1)
        schema.add_feature(name=name, group="morphology", description=desc)
        features_list.append(p_val.reshape(-1, 1))

    # 3. Zero-crossing count
    zero_crossings = np.sum(np.diff(np.signbit(beats), axis=1), axis=1)
    schema.add_feature(
        name="morph_zero_crossings",
        group="morphology",
        description="Number of zero-level baseline crossings",
    )
    features_list.append(zero_crossings.reshape(-1, 1))

    # 4. First-difference derivative statistics
    diff1 = np.diff(beats, axis=1)
    diff1_mean = np.mean(diff1, axis=1)
    diff1_std = np.std(diff1, axis=1)
    diff1_rms = np.sqrt(np.mean(diff1**2, axis=1))
    diff1_max_abs = np.max(np.abs(diff1), axis=1)

    diff_feats = [
        ("morph_diff1_mean", "Mean of first derivative (sample-to-sample diff)", diff1_mean),
        ("morph_diff1_std", "Standard deviation of first derivative", diff1_std),
        ("morph_diff1_rms", "Root Mean Square of first derivative", diff1_rms),
        ("morph_diff1_max_abs", "Maximum absolute first derivative (slope magnitude)", diff1_max_abs),
    ]

    for name, desc, arr in diff_feats:
        schema.add_feature(name=name, group="morphology", description=desc)
        features_list.append(arr.reshape(-1, 1))

    morph_matrix = np.hstack(features_list).astype(np.float32)
    return morph_matrix
