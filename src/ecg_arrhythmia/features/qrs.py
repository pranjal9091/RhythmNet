"""Feature Group C: Waveform QRS morphology width proxies and slope metrics.

Scientific Methodological Note:
These features represent mathematical waveform-width and slope proxies computed on
annotation-centered 216-sample beat windows. They are engineered proxy descriptors,
not clinically validated QRS duration measurements.
"""

from typing import List
import numpy as np

from ecg_arrhythmia.features.schema import FeatureSchema
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def _compute_width_at_fraction(beats: np.ndarray, fraction: float) -> np.ndarray:
    """Compute number of samples where amplitude exceeds fraction * max_amplitude per beat."""
    max_amps = np.max(beats, axis=1, keepdims=True)
    thresholds = fraction * max_amps
    # Count samples where beat amplitude >= threshold
    above_thresh = (beats >= thresholds)
    widths = np.sum(above_thresh, axis=1)
    return widths.astype(np.float32)


def extract_qrs_proxy_features(
    beats: np.ndarray,
    schema: FeatureSchema,
    fractions: List[float] = [0.25, 0.50, 0.75],
    pre_samples: int = 72,
    early_end: int = 54,
    central_end: int = 90,
) -> np.ndarray:
    """Extract QRS-width proxy descriptors, slope metrics, and sub-window energy distribution.

    Args:
        beats: 2D array of shape (N, 216).
        schema: FeatureSchema object to register feature definitions.
        fractions: List of amplitude fraction thresholds for width proxies.
        pre_samples: Annotation peak sample index (default: 72).
        early_end: End of early sub-window (default: 54).
        central_end: End of central QRS sub-window (default: 90).

    Returns:
        2D array of shape (N, D_qrs) containing float32 feature values.
    """
    if beats.ndim != 2:
        raise ValueError(f"Beats array must be 2D, got shape {beats.shape}")

    n_beats, n_samples = beats.shape
    features_list: List[np.ndarray] = []

    # 1. QRS width proxies at amplitude fractions
    for frac in fractions:
        name = f"qrs_width_{int(frac * 100)}pct"
        desc = f"QRS width proxy: number of samples above {int(frac * 100)}% max amplitude"
        w_arr = _compute_width_at_fraction(beats, frac)
        schema.add_feature(name=name, group="qrs", description=desc)
        features_list.append(w_arr.reshape(-1, 1))

    # 2. Slope metrics
    diff1 = np.diff(beats, axis=1)
    max_slope = np.max(np.abs(diff1), axis=1)

    # Slope before annotation center (samples 0..72) vs after center (72..216)
    slope_pre = np.max(np.abs(diff1[:, :pre_samples]), axis=1) if pre_samples > 1 else max_slope
    slope_post = np.max(np.abs(diff1[:, pre_samples:]), axis=1) if pre_samples < n_samples - 1 else max_slope

    # 3. Peak-to-valley distance (separation in samples)
    max_loc = np.argmax(beats, axis=1)
    min_loc = np.argmin(beats, axis=1)
    pos_neg_sep = np.abs(max_loc - min_loc).astype(np.float32)

    # 4. Sub-window energy distributions
    early_energy = np.sum(beats[:, :early_end] ** 2, axis=1)
    central_energy = np.sum(beats[:, early_end:central_end] ** 2, axis=1)
    late_energy = np.sum(beats[:, central_end:] ** 2, axis=1)

    qrs_feats = [
        ("qrs_max_slope", "Maximum absolute first derivative across beat", max_slope.astype(np.float32)),
        ("qrs_slope_before_center", "Maximum absolute slope prior to annotation center", slope_pre.astype(np.float32)),
        ("qrs_slope_after_center", "Maximum absolute slope following annotation center", slope_post.astype(np.float32)),
        ("qrs_pos_neg_peak_separation", "Distance in samples between max peak and min valley", pos_neg_sep),
        ("qrs_early_window_energy", "Signal energy in P-wave / early window (samples 0..54)", early_energy.astype(np.float32)),
        ("qrs_central_window_energy", "Signal energy in QRS central window (samples 54..90)", central_energy.astype(np.float32)),
        ("qrs_late_window_energy", "Signal energy in T-wave / late window (samples 90..216)", late_energy.astype(np.float32)),
    ]

    for name, desc, arr in qrs_feats:
        schema.add_feature(name=name, group="qrs", description=desc)
        features_list.append(arr.reshape(-1, 1))

    qrs_matrix = np.hstack(features_list).astype(np.float32)
    return qrs_matrix
