"""Feature Group B: Shape descriptors, extrema locations, area distributions, and pre/post energy ratios."""

from typing import List
import numpy as np

from ecg_arrhythmia.features.schema import FeatureSchema
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def extract_shape_features(
    beats: np.ndarray,
    schema: FeatureSchema,
    pre_samples: int = 72,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Extract temporal shape, area, and energy asymmetry features from beat waveforms.

    Args:
        beats: 2D array of shape (N, 216).
        schema: FeatureSchema object to register feature definitions.
        pre_samples: Sample index corresponding to annotation peak center (default: 72).
        eps: Small scalar to prevent division by zero.

    Returns:
        2D array of shape (N, D_shape) containing float32 feature values.
    """
    if beats.ndim != 2:
        raise ValueError(f"Beats array must be 2D, got shape {beats.shape}")

    n_beats, n_samples = beats.shape
    features_list: List[np.ndarray] = []

    # 1. Extrema locations
    max_loc = np.argmax(beats, axis=1)
    min_loc = np.argmin(beats, axis=1)
    norm_max_loc = max_loc / float(n_samples)
    norm_min_loc = min_loc / float(n_samples)

    # 2. Area descriptors
    pos_beats = np.maximum(beats, 0.0)
    neg_beats = np.abs(np.minimum(beats, 0.0))

    pos_area = np.sum(pos_beats, axis=1)
    neg_area = np.sum(neg_beats, axis=1)
    total_abs_area = pos_area + neg_area
    area_ratio = pos_area / (neg_area + eps)

    # 3. Energy before and after annotation peak (pre_samples=72)
    pre_energy = np.sum(beats[:, :pre_samples] ** 2, axis=1)
    post_energy = np.sum(beats[:, pre_samples:] ** 2, axis=1)
    energy_ratio_pre_post = pre_energy / (post_energy + eps)

    shape_feats = [
        ("shape_max_location", "Sample index of maximum peak amplitude (0..215)", max_loc.astype(np.float32)),
        ("shape_min_location", "Sample index of minimum valley amplitude (0..215)", min_loc.astype(np.float32)),
        ("shape_norm_max_location", "Normalized location of maximum amplitude (0.0..1.0)", norm_max_loc.astype(np.float32)),
        ("shape_norm_min_location", "Normalized location of minimum amplitude (0.0..1.0)", norm_min_loc.astype(np.float32)),
        ("shape_pos_area", "Area under positive signal deflection", pos_area.astype(np.float32)),
        ("shape_neg_area", "Area under negative signal deflection", neg_area.astype(np.float32)),
        ("shape_total_abs_area", "Total absolute area of beat waveform", total_abs_area.astype(np.float32)),
        ("shape_area_ratio", "Ratio of positive area to negative area", area_ratio.astype(np.float32)),
        ("shape_pre_peak_energy", "Signal energy in pre-annotation window (0..72)", pre_energy.astype(np.float32)),
        ("shape_post_peak_energy", "Signal energy in post-annotation window (72..216)", post_energy.astype(np.float32)),
        ("shape_energy_ratio_pre_post", "Ratio of pre-peak energy to post-peak energy", energy_ratio_pre_post.astype(np.float32)),
    ]

    for name, desc, arr in shape_feats:
        schema.add_feature(name=name, group="shape", description=desc)
        features_list.append(arr.reshape(-1, 1))

    shape_matrix = np.hstack(features_list).astype(np.float32)
    return shape_matrix
