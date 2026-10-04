"""Feature Group D: Record-local sequential rhythm context and RR interval features.

Leakage Prevention Safeguards:
1. RR features are computed STRICTLY within individual patient records. They NEVER cross record boundaries.
2. RR features are computed purely from chronological sample indices. They are NEVER conditioned on target labels.
3. First and last beats in a record receive np.nan for unavailable neighbors (never 0 or fabricated).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ecg_arrhythmia.features.schema import FeatureSchema
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def extract_rr_features(
    df_meta: pd.DataFrame,
    schema: FeatureSchema,
    fs: int = 360,
    local_window_beats: int = 5,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Extract record-local sequential RR interval features and context flags.

    Args:
        df_meta: Metadata DataFrame containing 'record_id', 'sample_index'.
        schema: FeatureSchema object to register feature definitions.
        fs: Sampling frequency in Hz (default: 360).
        local_window_beats: Number of beats in local sliding window for mean/std baseline.
        eps: Small scalar to prevent division by zero.

    Returns:
        2D array of shape (N, D_rr) containing feature values (with np.nan for missing context).
    """
    n_beats = len(df_meta)
    
    # Store original row index to guarantee exact alignment after grouping
    df = df_meta.copy()
    df["orig_idx"] = np.arange(n_beats)
    df = df.sort_values(by=["record_id", "sample_index"]).reset_index(drop=True)

    rr_prev = np.full(n_beats, np.nan, dtype=np.float64)
    rr_next = np.full(n_beats, np.nan, dtype=np.float64)
    
    record_ids = df["record_id"].values
    sample_indices = df["sample_index"].values

    # 1. Compute sequential RR intervals within each record
    for i in range(1, n_beats):
        if record_ids[i] == record_ids[i - 1]:
            diff_samples = sample_indices[i] - sample_indices[i - 1]
            if diff_samples > 0:
                rr_prev[i] = diff_samples / float(fs)

    for i in range(n_beats - 1):
        if record_ids[i] == record_ids[i + 1]:
            diff_samples = sample_indices[i + 1] - sample_indices[i]
            if diff_samples > 0:
                rr_next[i] = diff_samples / float(fs)

    df["rr_prev"] = rr_prev
    df["rr_next"] = rr_next

    # 2. Compute local sliding window RR reference metrics per record
    half_win = local_window_beats // 2

    local_mean = np.full(n_beats, np.nan, dtype=np.float64)
    local_std = np.full(n_beats, np.nan, dtype=np.float64)
    local_median = np.full(n_beats, np.nan, dtype=np.float64)
    local_min = np.full(n_beats, np.nan, dtype=np.float64)
    local_max = np.full(n_beats, np.nan, dtype=np.float64)

    # Process grouped by record ID to ensure windows never cross boundaries
    for _, group in df.groupby("record_id", sort=False):
        group_indices = group.index.to_numpy()
        group_rr = group["rr_prev"].to_numpy()

        for g_idx_pos, global_idx in enumerate(group_indices):
            start_p = max(0, g_idx_pos - half_win)
            end_p = min(len(group_indices), g_idx_pos + half_win + 1)
            win_vals = group_rr[start_p:end_p]
            valid_vals = win_vals[~np.isnan(win_vals)]

            if len(valid_vals) > 0:
                local_mean[global_idx] = np.mean(valid_vals)
                local_std[global_idx] = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
                local_median[global_idx] = np.median(valid_vals)
                local_min[global_idx] = np.min(valid_vals)
                local_max[global_idx] = np.max(valid_vals)

    # 3. Derived ratio & normalized RR features
    rr_ratio_prev = rr_prev / (local_mean + eps)
    rr_ratio_next = rr_next / (local_mean + eps)
    norm_rr_prev = (rr_prev - local_mean) / (local_std + eps)
    norm_rr_next = (rr_next - local_mean) / (local_std + eps)

    # 4. Context availability flags
    has_prev_rr = (~np.isnan(rr_prev)).astype(np.float64)
    has_next_rr = (~np.isnan(rr_next)).astype(np.float64)
    has_full_rr_context = (has_prev_rr * has_next_rr).astype(np.float64)

    df["rr_ratio_prev"] = rr_ratio_prev
    df["rr_ratio_next"] = rr_ratio_next
    df["local_rr_mean"] = local_mean
    df["local_rr_std"] = local_std
    df["local_rr_median"] = local_median
    df["local_rr_min"] = local_min
    df["local_rr_max"] = local_max
    df["normalized_rr_prev"] = norm_rr_prev
    df["normalized_rr_next"] = norm_rr_next
    df["has_prev_rr"] = has_prev_rr
    df["has_next_rr"] = has_next_rr
    df["has_full_rr_context"] = has_full_rr_context

    # 5. Restore original dataframe ordering
    df = df.sort_values(by="orig_idx").reset_index(drop=True)

    rr_feat_defs = [
        ("rr_prev", "Sequential RR interval to previous beat in seconds"),
        ("rr_next", "Sequential RR interval to next beat in seconds"),
        ("rr_ratio_prev", "Ratio of rr_prev to local mean RR interval"),
        ("rr_ratio_next", "Ratio of rr_next to local mean RR interval"),
        ("local_rr_mean", "Mean RR interval across local record window"),
        ("local_rr_std", "Standard deviation of RR intervals in local window"),
        ("local_rr_median", "Median RR interval in local window"),
        ("local_rr_min", "Minimum RR interval in local window"),
        ("local_rr_max", "Maximum RR interval in local window"),
        ("normalized_rr_prev", "Z-score normalized rr_prev relative to local window"),
        ("normalized_rr_next", "Z-score normalized rr_next relative to local window"),
        ("has_prev_rr", "Flag (1.0 or 0.0) indicating availability of previous beat context"),
        ("has_next_rr", "Flag (1.0 or 0.0) indicating availability of next beat context"),
        ("has_full_rr_context", "Flag (1.0 or 0.0) indicating availability of both prev and next context"),
    ]

    features_list = []
    for name, desc in rr_feat_defs:
        schema.add_feature(name=name, group="rr", description=desc)
        features_list.append(df[name].values.reshape(-1, 1))

    rr_matrix = np.hstack(features_list).astype(np.float32)
    return rr_matrix
