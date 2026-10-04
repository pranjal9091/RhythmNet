"""Feature Group E: Discrete Wavelet Transform (DWT) summary statistics."""

from typing import List
import numpy as np
import pywt

from ecg_arrhythmia.features.schema import FeatureSchema
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def extract_wavelet_features(
    beats: np.ndarray,
    schema: FeatureSchema,
    wavelet_name: str = "db4",
    level: int = 4,
) -> np.ndarray:
    """Extract summary statistics for Discrete Wavelet Transform (DWT) coefficient bands.

    Daubechies 4 ('db4') wavelets closely approximate QRS complex shapes. A 4-level
    decomposition of 216 samples produces approximation coefficients cA4 and detail
    coefficients cD4, cD3, cD2, cD1 across distinct frequency sub-bands.

    Args:
        beats: 2D array of shape (N, 216).
        schema: FeatureSchema object to register feature definitions.
        wavelet_name: PyWavelets wavelet identifier (default: 'db4').
        level: Wavelet decomposition level (default: 4).

    Returns:
        2D array of shape (N, D_wavelet) containing float32 feature values.
    """
    if beats.ndim != 2:
        raise ValueError(f"Beats array must be 2D, got shape {beats.shape}")

    n_beats, n_samples = beats.shape

    # Validate max decomposition level for signal length
    max_level = pywt.dwt_max_level(data_len=n_samples, filter_len=pywt.Wavelet(wavelet_name).dec_len)
    actual_level = min(level, max_level)

    if actual_level < level:
        logger.warning(
            f"Requested DWT level {level} exceeds max level {max_level} for signal len {n_samples}. "
            f"Using level {actual_level}."
        )

    # Perform DWT decomposition for all beats
    # pywt.wavedec with axis=1 returns [cA_level, cD_level, cD_level-1, ..., cD_1]
    coeffs = pywt.wavedec(beats, wavelet=wavelet_name, level=actual_level, axis=1)

    band_names = [f"cA{actual_level}"] + [f"cD{i}" for i in range(actual_level, 0, -1)]

    features_list: List[np.ndarray] = []

    for band_idx, band_arr in enumerate(coeffs):
        b_name = band_names[band_idx]

        b_mean = np.mean(band_arr, axis=1)
        b_std = np.std(band_arr, axis=1)
        b_energy = np.sum(band_arr**2, axis=1)
        b_rms = np.sqrt(np.mean(band_arr**2, axis=1))
        b_abs_mean = np.mean(np.abs(band_arr), axis=1)

        band_feats = [
            (f"dwt_{b_name}_mean", f"Mean of DWT {b_name} coefficient band", b_mean),
            (f"dwt_{b_name}_std", f"Standard deviation of DWT {b_name} coefficient band", b_std),
            (f"dwt_{b_name}_energy", f"Energy (sum of squared coefficients) of DWT {b_name} band", b_energy),
            (f"dwt_{b_name}_rms", f"Root Mean Square of DWT {b_name} coefficient band", b_rms),
            (f"dwt_{b_name}_abs_mean", f"Absolute mean of DWT {b_name} coefficient band", b_abs_mean),
        ]

        for feat_name, desc, arr in band_feats:
            schema.add_feature(name=feat_name, group="wavelet", description=desc)
            features_list.append(arr.reshape(-1, 1))

    wavelet_matrix = np.hstack(features_list).astype(np.float32)
    return wavelet_matrix
