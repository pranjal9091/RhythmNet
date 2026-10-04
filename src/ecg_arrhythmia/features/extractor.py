"""Combined ECG feature extraction pipeline and leakage-safe scikit-learn preprocessor."""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecg_arrhythmia.features.morphology import extract_morphology_features
from ecg_arrhythmia.features.qrs import extract_qrs_proxy_features
from ecg_arrhythmia.features.rr_features import extract_rr_features
from ecg_arrhythmia.features.schema import FeatureSchema
from ecg_arrhythmia.features.shape import extract_shape_features
from ecg_arrhythmia.features.wavelet import extract_wavelet_features
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


class ECGFeatureExtractor:
    """Extractor for converting ECG beat arrays and metadata into hand-crafted feature matrices."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract_features(
        self, beats: np.ndarray, metadata: pd.DataFrame
    ) -> Tuple[np.ndarray, FeatureSchema]:
        """Extract all configured feature groups deterministically.

        Args:
            beats: 2D numpy array of shape (N, 216).
            metadata: Metadata DataFrame of N rows.

        Returns:
            Tuple of (raw_feature_matrix (N, D), feature_schema)
        """
        assert len(beats) == len(metadata), (
            f"Row count mismatch between beats ({len(beats)}) and metadata ({len(metadata)})"
        )

        schema = FeatureSchema()
        feature_blocks = []

        # 1. Group A: Morphology / Amplitude
        morph_cfg = self.config.get("morphology", {})
        if morph_cfg.get("enabled", True):
            pcts = morph_cfg.get("percentiles", [5, 10, 25, 50, 75, 90, 95])
            f_morph = extract_morphology_features(beats, schema=schema, percentiles=pcts)
            feature_blocks.append(f_morph)

        # 2. Group B: Shape / Area
        shape_cfg = self.config.get("shape", {})
        if shape_cfg.get("enabled", True):
            f_shape = extract_shape_features(beats, schema=schema)
            feature_blocks.append(f_shape)

        # 3. Group C: QRS Width Proxies
        qrs_cfg = self.config.get("qrs", {})
        if qrs_cfg.get("enabled", True):
            fracs = qrs_cfg.get("amplitude_fractions", [0.25, 0.50, 0.75])
            f_qrs = extract_qrs_proxy_features(beats, schema=schema, fractions=fracs)
            feature_blocks.append(f_qrs)

        # 4. Group D: RR / Rhythm Context
        rr_cfg = self.config.get("rr", {})
        if rr_cfg.get("enabled", True):
            fs = rr_cfg.get("sampling_frequency", 360)
            win_beats = rr_cfg.get("local_window_beats", 5)
            f_rr = extract_rr_features(metadata, schema=schema, fs=fs, local_window_beats=win_beats)
            feature_blocks.append(f_rr)

        # 5. Group E: Wavelet DWT
        wavelet_cfg = self.config.get("wavelet", {})
        if wavelet_cfg.get("enabled", True):
            w_name = wavelet_cfg.get("wavelet_name", "db4")
            w_level = wavelet_cfg.get("decomposition_level", 4)
            f_wavelet = extract_wavelet_features(beats, schema=schema, wavelet_name=w_name, level=w_level)
            feature_blocks.append(f_wavelet)

        X_raw = np.hstack(feature_blocks).astype(np.float32)
        assert X_raw.shape[1] == len(schema), f"Matrix col count ({X_raw.shape[1]}) != schema length ({len(schema)})"

        return X_raw, schema


def fit_transform_preprocessor(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_path: Optional[Path] = None,
    imputer_strategy: str = "median",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """Fit SimpleImputer and StandardScaler STRICTLY on X_train, then transform X_train, X_val, and X_test.

    Leakage Prevention Guarantee:
    Imputation medians and scaling parameters (mean, std) are learned EXCLUSIVELY from X_train.
    Under no circumstances are validation or test features used to fit the pipeline.

    Args:
        X_train: Raw training feature matrix (N_train, D).
        X_val: Raw validation feature matrix (N_val, D).
        X_test: Raw testing feature matrix (N_test, D).
        save_path: Optional file path to save fitted Pipeline object.
        imputer_strategy: Imputation strategy ('median', 'mean').

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, fitted_pipeline)
    """
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", StandardScaler()),
        ]
    )

    logger.info("Fitting SimpleImputer and StandardScaler STRICTLY on Training features...")
    pipeline.fit(X_train)

    X_train_scaled = pipeline.transform(X_train).astype(np.float32)
    X_val_scaled = pipeline.transform(X_val).astype(np.float32)
    X_test_scaled = pipeline.transform(X_test).astype(np.float32)

    # Sanity assertion: Zero NaNs or Infs in scaled matrices
    for name, arr in [("train", X_train_scaled), ("val", X_val_scaled), ("test", X_test_scaled)]:
        assert not np.isnan(arr).any(), f"NaN values found in {name} scaled features!"
        assert not np.isinf(arr).any(), f"Inf values found in {name} scaled features!"

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        logger.info(f"Saved fitted preprocessor artifact to: {save_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, pipeline
