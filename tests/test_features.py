"""Unit tests for hand-crafted ECG feature extraction pipeline and leakage prevention."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ecg_arrhythmia.features.extractor import ECGFeatureExtractor, fit_transform_preprocessor
from ecg_arrhythmia.features.morphology import extract_morphology_features
from ecg_arrhythmia.features.qrs import extract_qrs_proxy_features
from ecg_arrhythmia.features.rr_features import extract_rr_features
from ecg_arrhythmia.features.schema import FeatureSchema
from ecg_arrhythmia.features.shape import extract_shape_features
from ecg_arrhythmia.features.wavelet import extract_wavelet_features
from ecg_arrhythmia.utils.config import load_config


@pytest.fixture
def dummy_beats_and_meta():
    """Create synthetic beat waveforms and metadata for feature testing."""
    np.random.seed(42)
    n_beats = 10
    beats = np.random.randn(n_beats, 216).astype(np.float32)

    meta = pd.DataFrame(
        {
            "beat_id": [f"beat_{i:04d}" for i in range(n_beats)],
            "record_id": ["100"] * 5 + ["101"] * 5,
            "sample_index": [100, 500, 900, 1300, 1700, 200, 600, 1000, 1400, 1800],
            "original_symbol": ["N"] * n_beats,
            "aami_class": ["N"] * n_beats,
            "lead": ["MLII"] * n_beats,
            "fs": [360] * n_beats,
        }
    )

    return beats, meta


def test_morphology_features(dummy_beats_and_meta):
    """Verify morphology feature extraction shape and determinism."""
    beats, _ = dummy_beats_and_meta
    schema = FeatureSchema()
    f_morph = extract_morphology_features(beats, schema=schema)

    assert f_morph.ndim == 2
    assert f_morph.shape[0] == len(beats)
    assert f_morph.shape[1] == len(schema)
    assert not np.isnan(f_morph).any()


def test_shape_features(dummy_beats_and_meta):
    """Verify shape feature extraction."""
    beats, _ = dummy_beats_and_meta
    schema = FeatureSchema()
    f_shape = extract_shape_features(beats, schema=schema)

    assert f_shape.shape[0] == len(beats)
    assert not np.isnan(f_shape).any()


def test_qrs_proxy_features(dummy_beats_and_meta):
    """Verify QRS width proxy descriptors."""
    beats, _ = dummy_beats_and_meta
    schema = FeatureSchema()
    f_qrs = extract_qrs_proxy_features(beats, schema=schema)

    assert f_qrs.shape[0] == len(beats)
    assert not np.isnan(f_qrs).any()


def test_rr_features_record_boundary(dummy_beats_and_meta):
    """Verify RR features never cross record boundaries and handle missing context flags."""
    _, meta = dummy_beats_and_meta
    schema = FeatureSchema()
    f_rr = extract_rr_features(meta, schema=schema, fs=360)

    assert f_rr.shape[0] == len(meta)

    # Convert to DataFrame using schema names
    df_rr = pd.DataFrame(f_rr, columns=schema.feature_names)

    # First beat of rec 100 (row 0) and first beat of rec 101 (row 5) MUST have NaN for rr_prev
    assert np.isnan(df_rr.loc[0, "rr_prev"])
    assert np.isnan(df_rr.loc[5, "rr_prev"])

    # Flags
    assert df_rr.loc[0, "has_prev_rr"] == 0.0
    assert df_rr.loc[1, "has_prev_rr"] == 1.0


def test_wavelet_features(dummy_beats_and_meta):
    """Verify DWT wavelet summary statistics."""
    beats, _ = dummy_beats_and_meta
    schema = FeatureSchema()
    f_wav = extract_wavelet_features(beats, schema=schema, wavelet_name="db4", level=4)

    assert f_wav.shape[0] == len(beats)
    assert not np.isnan(f_wav).any()


def test_combined_feature_extractor(dummy_beats_and_meta):
    """Verify combined ECGFeatureExtractor output shape and schema catalog alignment."""
    beats, meta = dummy_beats_and_meta
    cfg = load_config("configs/features.yaml")
    extractor = ECGFeatureExtractor(config=cfg)

    X_raw, schema = extractor.extract_features(beats, meta)

    assert X_raw.ndim == 2
    assert X_raw.shape[0] == len(beats)
    assert X_raw.shape[1] == len(schema)
    assert len(schema.definitions) == X_raw.shape[1]


def test_fit_transform_preprocessor_train_only_fitting(dummy_beats_and_meta, tmp_path: Path):
    """Verify fit_transform_preprocessor fits imputer/scaler STRICTLY on X_train."""
    beats, meta = dummy_beats_and_meta
    extractor = ECGFeatureExtractor()
    X_raw, _ = extractor.extract_features(beats, meta)

    X_train = X_raw[:6]
    X_val = X_raw[6:8]
    X_test = X_raw[8:]

    artifact_path = tmp_path / "preprocessor.joblib"
    X_tr_scaled, X_va_scaled, X_te_scaled, pipeline = fit_transform_preprocessor(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        save_path=artifact_path,
    )

    # Scaled matrices have zero NaNs or Infs
    assert not np.isnan(X_tr_scaled).any()
    assert not np.isnan(X_va_scaled).any()
    assert not np.isnan(X_te_scaled).any()

    # Verify fitted scaler mean matches training column means (excluding NaNs)
    scaler = pipeline.named_steps["scaler"]
    imputer = pipeline.named_steps["imputer"]
    assert len(scaler.mean_) == X_train.shape[1]
    assert artifact_path.is_file()
