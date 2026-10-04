"""Unit tests for Exploratory Data Analysis (EDA) and dataset auditing modules."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ecg_arrhythmia.eda.distribution import analyze_class_distributions
from ecg_arrhythmia.eda.patient_audit import (
    analyze_minority_patient_coverage,
    analyze_patient_contributions,
)
from ecg_arrhythmia.eda.rr_analysis import compute_rr_intervals, compute_rr_statistics
from ecg_arrhythmia.eda.statistics import (
    compare_partition_distributions,
    run_data_quality_audit,
)
from ecg_arrhythmia.eda.waveform_analysis import compute_morphology_statistics
from ecg_arrhythmia.utils.config import load_config


@pytest.fixture
def sample_eda_data():
    """Create synthetic metadata and waveform arrays for testing EDA logic."""
    df_train = pd.DataFrame(
        {
            "beat_id": [f"train_{i:04d}" for i in range(10)],
            "record_id": ["100"] * 5 + ["101"] * 5,
            "sample_index": [100, 500, 900, 1300, 1700, 200, 600, 1000, 1400, 1800],
            "original_symbol": ["N", "N", "V", "N", "A", "N", "N", "N", "V", "F"],
            "aami_class": ["N", "N", "V", "N", "S", "N", "N", "N", "V", "F"],
            "lead": ["MLII"] * 10,
            "fs": [360] * 10,
        }
    )

    df_test = pd.DataFrame(
        {
            "beat_id": [f"test_{i:04d}" for i in range(5)],
            "record_id": ["200"] * 5,
            "sample_index": [300, 700, 1100, 1500, 1900],
            "original_symbol": ["N", "V", "N", "N", "Q"],
            "aami_class": ["N", "V", "N", "N", "Q"],
            "lead": ["MLII"] * 5,
            "fs": [360] * 5,
        }
    )

    np.random.seed(42)
    beats_train = np.random.randn(10, 216).astype(np.float32)
    beats_test = np.random.randn(5, 216).astype(np.float32)

    dfs_dict = {"train": df_train, "test": df_test}
    beats_dict = {"train": beats_train, "test": beats_test}

    return dfs_dict, beats_dict


def test_analyze_class_distributions(sample_eda_data):
    """Verify class distribution summary calculation."""
    dfs_dict, _ = sample_eda_data
    df_dist = analyze_class_distributions(dfs_dict)

    assert len(df_dist) == 2
    assert "partition" in df_dist.columns

    train_row = df_dist[df_dist["partition"] == "train"].iloc[0]
    assert train_row["total_beats"] == 10
    assert train_row["count_N"] == 6
    assert train_row["count_V"] == 2
    assert train_row["count_S"] == 1
    assert train_row["count_F"] == 1
    assert train_row["count_Q"] == 0

    pct_sum = (
        train_row["pct_N"]
        + train_row["pct_S"]
        + train_row["pct_V"]
        + train_row["pct_F"]
        + train_row["pct_Q"]
    )
    assert abs(pct_sum - 100.0) < 1e-4


def test_analyze_patient_contributions(sample_eda_data):
    """Verify record-level contribution auditing."""
    dfs_dict, _ = sample_eda_data
    df_pat = analyze_patient_contributions(dfs_dict)

    assert len(df_pat) == 3  # records 100, 101, 200
    rec_100 = df_pat[df_pat["record_id"] == "100"].iloc[0]
    assert rec_100["total_beats"] == 5
    assert rec_100["count_N"] == 3
    assert rec_100["count_V"] == 1
    assert rec_100["count_S"] == 1


def test_analyze_minority_patient_coverage(sample_eda_data):
    """Verify minority class patient coverage metrics."""
    dfs_dict, _ = sample_eda_data
    df_cov = analyze_minority_patient_coverage(dfs_dict)

    assert not df_cov.empty
    train_v = df_cov[(df_cov["partition"] == "train") & (df_cov["aami_class"] == "V")].iloc[0]
    assert train_v["contributing_records"] == 2  # both 100 and 101 have V beats
    assert train_v["total_beats"] == 2


def test_compute_rr_intervals_no_cross_record():
    """Verify RR intervals are calculated strictly within record boundaries."""
    df_meta = pd.DataFrame(
        {
            "record_id": ["100", "100", "101", "101"],
            "sample_index": [100, 460, 200, 560],  # 360 samples apart = 1.0 sec
            "aami_class": ["N", "N", "N", "N"],
        }
    )

    df_rr = compute_rr_intervals(df_meta, fs=360)

    # First beat of rec 100 has NaN RR
    assert np.isnan(df_rr.loc[0, "rr_sec"])
    # Second beat of rec 100 is 360 samples / 360 Hz = 1.0 sec
    assert abs(df_rr.loc[1, "rr_sec"] - 1.0) < 1e-4

    # First beat of rec 101 has NaN RR (must NOT compute diff from rec 100 beat 2!)
    assert np.isnan(df_rr.loc[2, "rr_sec"])
    # Second beat of rec 101 is 1.0 sec
    assert abs(df_rr.loc[3, "rr_sec"] - 1.0) < 1e-4


def test_compute_morphology_statistics(sample_eda_data):
    """Verify non-ML morphology statistics calculation."""
    dfs_dict, beats_dict = sample_eda_data
    df_morph = compute_morphology_statistics(beats_dict, dfs_dict)

    assert not df_morph.empty
    assert "p2p_mean" in df_morph.columns
    assert "rms_mean" in df_morph.columns
    assert not df_morph["p2p_mean"].isna().any()


def test_run_data_quality_audit(sample_eda_data):
    """Verify 10-point data quality audit passes on clean synthetic dataset."""
    dfs_dict, beats_dict = sample_eda_data
    
    # Add required columns for audit
    for split_name, df in dfs_dict.items():
        df["rr_sec"] = 0.8

    df_audit = run_data_quality_audit(beats_dict, dfs_dict)

    assert not df_audit.empty
    critical_failures = df_audit[(df_audit["severity"] == "CRITICAL") & (~df_audit["passed"])]
    assert len(critical_failures) == 0, f"Critical failures found: {critical_failures}"


def test_eda_config_loading():
    """Verify loading configs/eda.yaml configuration."""
    cfg = load_config("configs/eda.yaml")
    assert "sampling_frequency" in cfg
    assert cfg["sampling_frequency"] == 360
    assert "plots" in cfg
    assert "dpi" in cfg["plots"]
