"""Tests for signal filtering, lead selection, beat segmentation, and per-beat normalization."""

import numpy as np
import pytest

from ecg_arrhythmia.preprocessing.filtering import (
    filter_ecg,
    select_ecg_lead,
    validate_sampling_rate,
)
from ecg_arrhythmia.preprocessing.normalization import normalize_beats_zscore
from ecg_arrhythmia.preprocessing.segmentation import segment_beats_from_record


# --- FILTERING TESTS ---

def test_filter_ecg_output_shape_and_values():
    """Verify that filter_ecg preserves array length and produces finite float32 outputs."""
    np.random.seed(42)
    raw_signal = np.random.randn(3600).astype(np.float32)
    filtered = filter_ecg(raw_signal, fs=360, low_hz=0.5, high_hz=40.0, order=4)

    assert len(filtered) == len(raw_signal)
    assert filtered.dtype == np.float32
    assert not np.isnan(filtered).any()
    assert not np.isinf(filtered).any()


def test_filter_ecg_deterministic():
    """Verify zero-phase filter is completely deterministic."""
    raw_signal = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
    f1 = filter_ecg(raw_signal, fs=360)
    f2 = filter_ecg(raw_signal, fs=360)

    np.testing.assert_array_almost_equal(f1, f2)


def test_filter_ecg_invalid_cutoffs():
    """Verify ValueError when cutoff frequencies are invalid."""
    raw_signal = np.ones(1000, dtype=np.float32)
    with pytest.raises(ValueError):
        filter_ecg(raw_signal, fs=360, low_hz=45.0, high_hz=40.0)  # low > high

    with pytest.raises(ValueError):
        filter_ecg(raw_signal, fs=360, low_hz=-1.0, high_hz=40.0)  # low <= 0


def test_validate_sampling_rate():
    """Verify sampling rate validator."""
    class DummyRecord:
        record_name = "100"
        fs = 360

    assert validate_sampling_rate(DummyRecord(), expected_fs=360) == 360

    class BadRecord:
        record_name = "999"
        fs = 250

    with pytest.raises(ValueError, match="sampling rate is 250 Hz, expected 360 Hz"):
        validate_sampling_rate(BadRecord(), expected_fs=360)


# --- SEGMENTATION TESTS ---

def test_segmentation_window_length_and_alignment():
    """Verify beat segmentation extracts exact pre_samples + post_samples windows."""
    fs = 360
    signal_len = 3600
    signal = np.sin(np.linspace(0, 50, signal_len)).astype(np.float32)

    ann_samples = np.array([200, 1000, 2000], dtype=int)
    ann_symbols = ["N", "V", "A"]

    waveforms, metadata, stats = segment_beats_from_record(
        signal=signal,
        annotation_samples=ann_samples,
        annotation_symbols=ann_symbols,
        record_id="100",
        lead_name="MLII",
        fs=fs,
        pre_samples=72,
        post_samples=144,
    )

    assert waveforms.shape == (3, 216)
    assert len(metadata) == 3
    assert stats.extracted_beats == 3
    assert stats.boundary_excluded == 0
    assert stats.unmapped_excluded == 0
    assert metadata[0]["aami_class"] == "N"
    assert metadata[1]["aami_class"] == "V"
    assert metadata[2]["aami_class"] == "S"  # 'A' maps to 'S'


def test_segmentation_boundary_handling():
    """Verify that beats too close to boundaries are safely excluded without error."""
    signal_len = 1000
    signal = np.random.randn(signal_len).astype(np.float32)

    # 30 is too close to start (< 72), 980 is too close to end (> 1000 - 144 = 856), 500 is valid
    ann_samples = np.array([30, 500, 980], dtype=int)
    ann_symbols = ["N", "N", "N"]

    waveforms, metadata, stats = segment_beats_from_record(
        signal=signal,
        annotation_samples=ann_samples,
        annotation_symbols=ann_symbols,
        record_id="100",
        lead_name="MLII",
        fs=360,
        pre_samples=72,
        post_samples=144,
    )

    assert waveforms.shape == (1, 216)
    assert len(metadata) == 1
    assert stats.extracted_beats == 1
    assert stats.boundary_excluded == 2
    assert metadata[0]["sample_index"] == 500


def test_segmentation_unmapped_symbol_exclusion():
    """Verify auxiliary markers (e.g. '+', '~') are excluded and tracked."""
    signal = np.random.randn(3600).astype(np.float32)
    ann_samples = np.array([500, 1000, 1500], dtype=int)
    ann_symbols = ["N", "+", "~"]  # '+' and '~' are non-beat markers

    waveforms, metadata, stats = segment_beats_from_record(
        signal=signal,
        annotation_samples=ann_samples,
        annotation_symbols=ann_symbols,
        record_id="100",
        lead_name="MLII",
    )

    assert waveforms.shape == (1, 216)
    assert len(metadata) == 1
    assert stats.unmapped_excluded == 2


# --- NORMALIZATION TESTS ---

def test_normalize_beats_zscore_properties():
    """Verify per-beat z-score normalization results in mean ~0 and std ~1."""
    np.random.seed(42)
    beats = np.random.randn(10, 216) * 5.0 + 10.0  # arbitrary amplitude & mean offset
    norm_beats = normalize_beats_zscore(beats)

    assert norm_beats.shape == beats.shape
    assert norm_beats.dtype == np.float32

    for i in range(10):
        mean_i = np.mean(norm_beats[i])
        std_i = np.std(norm_beats[i])
        assert abs(mean_i) < 1e-4, f"Beat {i} mean is {mean_i}, expected ~0"
        assert abs(std_i - 1.0) < 1e-4, f"Beat {i} std is {std_i}, expected ~1"


def test_normalize_beats_zscore_empty_and_flat():
    """Verify normalization gracefully handles empty array and flat signals."""
    empty = np.empty((0, 216), dtype=np.float32)
    norm_empty = normalize_beats_zscore(empty)
    assert norm_empty.shape == (0, 216)

    flat = np.ones((2, 216), dtype=np.float32) * 3.0
    norm_flat = normalize_beats_zscore(flat)
    assert not np.isnan(norm_flat).any()
