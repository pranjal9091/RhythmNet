"""Tests for WFDB data access layer and MIT-BIH record loading."""

from pathlib import Path
import numpy as np
import pytest

from ecg_arrhythmia.data.loader import MITBIHLoader, RecordMetadata
from ecg_arrhythmia.utils.config import load_config


@pytest.fixture(scope="module")
def data_loader():
    """Module-level loader fixture using test or configured raw data directory."""
    cfg = load_config("configs/data.yaml")
    raw_dir = cfg.get("dataset", {}).get("raw_dir", "data/raw/mitdb")
    loader = MITBIHLoader(raw_data_dir=raw_dir)
    # Ensure record 100 is available for testing
    loader.download_record("100")
    return loader


def test_download_and_load_record(data_loader: MITBIHLoader):
    """Verify that record 100 loads and has expected signal properties."""
    record = data_loader.load_record("100", sampfrom=0, sampto=1000)

    assert record is not None
    assert record.record_name == "100"
    assert record.fs == 360
    assert record.n_sig == 2
    assert record.p_signal.shape == (1000, 2)
    assert not np.isnan(record.p_signal).any(), "Signal should not contain NaN values"


def test_load_annotations(data_loader: MITBIHLoader):
    """Verify that annotation file for record 100 loads correctly."""
    ann = data_loader.load_annotations("100", sampfrom=0, sampto=10000)

    assert ann is not None
    assert len(ann.sample) > 0
    assert len(ann.symbol) == len(ann.sample)
    # R-peak sample indices must be strictly increasing
    assert np.all(np.diff(ann.sample) > 0)


def test_record_metadata(data_loader: MITBIHLoader):
    """Verify RecordMetadata dataclass properties for record 100."""
    meta = data_loader.get_record_metadata("100")

    assert isinstance(meta, RecordMetadata)
    assert meta.record_id == "100"
    assert meta.sampling_rate == 360
    assert meta.n_channels == 2
    assert "MLII" in meta.channels or "V5" in meta.channels
    assert meta.duration_sec > 1700.0  # Approx 30 minutes (1800 sec)
    assert meta.has_annotations is True


def test_is_record_present(data_loader: MITBIHLoader):
    """Verify record presence detection."""
    assert data_loader.is_record_present("100") is True
    assert data_loader.is_record_present("nonexistent_record_999") is False
