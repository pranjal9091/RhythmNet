"""Tests for utility modules: seeding, configuration, logging, and annotation mapping."""

import random
from pathlib import Path
import numpy as np
import pytest

from ecg_arrhythmia.data.annotations import (
    AAMI_CLASSES,
    AAMI_MAPPING,
    NON_BEAT_SYMBOLS,
    inspect_annotation_symbols,
    is_heartbeat,
    map_symbol_to_aami,
    summarize_aami_distribution,
)
from ecg_arrhythmia.utils.config import get_project_root, load_config, resolve_path
from ecg_arrhythmia.utils.logging import get_logger
from ecg_arrhythmia.utils.seed import seed_everything


def test_seed_everything_reproducibility():
    """Verify that seed_everything produces deterministic pseudorandom sequences."""
    seed_everything(1234)
    py_rand_1 = [random.random() for _ in range(5)]
    np_rand_1 = np.random.rand(5).tolist()

    seed_everything(1234)
    py_rand_2 = [random.random() for _ in range(5)]
    np_rand_2 = np.random.rand(5).tolist()

    assert py_rand_1 == py_rand_2
    assert np_rand_1 == np_rand_2


def test_load_config_valid():
    """Verify loading existing YAML config."""
    cfg = load_config("configs/data.yaml")
    assert "dataset" in cfg
    assert cfg["dataset"]["physionet_id"] == "mitdb"
    assert cfg["dataset"]["sampling_rate_hz"] == 360
    assert len(cfg["records"]["all"]) == 48


def test_load_config_missing():
    """Verify FileNotFoundError on nonexistent configuration path."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent_config.yaml")


def test_get_project_root():
    """Verify project root correctly identifies directory with pyproject.toml."""
    root = get_project_root()
    assert (root / "pyproject.toml").is_file()


def test_get_logger():
    """Verify logger setup."""
    logger = get_logger("test_module")
    assert logger.name == "test_module"
    assert len(logger.handlers) >= 1


def test_aami_mapping_rules():
    """Verify core AAMI EC57 class mappings."""
    # N Class
    for sym in ["N", "L", "R", "e", "j"]:
        assert map_symbol_to_aami(sym) == "N", f"Symbol {sym} must map to N"
        assert is_heartbeat(sym) is True

    # S Class
    for sym in ["A", "a", "J", "S"]:
        assert map_symbol_to_aami(sym) == "S", f"Symbol {sym} must map to S"
        assert is_heartbeat(sym) is True

    # V Class
    for sym in ["V", "E"]:
        assert map_symbol_to_aami(sym) == "V", f"Symbol {sym} must map to V"
        assert is_heartbeat(sym) is True

    # F Class
    assert map_symbol_to_aami("F") == "F"
    assert is_heartbeat("F") is True

    # Q Class
    for sym in ["/", "f", "Q"]:
        assert map_symbol_to_aami(sym) == "Q", f"Symbol {sym} must map to Q"
        assert is_heartbeat(sym) is True


def test_aami_non_beat_symbols():
    """Verify that auxiliary markers (rhythm changes, quality) are classified as non-beats."""
    for sym in ["[", "!", "]", "x", "(", ")", "p", "t", "u", "~", "+"]:
        assert map_symbol_to_aami(sym) is None, f"Symbol {sym} should not map to an AAMI beat class"
        assert is_heartbeat(sym) is False


def test_summarize_aami_distribution():
    """Verify distribution counter aggregation."""
    symbols = ["N", "N", "L", "V", "A", "F", "/", "+", "~"]
    counts, non_beats = summarize_aami_distribution(symbols)

    assert counts["N"] == 3  # N, N, L
    assert counts["S"] == 1  # A
    assert counts["V"] == 1  # V
    assert counts["F"] == 1  # F
    assert counts["Q"] == 1  # /
    assert non_beats == 2    # +, ~
