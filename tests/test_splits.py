"""Tests for patient-independent record splitting and leakage prevention."""

from pathlib import Path
import pytest

from ecg_arrhythmia.data.splits import (
    CANONICAL_DS1,
    CANONICAL_DS2,
    PACED_RECORDS,
    create_de_chazal_manifests,
    load_manifest,
    verify_no_overlap,
)


def test_canonical_splits_disjoint():
    """Verify that canonical DS1, DS2, and paced records have zero mutual overlap."""
    ds1_set = set(CANONICAL_DS1)
    ds2_set = set(CANONICAL_DS2)
    paced_set = set(PACED_RECORDS)

    # Size assertions
    assert len(ds1_set) == 22, f"DS1 must contain 22 records, found {len(ds1_set)}"
    assert len(ds2_set) == 22, f"DS2 must contain 22 records, found {len(ds2_set)}"
    assert len(paced_set) == 4, f"Paced records must be 4, found {len(paced_set)}"

    # Mutual exclusion
    assert ds1_set.isdisjoint(ds2_set), "DS1 and DS2 must be mutually disjoint!"
    assert ds1_set.isdisjoint(paced_set), "DS1 and paced records must be mutually disjoint!"
    assert ds2_set.isdisjoint(paced_set), "DS2 and paced records must be mutually disjoint!"


def test_verify_no_overlap_passes_on_disjoint():
    """verify_no_overlap should pass silently when sets are completely disjoint."""
    train = ["101", "106", "108"]
    val = ["109", "112"]
    test = ["100", "103"]

    # Should not raise
    verify_no_overlap(train, val, test)
    verify_no_overlap(train, None, test)


def test_verify_no_overlap_detects_train_val_leakage():
    """verify_no_overlap must raise ValueError when a record appears in both train and val."""
    train = ["101", "106", "108"]
    val = ["108", "112"]  # 108 is in both!
    test = ["100", "103"]

    with pytest.raises(ValueError, match="Records present in both Train and Validation"):
        verify_no_overlap(train, val, test)


def test_verify_no_overlap_detects_train_test_leakage():
    """verify_no_overlap must raise ValueError when a record appears in both train and test."""
    train = ["101", "106", "100"]
    val = ["109", "112"]
    test = ["100", "103"]  # 100 is in both!

    with pytest.raises(ValueError, match="Records present in both Train and Test"):
        verify_no_overlap(train, val, test)


def test_verify_no_overlap_detects_val_test_leakage():
    """verify_no_overlap must raise ValueError when a record appears in both val and test."""
    train = ["101", "106"]
    val = ["109", "200"]
    test = ["200", "103"]  # 200 is in both!

    with pytest.raises(ValueError, match="Records present in both Validation and Test"):
        verify_no_overlap(train, val, test)


def test_create_de_chazal_manifests(tmp_path: Path):
    """Test generating and saving manifest files."""
    manifest_dir = tmp_path / "manifests"
    splits = create_de_chazal_manifests(manifest_dir=manifest_dir, val_record_count=5, seed=42)

    assert len(splits["train"]) == 17
    assert len(splits["val"]) == 5
    assert len(splits["test"]) == 22

    # Manifest files exist on disk
    train_manifest = manifest_dir / "train_records.txt"
    val_manifest = manifest_dir / "val_records.txt"
    test_manifest = manifest_dir / "test_records.txt"

    assert train_manifest.is_file()
    assert val_manifest.is_file()
    assert test_manifest.is_file()

    # Loaded manifests match generated
    loaded_train = load_manifest(train_manifest)
    assert loaded_train == splits["train"]
