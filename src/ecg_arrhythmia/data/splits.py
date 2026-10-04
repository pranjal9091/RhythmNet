"""Patient and record-level split abstractions and leakage-prevention validators.

Strict inter-patient evaluation requires that beats from any given patient appear
in EXCLUSIVELY ONE partition (Train, Validation, or Test). Under no circumstance
may a record/patient ID cross partition boundaries.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Canonical de Chazal et al. (2004) Inter-Patient Records Split
# Total 44 non-paced records (22 in DS1, 22 in DS2)
PACED_RECORDS: Tuple[str, ...] = ("102", "104", "107", "217")

CANONICAL_DS1: Tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
)

CANONICAL_DS2: Tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
)


def verify_no_overlap(
    train_records: Iterable[str],
    val_records: Optional[Iterable[str]] = None,
    test_records: Optional[Iterable[str]] = None,
) -> None:
    """Verify that train, validation, and test record sets are strictly disjoint.

    Raises:
        ValueError: If any record appears in more than one partition.
    """
    train_set = set(str(r) for r in train_records)
    val_set = set(str(r) for r in val_records) if val_records is not None else set()
    test_set = set(str(r) for r in test_records) if test_records is not None else set()

    train_val = train_set & val_set
    if train_val:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: Records present in both Train and Validation sets: {sorted(train_val)}"
        )

    train_test = train_set & test_set
    if train_test:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: Records present in both Train and Test sets: {sorted(train_test)}"
        )

    val_test = val_set & test_set
    if val_test:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: Records present in both Validation and Test sets: {sorted(val_test)}"
        )


def create_de_chazal_manifests(
    manifest_dir: Path,
    val_record_count: int = 5,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Generate patient-independent train/val/test manifests following de Chazal DS1/DS2.

    DS2 is reserved as the final unseen test set (22 records).
    DS1 (22 records) is partitioned into train and validation sets at the record level.

    Args:
        manifest_dir: Directory where manifest text files will be written.
        val_record_count: Number of DS1 records to allocate to validation.
        seed: Random seed for deterministic train/val split within DS1.

    Returns:
        Dict mapping split names ('train', 'val', 'test') to lists of record IDs.
    """
    import random

    manifest_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    ds1_list = list(CANONICAL_DS1)
    rng.shuffle(ds1_list)

    val_records = sorted(ds1_list[:val_record_count])
    train_records = sorted(ds1_list[val_record_count:])
    test_records = sorted(CANONICAL_DS2)

    # Enforce zero leakage assertion
    verify_no_overlap(train_records, val_records, test_records)

    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    for split_name, rec_list in splits.items():
        manifest_file = manifest_dir / f"{split_name}_records.txt"
        with open(manifest_file, "w", encoding="utf-8") as f:
            for r in rec_list:
                f.write(f"{r}\n")

    return splits


def load_manifest(manifest_path: Path) -> List[str]:
    """Load a record manifest file containing one record ID per line."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return sorted(records)
