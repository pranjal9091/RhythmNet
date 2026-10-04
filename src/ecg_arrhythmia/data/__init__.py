"""Data access, annotation mapping, and patient-level splitting module."""

from ecg_arrhythmia.data.annotations import (
    AAMI_CLASSES,
    AAMI_MAPPING,
    NON_BEAT_SYMBOLS,
    inspect_annotation_symbols,
    is_heartbeat,
    map_symbol_to_aami,
    summarize_aami_distribution,
)
from ecg_arrhythmia.data.loader import MITBIHLoader, RecordMetadata
from ecg_arrhythmia.data.splits import (
    CANONICAL_DS1,
    CANONICAL_DS2,
    PACED_RECORDS,
    create_de_chazal_manifests,
    load_manifest,
    verify_no_overlap,
)

__all__ = [
    "MITBIHLoader",
    "RecordMetadata",
    "AAMI_CLASSES",
    "AAMI_MAPPING",
    "NON_BEAT_SYMBOLS",
    "is_heartbeat",
    "map_symbol_to_aami",
    "inspect_annotation_symbols",
    "summarize_aami_distribution",
    "CANONICAL_DS1",
    "CANONICAL_DS2",
    "PACED_RECORDS",
    "create_de_chazal_manifests",
    "load_manifest",
    "verify_no_overlap",
]
