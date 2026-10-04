"""Annotation-centered beat segmentation for ECG dataset construction."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from ecg_arrhythmia.data.annotations import map_symbol_to_aami
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentationStats:
    """Statistics tracking extracted beats and exclusion categories for a record."""

    record_id: str
    total_annotations: int
    extracted_beats: int
    unmapped_excluded: int
    boundary_excluded: int
    class_counts: Dict[str, int]


def segment_beats_from_record(
    signal: np.ndarray,
    annotation_samples: np.ndarray,
    annotation_symbols: List[str],
    record_id: str,
    lead_name: str,
    fs: int = 360,
    pre_samples: int = 72,
    post_samples: int = 144,
) -> Tuple[np.ndarray, List[Dict[str, Any]], SegmentationStats]:
    """Extract fixed-width beat waveforms centered on expert annotation locations.

    Scientific Disclaimer:
    Annotation-centered beat segmentation uses expert ground-truth R-peak locations
    for supervised benchmark dataset construction. Future inference deployment will integrate
    an automated R-peak detector (e.g. Pan-Tompkins).

    Args:
        signal: 1D filtered ECG signal array.
        annotation_samples: 1D array of sample indices for annotated beats.
        annotation_symbols: List of string annotation symbols.
        record_id: String record identifier.
        lead_name: Name of selected ECG lead (e.g., 'MLII').
        fs: Sampling frequency in Hz.
        pre_samples: Number of samples prior to annotation peak (default: 72 = 200 ms @ 360 Hz).
        post_samples: Number of samples after annotation peak (default: 144 = 400 ms @ 360 Hz).

    Returns:
        Tuple of (waveforms_matrix (N, total_samples), metadata_list, SegmentationStats)
    """
    total_window_samples = pre_samples + post_samples
    signal_length = len(signal)

    extracted_waveforms: List[np.ndarray] = []
    metadata_records: List[Dict[str, Any]] = []

    unmapped_count = 0
    boundary_count = 0
    class_counts: Dict[str, int] = {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}

    total_ann = len(annotation_samples)

    for idx in range(total_ann):
        r_peak = int(annotation_samples[idx])
        symbol = str(annotation_symbols[idx])

        # 1. Map raw WFDB symbol to AAMI class
        aami_class = map_symbol_to_aami(symbol)
        if aami_class is None:
            unmapped_count += 1
            continue

        # 2. Check boundary condition
        start_sample = r_peak - pre_samples
        end_sample = r_peak + post_samples

        if start_sample < 0 or end_sample > signal_length:
            boundary_count += 1
            logger.debug(
                f"Record {record_id}: Beat at sample {r_peak} excluded due to boundary constraint "
                f"[{start_sample}, {end_sample}] vs signal len {signal_length}"
            )
            continue

        # 3. Extract beat segment
        beat = signal[start_sample:end_sample].astype(np.float32)
        if len(beat) != total_window_samples:
            boundary_count += 1
            continue

        extracted_waveforms.append(beat)
        class_counts[aami_class] += 1

        metadata_records.append(
            {
                "record_id": record_id,
                "sample_index": r_peak,
                "original_symbol": symbol,
                "aami_class": aami_class,
                "lead": lead_name,
                "fs": fs,
                "pre_samples": pre_samples,
                "post_samples": post_samples,
            }
        )

    if extracted_waveforms:
        waveforms_matrix = np.vstack(extracted_waveforms)
    else:
        waveforms_matrix = np.empty((0, total_window_samples), dtype=np.float32)

    stats = SegmentationStats(
        record_id=record_id,
        total_annotations=total_ann,
        extracted_beats=len(extracted_waveforms),
        unmapped_excluded=unmapped_count,
        boundary_excluded=boundary_count,
        class_counts=class_counts,
    )

    return waveforms_matrix, metadata_records, stats
