#!/usr/bin/env python3
"""CLI script to run the ECG preprocessing, beat segmentation, and dataset construction pipeline."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ecg_arrhythmia.data.loader import MITBIHLoader
from ecg_arrhythmia.data.splits import load_manifest, verify_no_overlap
from ecg_arrhythmia.preprocessing.filtering import (
    filter_ecg,
    select_ecg_lead,
    validate_sampling_rate,
)
from ecg_arrhythmia.preprocessing.normalization import normalize_beats_zscore
from ecg_arrhythmia.preprocessing.segmentation import segment_beats_from_record
from ecg_arrhythmia.utils.config import load_config, resolve_path
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger("preprocess")


def process_partition(
    record_ids: List[str],
    loader: MITBIHLoader,
    prep_cfg: Dict[str, Any],
    split_name: str,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Process a single partition (train, val, or test) independently."""
    logger.info(f"--- Processing Partition: '{split_name}' ({len(record_ids)} records) ---")

    lead_cfg = prep_cfg.get("lead", {})
    primary_lead = lead_cfg.get("primary", "MLII")
    fallback_leads = lead_cfg.get("fallbacks", ["V5", "V1", "V2", "V4"])

    filter_cfg = prep_cfg.get("filter", {})
    low_hz = filter_cfg.get("low_hz", 0.5)
    high_hz = filter_cfg.get("high_hz", 40.0)
    order = filter_cfg.get("order", 4)
    zero_phase = filter_cfg.get("zero_phase", True)

    seg_cfg = prep_cfg.get("segmentation", {})
    pre_samples = seg_cfg.get("pre_samples", 72)
    post_samples = seg_cfg.get("post_samples", 144)

    partition_waveforms: List[np.ndarray] = []
    partition_metadata: List[Dict[str, Any]] = []

    total_annotations = 0
    total_unmapped = 0
    total_boundary = 0
    class_counts = {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}

    beat_global_id = 0

    for rid in tqdm(record_ids, desc=f"Preprocessing {split_name}"):
        # Load raw record and annotations via WFDB loader
        rec = loader.load_record(rid)
        ann = loader.load_annotations(rid)

        # Validate sampling rate (360 Hz)
        fs = validate_sampling_rate(rec, expected_fs=360)

        # Select primary or fallback lead signal
        raw_signal, selected_lead, _ = select_ecg_lead(
            rec, primary_lead=primary_lead, fallback_leads=fallback_leads
        )

        # Zero-phase Butterworth bandpass filter
        filtered_signal = filter_ecg(
            raw_signal,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            order=order,
            zero_phase=zero_phase,
        )

        # Annotation-centered beat segmentation
        waveforms, meta_records, stats = segment_beats_from_record(
            signal=filtered_signal,
            annotation_samples=ann.sample,
            annotation_symbols=ann.symbol,
            record_id=rid,
            lead_name=selected_lead,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
        )

        total_annotations += stats.total_annotations
        total_unmapped += stats.unmapped_excluded
        total_boundary += stats.boundary_excluded
        for c, count in stats.class_counts.items():
            class_counts[c] += count

        # Per-beat z-score normalization
        if waveforms.shape[0] > 0:
            norm_waveforms = normalize_beats_zscore(waveforms)
            partition_waveforms.append(norm_waveforms)

            for m in meta_records:
                m["beat_id"] = f"{split_name}_{beat_global_id:06d}"
                m["partition"] = split_name
                beat_global_id += 1
                partition_metadata.append(m)

    if partition_waveforms:
        all_waveforms = np.vstack(partition_waveforms).astype(np.float32)
    else:
        total_window = pre_samples + post_samples
        all_waveforms = np.empty((0, total_window), dtype=np.float32)

    df_meta = pd.DataFrame(partition_metadata)

    summary_stats = {
        "partition": split_name,
        "records_count": len(record_ids),
        "total_annotations": total_annotations,
        "extracted_beats": len(df_meta),
        "unmapped_excluded": total_unmapped,
        "boundary_excluded": total_boundary,
        "count_N": class_counts["N"],
        "count_S": class_counts["S"],
        "count_V": class_counts["V"],
        "count_F": class_counts["F"],
        "count_Q": class_counts["Q"],
    }

    return all_waveforms, df_meta, summary_stats


def generate_visual_sanity_check(
    processed_dir: Path, out_figure_path: Path
) -> None:
    """Generate representative preprocessed beat waveform plots for each AAMI class."""
    classes = ["N", "S", "V", "F", "Q"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)

    for idx, cls in enumerate(classes):
        ax = axes[idx]
        found = False

        # Search partitions for example beats
        for split in ["train", "val", "test"]:
            meta_path = processed_dir / split / "metadata.csv"
            beats_path = processed_dir / split / "beats.npy"

            if meta_path.is_file() and beats_path.is_file():
                df = pd.read_csv(meta_path)
                beats = np.load(beats_path)
                cls_df = df[df["aami_class"] == cls]

                if not cls_df.empty:
                    # Plot up to 3 example beats overlaid
                    sample_indices = cls_df.index[:3]
                    for s_idx in sample_indices:
                        rec_id = cls_df.loc[s_idx, "record_id"]
                        wf = beats[s_idx]
                        ax.plot(wf, alpha=0.75, label=f"Rec {rec_id}")

                    ax.set_title(f"Class '{cls}' Beats")
                    ax.set_xlabel("Sample Index (216 samples)")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    found = True
                    break

        if not found:
            ax.set_title(f"Class '{cls}' (No samples)")
            ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Normalized Amplitude (z-score)")
    fig.suptitle("Preprocessed & Normalized Beat Waveform Sanity Check (AAMI Classes)", fontsize=14)
    plt.tight_layout()

    out_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved visual sanity check waveform plot to: {out_figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ECG signal filtering, beat segmentation, and dataset construction."
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Path to data configuration file.",
    )
    parser.add_argument(
        "--prep-config",
        default="configs/preprocessing.yaml",
        help="Path to preprocessing configuration file.",
    )
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    prep_cfg = load_config(args.prep_config)

    raw_dir = resolve_path(data_cfg.get("dataset", {}).get("raw_dir", "data/raw/mitdb"))
    processed_dir = resolve_path(data_cfg.get("dataset", {}).get("processed_dir", "data/processed"))
    manifest_dir = resolve_path(data_cfg.get("dataset", {}).get("manifest_dir", "data/manifests"))

    # Load manifests
    train_manifest_file = manifest_dir / "train_records.txt"
    val_manifest_file = manifest_dir / "val_records.txt"
    test_manifest_file = manifest_dir / "test_records.txt"

    train_records = load_manifest(train_manifest_file)
    val_records = load_manifest(val_manifest_file)
    test_records = load_manifest(test_manifest_file)

    # 1. Leakage Verification
    logger.info("Performing patient-independent split leakage verification...")
    verify_no_overlap(train_records, val_records, test_records)
    logger.info("Leakage verification passed: Train, Val, and Test sets have ZERO record overlap.")

    # 2. Download missing records if needed
    loader = MITBIHLoader(raw_data_dir=raw_dir)
    all_target_records = train_records + val_records + test_records
    loader.download_all_records(all_target_records)

    # 3. Process each partition independently
    summary_rows = []
    partitions = [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]

    for split_name, rec_list in partitions:
        waveforms, df_meta, stats = process_partition(
            record_ids=rec_list,
            loader=loader,
            prep_cfg=prep_cfg,
            split_name=split_name,
        )

        out_split_dir = processed_dir / split_name
        out_split_dir.mkdir(parents=True, exist_ok=True)

        beats_file = out_split_dir / "beats.npy"
        meta_file = out_split_dir / "metadata.csv"

        np.save(beats_file, waveforms)
        df_meta.to_csv(meta_file, index=False)

        logger.info(
            f"Saved '{split_name}' dataset: {waveforms.shape[0]} beats -> "
            f"{beats_file} ({beats_file.stat().st_size / (1024*1024):.2f} MB)"
        )
        summary_rows.append(stats)

    # 4. Save preprocessing summary table
    df_summary = pd.DataFrame(summary_rows)
    tables_dir = resolve_path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = tables_dir / "preprocessing_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved preprocessing summary to: {summary_csv}")

    # 5. Generate visual sanity check plot
    figures_dir = resolve_path("reports/figures")
    sanity_plot = figures_dir / "preprocessed_beats_sample.png"
    generate_visual_sanity_check(processed_dir, sanity_plot)

    # 6. Display Final Summary Report
    print("\n" + "=" * 80)
    print("                MILESTONE 2 — PREPROCESSING & DATASET REPORT                  ")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
