#!/usr/bin/env python3
"""CLI script to execute the complete Exploratory Data Analysis (EDA) and dataset audit suite."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from ecg_arrhythmia.data.splits import verify_no_overlap
from ecg_arrhythmia.eda.distribution import (
    analyze_class_distributions,
    plot_class_distribution,
)
from ecg_arrhythmia.eda.patient_audit import (
    analyze_minority_patient_coverage,
    analyze_patient_contributions,
    plot_record_class_heatmap,
)
from ecg_arrhythmia.eda.rr_analysis import (
    compute_rr_intervals,
    compute_rr_statistics,
    plot_rr_distribution_by_class,
    plot_rr_distribution_by_partition,
)
from ecg_arrhythmia.eda.statistics import (
    compare_partition_distributions,
    plot_partition_comparison,
    run_data_quality_audit,
)
from ecg_arrhythmia.eda.waveform_analysis import (
    compute_morphology_statistics,
    plot_waveform_overlays,
    plot_waveforms_by_class,
)
from ecg_arrhythmia.utils.config import load_config, resolve_path
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger("run_eda")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Exploratory Data Analysis (EDA) and Data Quality Audit."
    )
    parser.add_argument(
        "--config",
        default="configs/eda.yaml",
        help="Path to EDA configuration file (default: configs/eda.yaml).",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Path to data configuration file (default: configs/data.yaml).",
    )
    args = parser.parse_args()

    eda_cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)

    processed_dir = resolve_path(data_cfg.get("dataset", {}).get("processed_dir", "data/processed"))
    tables_dir = resolve_path("reports/tables")
    figures_dir = resolve_path("reports/figures")

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dpi = eda_cfg.get("plots", {}).get("dpi", 200)
    fs = eda_cfg.get("sampling_frequency", 360)

    # 1. Load preprocessed datasets and metadata
    logger.info(f"Loading preprocessed datasets from '{processed_dir}'...")

    dfs_dict: Dict[str, pd.DataFrame] = {}
    beats_dict: Dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        meta_file = processed_dir / split / "metadata.csv"
        beats_file = processed_dir / split / "beats.npy"

        if not meta_file.is_file() or not beats_file.is_file():
            logger.error(
                f"Processed files missing for partition '{split}' at {processed_dir}.\n"
                f"Run 'python scripts/preprocess.py' first!"
            )
            sys.exit(1)

        df = pd.read_csv(meta_file)
        beats = np.load(beats_file)

        dfs_dict[split] = df
        beats_dict[split] = beats
        logger.info(f"Loaded '{split}': {len(df)} metadata rows, {beats.shape} waveforms.")

    # 2. Verify zero patient record leakage
    train_recs = dfs_dict["train"]["record_id"].unique()
    val_recs = dfs_dict["val"]["record_id"].unique()
    test_recs = dfs_dict["test"]["record_id"].unique()
    verify_no_overlap(train_recs, val_recs, test_recs)
    logger.info("Patient leakage verification PASSED: zero record overlap across Train, Val, and Test.")

    # 3. Class Distribution Analysis
    logger.info("Running Class Distribution Analysis...")
    df_class_dist = analyze_class_distributions(dfs_dict)
    class_dist_csv = tables_dir / "class_distribution.csv"
    df_class_dist.to_csv(class_dist_csv, index=False)
    
    class_dist_fig = figures_dir / "class_distribution.png"
    plot_class_distribution(df_class_dist, class_dist_fig, dpi=dpi)

    # 4. Patient / Record Contribution Audit & Minority Coverage
    logger.info("Running Patient / Record Contribution Audit...")
    df_patient_contrib = analyze_patient_contributions(dfs_dict)
    patient_contrib_csv = tables_dir / "record_class_distribution.csv"
    df_patient_contrib.to_csv(patient_contrib_csv, index=False)

    heatmap_fig = figures_dir / "record_class_heatmap.png"
    plot_record_class_heatmap(df_patient_contrib, heatmap_fig, dpi=dpi)

    df_coverage = analyze_minority_patient_coverage(dfs_dict)
    coverage_csv = tables_dir / "class_patient_coverage.csv"
    df_coverage.to_csv(coverage_csv, index=False)

    # 5. Waveform Morphology Analysis & Statistics
    logger.info("Running Waveform Morphology Analysis & Overlay Plots...")
    n_examples = eda_cfg.get("plots", {}).get("waveform_examples_per_class", 5)
    n_overlay = eda_cfg.get("plots", {}).get("overlay_examples_per_class", 100)

    waveforms_fig = figures_dir / "waveforms_by_class.png"
    plot_waveforms_by_class(beats_dict, dfs_dict, waveforms_fig, n_examples=n_examples, dpi=dpi, fs=fs)

    overlays_fig = figures_dir / "class_waveform_overlays.png"
    plot_waveform_overlays(beats_dict, dfs_dict, overlays_fig, n_overlay=n_overlay, dpi=dpi, fs=fs)

    df_morph_stats = compute_morphology_statistics(beats_dict, dfs_dict)
    morph_csv = tables_dir / "morphology_statistics.csv"
    df_morph_stats.to_csv(morph_csv, index=False)

    # 6. Sequential RR Interval & Rhythm Context Analysis
    logger.info("Running Sequential RR Interval & Rhythm Context Analysis...")
    dfs_with_rr: Dict[str, pd.DataFrame] = {}
    for split_name, df in dfs_dict.items():
        dfs_with_rr[split_name] = compute_rr_intervals(df, fs=fs)

    percentiles = eda_cfg.get("statistics", {}).get("percentiles", [1, 5, 25, 50, 75, 95, 99])
    df_rr_stats = compute_rr_statistics(dfs_with_rr, percentiles=percentiles)
    rr_stats_csv = tables_dir / "rr_statistics.csv"
    df_rr_stats.to_csv(rr_stats_csv, index=False)

    rr_class_fig = figures_dir / "rr_distribution_by_class.png"
    plot_rr_distribution_by_class(dfs_with_rr, rr_class_fig, dpi=dpi)

    rr_part_fig = figures_dir / "rr_distribution_by_partition.png"
    plot_rr_distribution_by_partition(dfs_with_rr, rr_part_fig, dpi=dpi)

    # 7. Data Quality Audit (10 automated integrity checks)
    logger.info("Running 10-point Automated Data Quality Audit...")
    df_audit = run_data_quality_audit(beats_dict, dfs_with_rr)
    audit_csv = tables_dir / "data_quality_audit.csv"
    df_audit.to_csv(audit_csv, index=False)

    critical_failures = df_audit[(df_audit["severity"] == "CRITICAL") & (~df_audit["passed"])]
    if not critical_failures.empty:
        logger.error(f"CRITICAL DATA INTEGRITY FAILURES DETECTED:\n{critical_failures}")
        sys.exit(1)

    # 8. Train / Val / Test Distribution Comparison & JSD
    logger.info("Running Partition Distribution Comparison & Jensen-Shannon Divergence...")
    df_partition_comp = compare_partition_distributions(dfs_dict)
    comp_csv = tables_dir / "partition_comparison.csv"
    df_partition_comp.to_csv(comp_csv, index=False)

    comp_fig = figures_dir / "partition_class_comparison.png"
    plot_partition_comparison(df_partition_comp, comp_fig, dpi=dpi)

    # 9. Terminal Executive Summary Report
    print("\n" + "=" * 85)
    print("               MILESTONE 3 — EXPLORATORY DATA ANALYSIS & AUDIT REPORT          ")
    print("=" * 85)
    print("\n--- 1. Class Distributions Across Partitions ---")
    print(df_class_dist[["partition", "total_records", "total_beats", "count_N", "count_S", "count_V", "count_F", "count_Q", "imbalance_ratio"]].to_string(index=False))

    print("\n--- 2. Minority Class Patient Coverage ---")
    print(df_coverage[["partition", "aami_class", "contributing_records", "total_partition_records", "record_coverage_pct", "total_beats", "median_beats_per_record"]].to_string(index=False))

    print("\n--- 3. Automated Data Quality Audit Summary ---")
    failed_checks = df_audit[~df_audit["passed"]]
    if failed_checks.empty:
        print("  ALL 10 DATA QUALITY CHECKS PASSED SUCCESSFULLY (0 issues).")
    else:
        print(failed_checks[["check_name", "partition", "severity", "details"]].to_string(index=False))

    print("\n" + "=" * 85 + "\n")


if __name__ == "__main__":
    main()
