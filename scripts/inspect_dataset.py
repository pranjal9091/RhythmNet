#!/usr/bin/env python3
"""CLI script to inspect available MIT-BIH records, metadata, and annotation symbol distributions."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ecg_arrhythmia.data.annotations import (
    AAMI_CLASSES,
    inspect_annotation_symbols,
    map_symbol_to_aami,
    summarize_aami_distribution,
)
from ecg_arrhythmia.data.loader import MITBIHLoader
from ecg_arrhythmia.utils.config import load_config, resolve_path
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger("inspect_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect metadata, signals, and annotations of available MIT-BIH records."
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data configuration file (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--save-table",
        action="store_true",
        default=True,
        help="Save record inventory table to reports/tables/record_inventory.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg.get("dataset", {}).get("raw_dir", "data/raw/mitdb")

    loader = MITBIHLoader(raw_data_dir=raw_dir)
    records = loader.list_local_records()

    if not records:
        logger.warning(
            f"No complete MIT-BIH records (.hea, .dat, .atr) found in '{raw_dir}'.\n"
            f"Run 'python scripts/download_data.py --sample' or 'python scripts/download_data.py --all' first."
        )
        return

    logger.info("================================================================================")
    logger.info(f"                MIT-BIH ARRHYTHMIA DATABASE INSPECTION REPORT                  ")
    logger.info("================================================================================")
    logger.info(f"Local storage path : {raw_dir}")
    logger.info(f"Discovered records : {len(records)} records available locally\n")

    inventory_rows = []
    total_raw_symbols: Counter = Counter()
    total_aami_counts: Dict[str, int] = {c: 0 for c in AAMI_CLASSES}
    total_non_beats = 0

    print(
        f"{'Record':<8} {'Sampling':<10} {'Channels':<18} {'Duration':<12} "
        f"{'Samples':<10} {'Beats/Sym':<10}"
    )
    print("-" * 76)

    for rid in records:
        meta = loader.get_record_metadata(rid)
        ann = loader.load_annotations(rid)

        symbols = ann.symbol
        total_raw_symbols.update(symbols)
        aami_dist, non_beats = summarize_aami_distribution(symbols)
        for c, count in aami_dist.items():
            total_aami_counts[c] += count
        total_non_beats += non_beats

        channels_str = "/".join(meta.channels)
        duration_min = f"{meta.duration_sec / 60.0:.1f} min"

        print(
            f"{meta.record_id:<8} {meta.sampling_rate:<4} Hz    "
            f"{channels_str:<18} {duration_min:<12} {meta.n_samples:<10} {len(symbols):<10}"
        )

        inventory_rows.append(
            {
                "record_id": meta.record_id,
                "sampling_rate_hz": meta.sampling_rate,
                "n_channels": meta.n_channels,
                "channels": channels_str,
                "duration_sec": meta.duration_sec,
                "n_samples": meta.n_samples,
                "total_annotations": len(symbols),
                "comments": " | ".join(meta.comments),
            }
        )

    print("-" * 76)
    print("\nAnnotation Symbol Distribution (Raw WFDB Symbols):")
    for sym, count in sorted(total_raw_symbols.items(), key=lambda x: x[1], reverse=True):
        mapped = map_symbol_to_aami(sym)
        mapped_desc = f"-> AAMI '{mapped}'" if mapped else "-> Non-beat marker"
        print(f"  Symbol '{sym}': {count:>7} occurrences  ({mapped_desc})")

    print("\nAAMI EC57 Class Distribution Summary across Inspected Records:")
    total_beats = sum(total_aami_counts.values())
    for cls in AAMI_CLASSES:
        cnt = total_aami_counts[cls]
        pct = (cnt / total_beats * 100.0) if total_beats > 0 else 0.0
        print(f"  Class {cls:<2} : {cnt:>7} beats ({pct:>5.1f}%)")
    print(f"  Non-beat markers (rhythm/quality/noise) : {total_non_beats:>7}")
    print(f"  Total valid cardiac beats              : {total_beats:>7}")

    if args.save_table and inventory_rows:
        tables_dir = resolve_path("reports/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        out_csv = tables_dir / "record_inventory.csv"
        df = pd.DataFrame(inventory_rows)
        df.to_csv(out_csv, index=False)
        logger.info(f"\nSaved machine-readable record inventory to: {out_csv}")


if __name__ == "__main__":
    main()
