#!/usr/bin/env python3
"""CLI script to download MIT-BIH Arrhythmia Database records from PhysioNet via WFDB."""

import argparse
import sys
from pathlib import Path

from ecg_arrhythmia.data.loader import MITBIHLoader
from ecg_arrhythmia.utils.config import load_config
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger("download_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MIT-BIH Arrhythmia Database records from PhysioNet."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all 48 records in the MIT-BIH Arrhythmia Database.",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        help="Specific record IDs to download (e.g., --records 100 101 106).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download representative verification records (100, 101, 106, 200).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-download and overwrite existing records.",
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data configuration file (default: configs/data.yaml).",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg.get("dataset", {}).get("raw_dir", "data/raw/mitdb")
    all_records = cfg.get("records", {}).get("all", [])

    loader = MITBIHLoader(raw_data_dir=raw_dir)

    if args.all:
        target_records = all_records
    elif args.records:
        target_records = args.records
    elif args.sample:
        target_records = ["100", "101", "106", "200"]
    else:
        # Default behavior: download sample records for quick inspection
        logger.info("No download mode specified. Defaulting to sample records: ['100', '101', '106', '200'].")
        logger.info("Pass --all to download all 48 records or --records <IDs...> for specific records.")
        target_records = ["100", "101", "106", "200"]

    logger.info(f"Targeting {len(target_records)} records for download into '{raw_dir}'...")

    successful = []
    failed = []

    for rid in target_records:
        rid_str = str(rid).strip()
        try:
            loader.download_record(rid_str, overwrite=args.overwrite)
            successful.append(rid_str)
        except Exception as exc:
            logger.error(f"Error downloading record {rid_str}: {exc}")
            failed.append(rid_str)

    logger.info("--------------------------------------------------")
    logger.info(f"Download completed. Available: {len(successful)} | Failed: {len(failed)}")
    if failed:
        logger.warning(f"Failed record IDs: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
