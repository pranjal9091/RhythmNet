"""Patient and record-level contribution auditing and minority class coverage analysis."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecg_arrhythmia.data.annotations import AAMI_CLASSES
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_patient_contributions(dfs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate per-record beat counts, class percentages, and dominant classes."""
    rows = []

    for split_name, df in dfs_dict.items():
        partition_total = len(df)
        records = sorted(df["record_id"].unique())

        for rid in records:
            rec_df = df[df["record_id"] == rid]
            rec_total = len(rec_df)

            counts = rec_df["aami_class"].value_counts().to_dict()
            c_counts = {cls: counts.get(cls, 0) for cls in AAMI_CLASSES}

            c_pcts = {
                f"pct_{cls}": (c_counts[cls] / rec_total * 100.0) if rec_total > 0 else 0.0
                for cls in AAMI_CLASSES
            }

            dominant_class = max(c_counts, key=c_counts.get)
            active_classes = sum(1 for cnt in c_counts.values() if cnt > 0)
            pct_of_partition = (rec_total / partition_total * 100.0) if partition_total > 0 else 0.0

            row = {
                "partition": split_name,
                "record_id": str(rid),
                "total_beats": rec_total,
                "pct_of_partition": pct_of_partition,
                "dominant_class": dominant_class,
                "active_classes": active_classes,
                "count_N": c_counts["N"],
                "pct_N": c_pcts["pct_N"],
                "count_S": c_counts["S"],
                "pct_S": c_pcts["pct_S"],
                "count_V": c_counts["V"],
                "pct_V": c_pcts["pct_V"],
                "count_F": c_counts["F"],
                "pct_F": c_pcts["pct_F"],
                "count_Q": c_counts["Q"],
                "pct_Q": c_pcts["pct_Q"],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def analyze_minority_patient_coverage(dfs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate patient coverage metrics for each AAMI class across partitions."""
    rows = []

    for split_name, df in dfs_dict.items():
        total_records_in_split = df["record_id"].nunique()

        for cls in AAMI_CLASSES:
            cls_df = df[df["aami_class"] == cls]
            total_beats = len(cls_df)

            if total_beats == 0:
                rows.append({
                    "partition": split_name,
                    "aami_class": cls,
                    "contributing_records": 0,
                    "total_partition_records": total_records_in_split,
                    "record_coverage_pct": 0.0,
                    "total_beats": 0,
                    "median_beats_per_record": 0.0,
                    "min_beats_per_record": 0,
                    "max_beats_per_record": 0,
                })
                continue

            record_counts = cls_df["record_id"].value_counts()
            n_contrib = len(record_counts)
            coverage_pct = (n_contrib / total_records_in_split * 100.0) if total_records_in_split > 0 else 0.0

            rows.append({
                "partition": split_name,
                "aami_class": cls,
                "contributing_records": n_contrib,
                "total_partition_records": total_records_in_split,
                "record_coverage_pct": coverage_pct,
                "total_beats": total_beats,
                "median_beats_per_record": float(record_counts.median()),
                "min_beats_per_record": int(record_counts.min()),
                "max_beats_per_record": int(record_counts.max()),
            })

    return pd.DataFrame(rows)


def plot_record_class_heatmap(
    record_df: pd.DataFrame, out_path: Any, dpi: int = 200
) -> None:
    """Generate record vs class percentage heatmap."""
    pivot_df = record_df.pivot(index="record_id", columns="partition", values="total_beats")
    
    # Prepare percentage matrix: rows = record_id, cols = N, S, V, F, Q
    pct_cols = [f"pct_{cls}" for cls in AAMI_CLASSES]
    heatmap_matrix = record_df.set_index("record_id")[pct_cols]
    heatmap_matrix.columns = list(AAMI_CLASSES)

    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Class Percentage within Record (%)"},
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title("Per-Record Class Composition (%) Heatmap", fontsize=14, pad=15)
    ax.set_xlabel("AAMI EC57 Class")
    ax.set_ylabel("Patient Record ID")
    plt.tight_layout()

    out_path = str(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved record class heatmap to: {out_path}")
