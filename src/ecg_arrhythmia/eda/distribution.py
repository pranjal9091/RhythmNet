"""Class distribution analysis and visualization for ECG arrhythmia partitions."""

from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_arrhythmia.data.annotations import AAMI_CLASSES
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_class_distributions(dfs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate detailed class distribution metrics across dataset partitions.

    Args:
        dfs_dict: Dict mapping partition names ('train', 'val', 'test') to metadata DataFrames.

    Returns:
        DataFrame containing summary statistics per partition.
    """
    rows = []

    for split_name, df in dfs_dict.items():
        total_beats = len(df)
        total_records = df["record_id"].nunique() if "record_id" in df.columns else 0

        counts = df["aami_class"].value_counts().to_dict()
        class_counts = {cls: counts.get(cls, 0) for cls in AAMI_CLASSES}

        # Calculate percentages
        class_pcts = {
            f"{cls}_pct": (class_counts[cls] / total_beats * 100.0) if total_beats > 0 else 0.0
            for cls in AAMI_CLASSES
        }

        # Class presence & ratios
        active_classes = sum(1 for cnt in class_counts.values() if cnt > 0)
        max_count = max(class_counts.values()) if total_beats > 0 else 0
        min_nonzero_count = min(cnt for cnt in class_counts.values() if cnt > 0) if active_classes > 0 else 1
        imbalance_ratio = (max_count / min_nonzero_count) if min_nonzero_count > 0 else np.nan

        row = {
            "partition": split_name,
            "total_records": total_records,
            "total_beats": total_beats,
            "count_N": class_counts["N"],
            "pct_N": class_pcts["N_pct"],
            "count_S": class_counts["S"],
            "pct_S": class_pcts["S_pct"],
            "count_V": class_counts["V"],
            "pct_V": class_pcts["V_pct"],
            "count_F": class_counts["F"],
            "pct_F": class_pcts["F_pct"],
            "count_Q": class_counts["Q"],
            "pct_Q": class_pcts["Q_pct"],
            "active_classes": active_classes,
            "imbalance_ratio": imbalance_ratio,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_class_distribution(
    summary_df: pd.DataFrame, out_path: Any, dpi: int = 200
) -> None:
    """Generate publication-ready class distribution visualization with log-scale counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    partitions = summary_df["partition"].tolist()
    x = np.arange(len(AAMI_CLASSES))
    width = 0.25

    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

    # Left plot: Absolute counts (log scale)
    for i, p in enumerate(partitions):
        row = summary_df[summary_df["partition"] == p].iloc[0]
        counts = [row[f"count_{cls}"] for cls in AAMI_CLASSES]
        offset = (i - 1) * width
        rects = ax1.bar(x + offset, counts, width, label=p.capitalize(), color=colors.get(p, None))
        
        # Annotate exact count values above bars
        for rect, c in zip(rects, counts):
            height = rect.get_height()
            if height > 0:
                ax1.annotate(
                    f"{c}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )

    ax1.set_yscale("log")
    ax1.set_ylabel("Heartbeat Count (Log Scale)")
    ax1.set_title("Absolute Class Counts by Partition (Log Scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Class {cls}" for cls in AAMI_CLASSES])
    ax1.legend(title="Partition")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # Right plot: Percentages (linear scale)
    for i, p in enumerate(partitions):
        row = summary_df[summary_df["partition"] == p].iloc[0]
        pcts = [row[f"pct_{cls}"] for cls in AAMI_CLASSES]
        offset = (i - 1) * width
        ax2.bar(x + offset, pcts, width, label=p.capitalize(), color=colors.get(p, None))

    ax2.set_ylabel("Percentage of Partition (%)")
    ax2.set_title("Relative Class Proportions (%) by Partition")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Class {cls}" for cls in AAMI_CLASSES])
    ax2.legend(title="Partition")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("AAMI EC57 Class Distribution & Imbalance Audit", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = str(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved class distribution figure to: {out_path}")
