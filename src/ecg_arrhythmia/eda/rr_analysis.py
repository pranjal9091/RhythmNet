"""Sequential RR interval calculation, rhythm context analysis, and RR distribution plots."""

from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecg_arrhythmia.data.annotations import AAMI_CLASSES
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def compute_rr_intervals(df_meta: pd.DataFrame, fs: int = 360) -> pd.DataFrame:
    """Calculate sequential RR intervals between consecutive annotated beats per record.

    Scientific Safeguard:
    RR intervals are calculated STRICTLY within each individual patient record.
    The calculation is reset at record boundaries so that no RR interval is ever computed
    between two different records. First beat of each record receives NaN.

    Args:
        df_meta: Metadata DataFrame containing 'record_id', 'sample_index', 'aami_class'.
        fs: Sampling frequency in Hz (default: 360).

    Returns:
        DataFrame copy with appended 'rr_sec', 'rr_prev', 'rr_next' columns.
    """
    df = df_meta.copy()

    # Preserve original index for alignment
    df["orig_idx"] = df.index
    df = df.sort_values(by=["record_id", "sample_index"]).reset_index(drop=True)

    rr_sec = np.full(len(df), np.nan, dtype=np.float64)
    rr_prev = np.full(len(df), np.nan, dtype=np.float64)
    rr_next = np.full(len(df), np.nan, dtype=np.float64)

    record_ids = df["record_id"].values
    sample_indices = df["sample_index"].values

    for i in range(1, len(df)):
        if record_ids[i] == record_ids[i - 1]:
            diff_samples = sample_indices[i] - sample_indices[i - 1]
            if diff_samples > 0:
                dt_sec = diff_samples / float(fs)
                rr_sec[i] = dt_sec
                rr_prev[i] = dt_sec

    for i in range(len(df) - 1):
        if record_ids[i] == record_ids[i + 1]:
            diff_samples = sample_indices[i + 1] - sample_indices[i]
            if diff_samples > 0:
                rr_next[i] = diff_samples / float(fs)

    df["rr_sec"] = rr_sec
    df["rr_prev"] = rr_prev
    df["rr_next"] = rr_next

    # Restore original ordering
    df = df.sort_values(by="orig_idx").drop(columns=["orig_idx"]).reset_index(drop=True)
    return df


def compute_rr_statistics(
    dfs_with_rr: Dict[str, pd.DataFrame],
    percentiles: List[int] = [1, 5, 25, 50, 75, 95, 99],
) -> pd.DataFrame:
    """Calculate aggregate RR interval statistics per class across partitions."""
    stat_rows = []

    for split_name, df in dfs_with_rr.items():
        if "rr_sec" not in df.columns:
            df = compute_rr_intervals(df)

        for cls in AAMI_CLASSES:
            cls_df = df[df["aami_class"] == cls]
            rr_vals = cls_df["rr_sec"].dropna().values

            count = len(rr_vals)
            if count == 0:
                continue

            pct_dict = {}
            for p in percentiles:
                pct_dict[f"p{p}"] = float(np.percentile(rr_vals, p))

            row = {
                "partition": split_name,
                "aami_class": cls,
                "valid_rr_count": count,
                "mean_sec": float(np.mean(rr_vals)),
                "std_sec": float(np.std(rr_vals)) if count > 1 else 0.0,
                "median_sec": float(np.median(rr_vals)),
                "min_sec": float(np.min(rr_vals)),
                "max_sec": float(np.max(rr_vals)),
                **pct_dict,
            }
            stat_rows.append(row)

    return pd.DataFrame(stat_rows)


def plot_rr_distribution_by_class(
    dfs_with_rr: Dict[str, pd.DataFrame], out_path: Any, dpi: int = 200
) -> None:
    """Generate boxplot of RR interval distributions across AAMI classes."""
    combined_list = []
    for split_name, df in dfs_with_rr.items():
        if "rr_sec" not in df.columns:
            df = compute_rr_intervals(df)
        temp = df[["aami_class", "rr_sec"]].dropna().copy()
        temp["partition"] = split_name
        combined_list.append(temp)

    if not combined_list:
        return

    full_df = pd.concat(combined_list, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=full_df,
        x="aami_class",
        y="rr_sec",
        hue="partition",
        palette="Set2",
        showfliers=False,  # Exclude extreme outliers for clean visualization
        ax=ax,
    )

    ax.set_title("Exploratory RR Interval Distribution by AAMI Class & Partition", fontsize=13)
    ax.set_xlabel("AAMI EC57 Beat Class")
    ax.set_ylabel("Sequential RR Interval (Seconds)")
    ax.set_ylim(0.0, 2.5)  # Focus on physiological range 0-2.5s
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Partition")

    plt.tight_layout()

    out_path = str(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved RR distribution by class plot to: {out_path}")


def plot_rr_distribution_by_partition(
    dfs_with_rr: Dict[str, pd.DataFrame], out_path: Any, dpi: int = 200
) -> None:
    """Generate KDE distribution comparison of RR intervals across partitions."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

    for split_name, df in dfs_with_rr.items():
        if "rr_sec" not in df.columns:
            df = compute_rr_intervals(df)
        rr_vals = df["rr_sec"].dropna().values
        # Filter physiological range 0.2 to 2.5s for clean KDE
        valid_rr = rr_vals[(rr_vals >= 0.2) & (rr_vals <= 2.5)]

        sns.kdeplot(
            valid_rr,
            label=f"{split_name.capitalize()} (N={len(valid_rr)})",
            color=colors.get(split_name, None),
            ax=ax,
            linewidth=2.0,
        )

    ax.set_title("RR Interval Kernel Density Estimate (KDE) Across Partitions", fontsize=13)
    ax.set_xlabel("Sequential RR Interval (Seconds)")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Partition")

    plt.tight_layout()

    out_path = str(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved RR distribution by partition plot to: {out_path}")
