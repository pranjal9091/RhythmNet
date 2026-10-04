"""Automated data quality auditing, partition distribution comparison, and Jensen-Shannon divergence."""

from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon

from ecg_arrhythmia.data.annotations import AAMI_CLASSES
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def run_data_quality_audit(
    beats_dict: Dict[str, np.ndarray], dfs_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Perform 10 automated dataset integrity and quality checks across partitions.

    Returns:
        DataFrame containing audit results with check_name, partition, passed, issue_count, severity, and details.
    """
    audit_rows = []

    for split_name in ["train", "val", "test"]:
        if split_name not in dfs_dict or split_name not in beats_dict:
            continue

        df = dfs_dict[split_name]
        beats = beats_dict[split_name]

        # Check 1: NaN or Inf in waveforms
        nan_count = int(np.sum(np.isnan(beats)))
        inf_count = int(np.sum(np.isinf(beats)))
        nan_inf_issues = nan_count + inf_count
        audit_rows.append({
            "check_name": "nan_or_inf_in_waveforms",
            "partition": split_name,
            "passed": nan_inf_issues == 0,
            "issue_count": nan_inf_issues,
            "severity": "CRITICAL",
            "details": f"Found {nan_count} NaNs and {inf_count} Infs in {split_name} beats array.",
        })

        # Check 2: Invalid waveform shape
        expected_cols = 216
        is_shape_valid = (beats.ndim == 2 and beats.shape[1] == expected_cols)
        audit_rows.append({
            "check_name": "invalid_waveform_shape",
            "partition": split_name,
            "passed": is_shape_valid,
            "issue_count": 0 if is_shape_valid else 1,
            "severity": "CRITICAL",
            "details": f"Shape is {beats.shape}, expected (N, {expected_cols}).",
        })

        # Check 3: Metadata row count vs waveform count mismatch
        count_mismatch = abs(len(beats) - len(df))
        audit_rows.append({
            "check_name": "metadata_count_mismatch",
            "partition": split_name,
            "passed": count_mismatch == 0,
            "issue_count": count_mismatch,
            "severity": "CRITICAL",
            "details": f"Waveforms count ({len(beats)}) vs Metadata rows ({len(df)}).",
        })

        # Check 4: Duplicate beat_id
        dup_beat_ids = int(df["beat_id"].duplicated().sum()) if "beat_id" in df.columns else 0
        audit_rows.append({
            "check_name": "duplicate_beat_ids",
            "partition": split_name,
            "passed": dup_beat_ids == 0,
            "issue_count": dup_beat_ids,
            "severity": "HIGH",
            "details": f"Found {dup_beat_ids} duplicate beat_id values.",
        })

        # Check 5: Duplicate (record_id, sample_index) pairs
        dup_pairs = int(df.duplicated(subset=["record_id", "sample_index"]).sum())
        audit_rows.append({
            "check_name": "duplicate_record_sample_pairs",
            "partition": split_name,
            "passed": dup_pairs == 0,
            "issue_count": dup_pairs,
            "severity": "HIGH",
            "details": f"Found {dup_pairs} duplicate (record_id, sample_index) pairs.",
        })

        # Check 6: Invalid AAMI labels
        invalid_labels = int((~df["aami_class"].isin(AAMI_CLASSES)).sum())
        audit_rows.append({
            "check_name": "invalid_aami_labels",
            "partition": split_name,
            "passed": invalid_labels == 0,
            "issue_count": invalid_labels,
            "severity": "CRITICAL",
            "details": f"Found {invalid_labels} invalid AAMI class labels.",
        })

        # Check 7: Missing required metadata columns
        req_cols = {"beat_id", "record_id", "sample_index", "aami_class", "lead", "fs"}
        missing_cols = list(req_cols - set(df.columns))
        audit_rows.append({
            "check_name": "missing_required_columns",
            "partition": split_name,
            "passed": len(missing_cols) == 0,
            "issue_count": len(missing_cols),
            "severity": "HIGH",
            "details": f"Missing metadata columns: {missing_cols}",
        })

        # Check 8: Inconsistent sampling frequency
        bad_fs_count = int((df["fs"] != 360).sum()) if "fs" in df.columns else 0
        audit_rows.append({
            "check_name": "inconsistent_sampling_frequency",
            "partition": split_name,
            "passed": bad_fs_count == 0,
            "issue_count": bad_fs_count,
            "severity": "MEDIUM",
            "details": f"Found {bad_fs_count} rows with fs != 360 Hz.",
        })

        # Check 9: Suspiciously constant / zero-variance waveforms
        if len(beats) > 0:
            stds = np.std(beats, axis=1)
            zero_var_count = int(np.sum(stds < 1e-6))
        else:
            zero_var_count = 0
        audit_rows.append({
            "check_name": "zero_variance_waveforms",
            "partition": split_name,
            "passed": zero_var_count == 0,
            "issue_count": zero_var_count,
            "severity": "MEDIUM",
            "details": f"Found {zero_var_count} beats with standard deviation < 1e-6.",
        })

        # Check 10: Impossible RR intervals (if rr_sec available)
        if "rr_sec" in df.columns:
            valid_rr = df["rr_sec"].dropna()
            impossible_rr = int(((valid_rr < 0.2) | (valid_rr > 3.0)).sum())
        else:
            impossible_rr = 0
        audit_rows.append({
            "check_name": "impossible_rr_intervals",
            "partition": split_name,
            "passed": impossible_rr == 0,
            "issue_count": impossible_rr,
            "severity": "LOW",
            "details": f"Found {impossible_rr} RR intervals outside physiological bounds (0.2s - 3.0s).",
        })

    return pd.DataFrame(audit_rows)


def compare_partition_distributions(dfs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate partition class proportions and Jensen-Shannon divergence between partitions."""
    rows = []
    prob_vectors = {}

    for split_name, df in dfs_dict.items():
        total = len(df)
        counts = df["aami_class"].value_counts().to_dict()
        vec = np.array([counts.get(cls, 0) for cls in AAMI_CLASSES], dtype=np.float64)
        prob_vec = vec / (total if total > 0 else 1.0)
        prob_vectors[split_name] = prob_vec

        for cls in AAMI_CLASSES:
            cnt = counts.get(cls, 0)
            pct = (cnt / total * 100.0) if total > 0 else 0.0
            rows.append({
                "partition": split_name,
                "aami_class": cls,
                "count": cnt,
                "percentage": pct,
            })

    df_comp = pd.DataFrame(rows)

    # Compute pairwise Jensen-Shannon Divergence
    jsd_results = {}
    splits = list(dfs_dict.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            p, q = prob_vectors[s1], prob_vectors[s2]
            # scipy jensenshannon returns JS distance (sqrt of JS divergence)
            js_dist = jensenshannon(p, q)
            js_div = float(js_dist**2) if not np.isnan(js_dist) else 0.0
            jsd_results[f"jsd_{s1}_vs_{s2}"] = js_div

    logger.info(f"Jensen-Shannon Divergences between class distributions: {jsd_results}")
    return df_comp


def plot_partition_comparison(
    comparison_df: pd.DataFrame, out_path: Any, dpi: int = 200
) -> None:
    """Generate grouped bar plot comparing class proportions across Train, Val, and Test."""
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=comparison_df,
        x="aami_class",
        y="percentage",
        hue="partition",
        palette="Blues_d",
        ax=ax,
    )

    ax.set_title("Class Proportions (%) Comparison Across Train, Validation, and Test", fontsize=13)
    ax.set_xlabel("AAMI EC57 Class")
    ax.set_ylabel("Percentage of Partition (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Partition")

    # Annotate percentage values on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}%",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    out_path = str(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved partition class comparison figure to: {out_path}")
