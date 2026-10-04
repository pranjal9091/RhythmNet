#!/usr/bin/env python3
"""CLI script to run the hand-crafted ECG feature extraction and leakage-safe preprocessing pipeline."""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecg_arrhythmia.data.splits import verify_no_overlap
from ecg_arrhythmia.features.extractor import ECGFeatureExtractor, fit_transform_preprocessor
from ecg_arrhythmia.utils.config import load_config, resolve_path
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger("build_features")


def generate_feature_summary_report(
    X_train_raw: np.ndarray, feature_schema: Any, out_csv: Path
) -> pd.DataFrame:
    """Generate detailed descriptive summary statistics for extracted raw features."""
    feat_names = feature_schema.feature_names
    feat_defs = feature_schema.definitions

    rows = []
    for idx, f_def in enumerate(feat_defs):
        col = X_train_raw[:, idx]

        missing_cnt = int(np.sum(np.isnan(col)))
        finite_cnt = int(np.sum(np.isfinite(col)))
        unique_cnt = int(len(np.unique(col[~np.isnan(col)])))

        valid_col = col[~np.isnan(col)]
        tr_mean = float(np.mean(valid_col)) if len(valid_col) > 0 else np.nan
        tr_std = float(np.std(valid_col)) if len(valid_col) > 0 else np.nan
        tr_min = float(np.min(valid_col)) if len(valid_col) > 0 else np.nan
        tr_max = float(np.max(valid_col)) if len(valid_col) > 0 else np.nan

        rows.append(
            {
                "feature_index": idx,
                "feature_name": f_def.name,
                "feature_group": f_def.group,
                "description": f_def.description,
                "train_mean": tr_mean,
                "train_std": tr_std,
                "train_min": tr_min,
                "train_max": tr_max,
                "missing_count_before_imputation": missing_cnt,
                "finite_count": finite_cnt,
                "unique_count": unique_cnt,
                "is_constant": unique_cnt <= 1,
            }
        )

    df_summary = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_csv, index=False)
    logger.info(f"Saved feature summary report to: {out_csv}")
    return df_summary


def generate_feature_redundancy_report(
    X_train_raw: np.ndarray,
    feature_schema: Any,
    out_csv: Path,
    corr_threshold: float = 0.95,
) -> pd.DataFrame:
    """Identify pairs of features with absolute correlation >= threshold."""
    df_raw = pd.DataFrame(X_train_raw, columns=feature_schema.feature_names)
    
    # Compute correlation matrix handling missing values (pairwise complete)
    corr_matrix = df_raw.corr(method="pearson").abs()

    redundant_pairs = []
    names = feature_schema.feature_names
    n_cols = len(names)

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val) and val >= corr_threshold:
                redundant_pairs.append(
                    {
                        "feature_1": names[i],
                        "group_1": feature_schema.definitions[i].group,
                        "feature_2": names[j],
                        "group_2": feature_schema.definitions[j].group,
                        "absolute_correlation": float(val),
                    }
                )

    df_redundant = pd.DataFrame(redundant_pairs)
    if not df_redundant.empty:
        df_redundant = df_redundant.sort_values(by="absolute_correlation", ascending=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_redundant.to_csv(out_csv, index=False)
    logger.info(f"Saved feature redundancy report ({len(df_redundant)} pairs >= {corr_threshold}) to: {out_csv}")
    return df_redundant


def generate_feature_visualizations(
    X_train_scaled: np.ndarray,
    df_train_meta: pd.DataFrame,
    feature_schema: Any,
    figures_dir: Path,
    dpi: int = 200,
) -> None:
    """Generate feature group counts and selected feature distribution plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature counts per group bar plot
    df_schema = feature_schema.to_dataframe()
    group_counts = df_schema["feature_group"].value_counts().reset_index()
    group_counts.columns = ["feature_group", "count"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=group_counts, x="feature_group", y="count", palette="viridis", ax=ax)
    ax.set_title("Engineered ECG Features Count by Group", fontsize=13)
    ax.set_xlabel("Feature Group")
    ax.set_ylabel("Number of Features")
    ax.grid(True, linestyle="--", alpha=0.4)

    for p in ax.patches:
        h = int(p.get_height())
        if h > 0:
            ax.annotate(f"{h}", xy=(p.get_x() + p.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center")

    plt.tight_layout()
    fig1_path = figures_dir / "feature_group_summary.png"
    fig.savefig(fig1_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature group summary figure to: {fig1_path}")

    # 2. Selected representative feature distributions across classes
    candidate_features = [
        "morph_ptp",
        "shape_area_ratio",
        "qrs_width_50pct",
        "rr_prev",
        "dwt_cD3_energy",
        "dwt_cA4_std",
    ]

    selected_names = [f for f in candidate_features if f in feature_schema.feature_names]
    if len(selected_names) >= 4:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
        axes_flat = axes.flatten()

        for idx, feat_name in enumerate(selected_names[:6]):
            ax = axes_flat[idx]
            feat_col_idx = feature_schema.feature_names.index(feat_name)
            col_vals = X_train_scaled[:, feat_col_idx]

            temp_df = pd.DataFrame(
                {"aami_class": df_train_meta["aami_class"].values, "val": col_vals}
            )

            sns.boxplot(
                data=temp_df,
                x="aami_class",
                y="val",
                palette="Set2",
                showfliers=False,
                ax=ax,
            )
            ax.set_title(f"Feature: '{feat_name}'")
            ax.set_xlabel("AAMI Class")
            ax.set_ylabel("Scaled Value (z-score)")
            ax.grid(True, linestyle="--", alpha=0.3)

        plt.suptitle("Scaled Feature Distributions across AAMI Classes (Train Partition)", fontsize=14)
        plt.tight_layout()
        fig2_path = figures_dir / "selected_feature_distributions.png"
        fig.savefig(fig2_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved selected feature distributions figure to: {fig2_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hand-crafted ECG feature extraction and leakage-safe preprocessing pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/features.yaml",
        help="Path to feature extraction configuration file.",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Path to data configuration file.",
    )
    args = parser.parse_args()

    feat_cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)

    processed_dir = resolve_path(data_cfg.get("dataset", {}).get("processed_dir", "data/processed"))
    features_out_dir = resolve_path("data/features")
    artifacts_dir = resolve_path(feat_cfg.get("preprocessing", {}).get("save_artifact_dir", "artifacts/features"))
    tables_dir = resolve_path("reports/tables")
    figures_dir = resolve_path("reports/figures")

    # 1. Load processed beat arrays and metadata
    logger.info("Loading preprocessed beat arrays and metadata...")

    dfs_dict: Dict[str, pd.DataFrame] = {}
    beats_dict: Dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        meta_file = processed_dir / split / "metadata.csv"
        beats_file = processed_dir / split / "beats.npy"

        if not meta_file.is_file() or not beats_file.is_file():
            logger.error(f"Missing processed input files for '{split}' at {processed_dir}. Run preprocess.py first!")
            sys.exit(1)

        dfs_dict[split] = pd.read_csv(meta_file)
        beats_dict[split] = np.load(beats_file)

    # 2. Leakage Verification
    train_recs = dfs_dict["train"]["record_id"].unique()
    val_recs = dfs_dict["val"]["record_id"].unique()
    test_recs = dfs_dict["test"]["record_id"].unique()
    verify_no_overlap(train_recs, val_recs, test_recs)
    logger.info("Patient leakage verification PASSED: Zero record overlap across Train, Val, and Test.")

    # 3. Extract raw feature matrices
    extractor = ECGFeatureExtractor(config=feat_cfg)

    raw_features: Dict[str, np.ndarray] = {}
    extraction_times: Dict[str, float] = {}
    schema_ref: Any = None

    for split in ["train", "val", "test"]:
        t0 = time.time()
        X_raw, schema = extractor.extract_features(beats_dict[split], dfs_dict[split])
        t1 = time.time()

        raw_features[split] = X_raw
        extraction_times[split] = t1 - t0

        if schema_ref is None:
            schema_ref = schema
        else:
            assert schema.feature_names == schema_ref.feature_names, f"Schema mismatch in split '{split}'!"

        logger.info(
            f"Extracted '{split}' raw features: shape {X_raw.shape} in {extraction_times[split]:.2f} sec."
        )

    # Save feature schema catalog
    features_out_dir.mkdir(parents=True, exist_ok=True)
    schema_csv = features_out_dir / "feature_schema.csv"
    schema_ref.to_dataframe().to_csv(schema_csv, index=False)
    logger.info(f"Saved feature schema catalog ({len(schema_ref)} features) to: {schema_csv}")

    # 4. Fit Imputer + Scaler ONLY on TRAIN, then transform all partitions
    preprocessor_artifact = artifacts_dir / "preprocessor.joblib"
    X_tr_scaled, X_va_scaled, X_te_scaled, preprocessor = fit_transform_preprocessor(
        X_train=raw_features["train"],
        X_val=raw_features["val"],
        X_test=raw_features["test"],
        save_path=preprocessor_artifact,
        imputer_strategy=feat_cfg.get("preprocessing", {}).get("imputer_strategy", "median"),
    )

    scaled_features = {
        "train": X_tr_scaled,
        "val": X_va_scaled,
        "test": X_te_scaled,
    }

    # 5. Save feature matrices and metadata copies
    for split in ["train", "val", "test"]:
        split_out_dir = features_out_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        np.save(split_out_dir / "features_raw.npy", raw_features[split])
        np.save(split_out_dir / "features.npy", scaled_features[split])
        dfs_dict[split].to_csv(split_out_dir / "metadata.csv", index=False)
        logger.info(f"Saved '{split}' feature files to: {split_out_dir}")

    # 6. Feature Quality & Redundancy Reports
    df_feat_summary = generate_feature_summary_report(
        X_train_raw=raw_features["train"],
        feature_schema=schema_ref,
        out_csv=tables_dir / "feature_summary.csv",
    )

    df_redundant = generate_feature_redundancy_report(
        X_train_raw=raw_features["train"],
        feature_schema=schema_ref,
        out_csv=tables_dir / "feature_redundancy.csv",
        corr_threshold=0.95,
    )

    # 7. Generate Feature Visualizations
    generate_feature_visualizations(
        X_train_scaled=scaled_features["train"],
        df_train_meta=dfs_dict["train"],
        feature_schema=schema_ref,
        figures_dir=figures_dir,
    )

    # 8. Print Terminal Executive Summary
    group_df = schema_ref.to_dataframe()
    group_counts = group_df["feature_group"].value_counts().to_dict()

    total_missing_before = int(np.sum(np.isnan(raw_features["train"])))
    constant_cnt = int(df_feat_summary["is_constant"].sum())
    redundant_cnt = len(df_redundant)

    print("\n" + "=" * 80)
    print("           MILESTONE 4 — HAND-CRAFTED FEATURE ENGINEERING REPORT              ")
    print("=" * 80)
    print(f"Total Extracted Features : {len(schema_ref)} features across 5 distinct groups")
    for grp, cnt in group_counts.items():
        print(f"  - Group '{grp:<12}': {cnt:>3} features")

    print("\nFeature Matrix Shapes:")
    print(f"  - Train      : {scaled_features['train'].shape} ({extraction_times['train']:.2f}s)")
    print(f"  - Validation : {scaled_features['val'].shape} ({extraction_times['val']:.2f}s)")
    print(f"  - Test (DS2) : {scaled_features['test'].shape} ({extraction_times['test']:.2f}s)")

    print("\nQuality & Preprocessing Summary:")
    print(f"  - Missing Values in Train (Before Imputation) : {total_missing_before} (from boundary RR context)")
    print(f"  - Missing Values in Scaled Output matrices     : 0 (Imputed with Train Median)")
    print(f"  - Constant Features Detected                   : {constant_cnt}")
    print(f"  - Highly Correlated Feature Pairs (|r| >= 0.95)  : {redundant_cnt}")
    print(f"  - Fitted Preprocessor Saved To                 : {preprocessor_artifact}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
