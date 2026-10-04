"""
CLI script to run Milestone 7 Class Imbalance Ablation Study.

Usage:
    python scripts/run_imbalance_ablation.py --config configs/imbalance_ablation.yaml
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from ecg_arrhythmia.utils import get_logger, load_config, seed_everything, get_project_root
from ecg_arrhythmia.deep_learning import (
    ECGBeatDataset,
    ECG1DCNN,
    FocalLoss,
    compute_inverse_class_weights,
    compute_sqrt_inverse_class_weights,
    create_weighted_sampler,
    get_effective_sampling_stats,
    train_deep_model,
    predict_deep_model,
    measure_deep_inference_latency,
    get_model_param_counts,
)
from ecg_arrhythmia.models import (
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_confusion_matrices,
    get_model_size_mb,
    plot_confusion_matrix,
    DEFAULT_LABEL_MAPPING,
    DEFAULT_CLASS_NAMES,
)

logger = get_logger("imbalance_ablation")


def plot_imbalance_per_class_f1(per_class_df: pd.DataFrame, output_path: Path) -> None:
    """Generates grouped bar chart comparing per-class F1-scores across imbalance experiments."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df = per_class_df[per_class_df["partition"] == "test"].copy()

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=test_df,
        x="class",
        y="f1",
        hue="experiment",
        palette="tab10",
        order=DEFAULT_CLASS_NAMES,
    )
    plt.title("Per-Class F1-Score Comparison Across Imbalance Strategies (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("AAMI EC57 Class", fontsize=11, labelpad=8)
    plt.ylabel("F1-Score", fontsize=11, labelpad=8)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Imbalance Strategy", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0.01:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=7,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_imbalance_macro_f1(overall_df: pd.DataFrame, output_path: Path) -> None:
    """Generates bar chart comparing Macro-F1 scores across validation and test sets."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=overall_df,
        x="experiment",
        y="macro_f1",
        hue="partition",
        palette="Set2",
    )
    plt.title("Macro-F1 Score Across Imbalance Strategies", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Experiment Strategy", fontsize=11, labelpad=8)
    plt.ylabel("Macro-F1 Score", fontsize=11, labelpad=8)
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Partition", frameon=True)

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone 7 Class Imbalance Ablation Study")
    parser.add_argument("--config", type=str, default="configs/imbalance_ablation.yaml", help="Path to imbalance config")
    parser.add_argument("--experiment", type=str, default=None, help="Optionally run a single experiment by name")
    args = parser.parse_args()

    project_root = get_project_root()
    config_path = project_root / args.config

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    seed = config.get("seed", 42)
    seed_everything(seed)

    # Device auto-selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using compute device: {device}")

    # Load preprocessed datasets
    data_dir = project_root / "data" / "processed"
    datasets = {}
    for part in ["train", "val", "test"]:
        beats_path = data_dir / part / "beats.npy"
        meta_path = data_dir / part / "metadata.csv"
        datasets[part] = ECGBeatDataset(beats=beats_path, metadata=meta_path, label_mapping=DEFAULT_LABEL_MAPPING)

    y_train = datasets["train"].labels
    y_val = datasets["val"].labels
    y_test = datasets["test"].labels

    reports_tables_dir = project_root / "reports" / "tables"
    reports_figures_dir = project_root / "reports" / "figures"
    reports_pred_dir = project_root / "reports" / "predictions" / "m7"
    m7_models_dir = project_root / "artifacts" / "models" / "deep_learning" / "m7"

    reports_tables_dir.mkdir(parents=True, exist_ok=True)
    reports_figures_dir.mkdir(parents=True, exist_ok=True)
    reports_pred_dir.mkdir(parents=True, exist_ok=True)
    m7_models_dir.mkdir(parents=True, exist_ok=True)

    # Audit training class weights and sampling stats
    sampling_stats_df = get_effective_sampling_stats(y_train, class_names=DEFAULT_CLASS_NAMES)
    sampling_stats_df.to_csv(reports_tables_dir / "imbalance_sampling_distribution.csv", index=False)
    
    weights_summary_df = sampling_stats_df[["class", "count", "inverse_frequency_weight", "sqrt_inverse_frequency_weight"]]
    weights_summary_df.to_csv(reports_tables_dir / "imbalance_class_weights.csv", index=False)
    logger.info(f"Audited class weights and sampling distribution saved to {reports_tables_dir}")

    exp_list = config.get("experiments", [])
    if args.experiment:
        exp_list = [e for e in exp_list if e["name"] == args.experiment]
        if not exp_list:
            logger.error(f"Experiment '{args.experiment}' not found in config!")
            sys.exit(1)

    batch_size = config.get("training", {}).get("batch_size", 256)
    num_workers = config.get("runtime", {}).get("num_workers", 0)

    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    overall_results = []
    per_class_results = []
    run_records = []
    error_summary_rows = []

    labels_list = [DEFAULT_LABEL_MAPPING[c] for c in DEFAULT_CLASS_NAMES]

    for exp in exp_list:
        exp_name = exp["name"]
        strategy = exp["strategy"]
        logger.info(f"\n==================================================")
        logger.info(f"RUNNING M7 EXPERIMENT: {exp_name} ({strategy})")
        logger.info(f"==================================================")

        seed_everything(seed)

        # DataLoader setup
        if strategy == "weighted_random_sampler":
            sampler = create_weighted_sampler(y_train, num_classes=5, seed=seed)
            train_loader = DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        else:
            train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # Loss function setup
        if strategy == "weighted_ce_inverse":
            w = compute_inverse_class_weights(y_train, num_classes=5).to(device)
            criterion = nn.CrossEntropyLoss(weight=w)
            class_weights_str = str(w.cpu().numpy().tolist())
        elif strategy == "weighted_ce_sqrt_inverse":
            w = compute_sqrt_inverse_class_weights(y_train, num_classes=5).to(device)
            criterion = nn.CrossEntropyLoss(weight=w)
            class_weights_str = str(w.cpu().numpy().tolist())
        elif strategy == "focal":
            gamma = exp.get("gamma", 2.0)
            criterion = FocalLoss(gamma=gamma, weight=None)
            class_weights_str = "None (focal gamma=2.0)"
        else:
            criterion = nn.CrossEntropyLoss()
            class_weights_str = "None"

        # Model setup
        cfg_model = config.get("model", {})
        model = ECG1DCNN(
            in_channels=cfg_model.get("in_channels", 1),
            num_classes=cfg_model.get("num_classes", 5),
            dropout_rates=tuple(cfg_model.get("dropout_rates", (0.15, 0.20, 0.30))),
        )

        checkpoint_path = m7_models_dir / f"{exp_name}.pt"

        # Train model
        t_start = time.time()
        model, history_df, run_summary = train_deep_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_path=checkpoint_path,
            criterion=criterion,
        )
        t_duration = time.time() - t_start

        ckpt_size_mb = get_model_size_mb(checkpoint_path)
        latency_mets = measure_deep_inference_latency(model, test_loader, device)

        run_records.append({
            "experiment": exp_name,
            "strategy": strategy,
            "seed": seed,
            "device": str(device),
            "epochs_trained": run_summary["total_epochs"],
            "best_epoch": run_summary["best_epoch"],
            "best_val_macro_f1": run_summary["best_val_macro_f1"],
            "class_weights": class_weights_str,
            "duration_sec": t_duration,
        })

        # Evaluate on validation and test
        for part_name, loader, dataset in [("val", val_loader, datasets["val"]), ("test", test_loader, datasets["test"])]:
            y_pred, y_prob = predict_deep_model(model, loader, device)
            y_true_part = dataset.labels
            meta_df = dataset.metadata_df

            # Overall metrics
            overall_mets = compute_overall_metrics(
                y_true=y_true_part,
                y_pred=y_pred,
                y_prob=y_prob,
                labels=labels_list,
                n_bins=config.get("evaluation", {}).get("ece_bins", 10),
            )
            overall_mets.update({
                "experiment": exp_name,
                "partition": part_name,
                "latency_ms_per_beat": latency_mets["ms_per_beat"],
                "model_size_mb": ckpt_size_mb,
            })
            overall_results.append(overall_mets)

            # Per-class metrics
            per_class_df = compute_per_class_metrics(
                y_true=y_true_part,
                y_pred=y_pred,
                y_prob=y_prob,
                class_names=DEFAULT_CLASS_NAMES,
                label_mapping=DEFAULT_LABEL_MAPPING,
            )
            per_class_df["experiment"] = exp_name
            per_class_df["partition"] = part_name
            per_class_results.append(per_class_df)

            # Predictions CSV
            reverse_map = {i: c for i, c in enumerate(DEFAULT_CLASS_NAMES)}
            pred_df = pd.DataFrame()
            pred_df["beat_id"] = meta_df.get("beat_id", np.arange(len(meta_df)))
            pred_df["record_id"] = meta_df.get("record_id", "")
            pred_df["sample_index"] = meta_df.get("sample_index", 0)
            pred_df["true_class"] = [reverse_map[y] for y in y_true_part]
            pred_df["predicted_class"] = [reverse_map[y] for y in y_pred]
            pred_df["confidence"] = np.max(y_prob, axis=1)
            for i, c in enumerate(DEFAULT_CLASS_NAMES):
                pred_df[f"probability_{c}"] = y_prob[:, i]

            pred_csv_path = reports_pred_dir / f"{exp_name}_{part_name}.csv"
            pred_df.to_csv(pred_csv_path, index=False)

            # Confusion matrices
            cm_raw, cm_norm = compute_confusion_matrices(y_true_part, y_pred, labels=labels_list)
            cm_png_path = reports_figures_dir / f"confusion_matrix_{exp_name}_{part_name}.png"
            plot_confusion_matrix(
                cm=cm_raw,
                class_names=DEFAULT_CLASS_NAMES,
                title=f"Confusion Matrix: {exp_name} ({part_name.upper()})",
                output_path=cm_png_path,
                fmt="d",
            )

            # Record confusion patterns for error analysis
            if part_name == "test":
                cm_df = pd.DataFrame(cm_raw, index=DEFAULT_CLASS_NAMES, columns=DEFAULT_CLASS_NAMES)
                for t_cls in DEFAULT_CLASS_NAMES:
                    for p_cls in DEFAULT_CLASS_NAMES:
                        error_summary_rows.append({
                            "experiment": exp_name,
                            "true_class": t_cls,
                            "predicted_class": p_cls,
                            "count": int(cm_df.loc[t_cls, p_cls]),
                        })

    # Combine metrics into comparison tables
    all_overall_df = pd.DataFrame(overall_results)
    all_per_class_df = pd.concat(per_class_results, ignore_index=True)

    # Pivot per-class F1 for test comparison table
    test_overall = all_overall_df[all_overall_df["partition"] == "test"].copy()
    test_per_class = all_per_class_df[all_per_class_df["partition"] == "test"].copy()

    f1_piv = test_per_class.pivot(index="experiment", columns="class", values="f1")
    f1_piv.columns = [f"{c}_f1" for c in f1_piv.columns]

    test_comp_df = test_overall.merge(f1_piv, on="experiment", how="left")
    cols_order = [
        "experiment", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "macro_pr_auc", "macro_roc_auc", "ece",
        "N_f1", "S_f1", "V_f1", "F_f1", "Q_f1", "latency_ms_per_beat", "model_size_mb"
    ]
    test_comp_df = test_comp_df[cols_order]
    test_comp_df.to_csv(reports_tables_dir / "imbalance_ablation_comparison.csv", index=False)

    val_overall = all_overall_df[all_overall_df["partition"] == "val"].copy()
    val_per_class = all_per_class_df[all_per_class_df["partition"] == "val"].copy()
    val_f1_piv = val_per_class.pivot(index="experiment", columns="class", values="f1")
    val_f1_piv.columns = [f"{c}_f1" for c in val_f1_piv.columns]
    val_comp_df = val_overall.merge(val_f1_piv, on="experiment", how="left")[cols_order]
    val_comp_df.to_csv(reports_tables_dir / "imbalance_ablation_val_comparison.csv", index=False)

    pd.DataFrame(run_records).to_csv(reports_tables_dir / "m7_experiment_runs.csv", index=False)
    pd.DataFrame(error_summary_rows).to_csv(reports_tables_dir / "imbalance_error_summary.csv", index=False)

    # Plot comparison figures
    plot_imbalance_per_class_f1(all_per_class_df, reports_figures_dir / "imbalance_per_class_f1.png")
    plot_imbalance_macro_f1(all_overall_df, reports_figures_dir / "imbalance_macro_f1_comparison.png")

    logger.info("\n==================================================")
    logger.info("MILESTONE 7 IMBALANCE ABLATION SUMMARY (TEST SET)")
    logger.info("==================================================")
    print(test_comp_df[["experiment", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "S_f1", "V_f1", "F_f1", "Q_f1", "ece"]].to_string(index=False))


if __name__ == "__main__":
    main()
