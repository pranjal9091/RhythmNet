"""
CLI Script to train and evaluate 1D Convolutional Neural Network (CNN) Baseline on Raw ECG Beats.

Usage:
    python scripts/train_cnn.py --config configs/deep_learning.yaml
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecg_arrhythmia.utils import get_logger, load_config, seed_everything, get_project_root
from ecg_arrhythmia.deep_learning import (
    ECGBeatDataset,
    ECG1DCNN,
    train_deep_model,
    predict_deep_model,
    measure_deep_inference_latency,
    get_model_param_counts,
    plot_training_curves,
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

logger = get_logger("train_cnn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D CNN Baseline on Raw ECG Waveforms")
    parser.add_argument("--config", type=str, default="configs/deep_learning.yaml", help="Path to deep learning config file")
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
    partitions = ["train", "val", "test"]
    datasets = {}

    for part in partitions:
        beats_path = data_dir / part / "beats.npy"
        meta_path = data_dir / part / "metadata.csv"

        if not beats_path.exists() or not meta_path.exists():
            logger.error(f"Processed beats or metadata missing for {part} in {data_dir / part}")
            sys.exit(1)

        dataset = ECGBeatDataset(beats=beats_path, metadata=meta_path, label_mapping=DEFAULT_LABEL_MAPPING)
        datasets[part] = dataset
        logger.info(f"Loaded {part} dataset: {len(dataset)} beats, shape {dataset.beats.shape}")

    # Data integrity checks
    assert datasets["train"].beats.shape == (41262, 216), "Train shape mismatch!"
    assert datasets["val"].beats.shape == (9744, 216), "Val shape mismatch!"
    assert datasets["test"].beats.shape == (49694, 216), "Test shape mismatch!"
    
    for part in partitions:
        assert not np.isnan(datasets[part].beats).any(), f"NaN in {part} beats!"
        assert not np.isinf(datasets[part].beats).any(), f"Inf in {part} beats!"

    # Disjoint record verification
    train_recs = set(datasets["train"].metadata_df["record_id"].unique())
    val_recs = set(datasets["val"].metadata_df["record_id"].unique())
    test_recs = set(datasets["test"].metadata_df["record_id"].unique())

    assert len(train_recs.intersection(val_recs)) == 0, "Train and Val records overlap!"
    assert len(train_recs.intersection(test_recs)) == 0, "Train and Test records overlap!"
    assert len(val_recs.intersection(test_recs)) == 0, "Val and Test records overlap!"
    logger.info("Data integrity checks PASSED (shapes verified, 0 NaNs, 0 record leakage).")

    # Build DataLoaders
    batch_size = config.get("training", {}).get("batch_size", 256)
    num_workers = config.get("runtime", {}).get("num_workers", 0)

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build 1D CNN
    cfg_model = config.get("model", {})
    dropout_rates = tuple(cfg_model.get("dropout_rates", (0.15, 0.20, 0.30)))
    model = ECG1DCNN(
        in_channels=cfg_model.get("in_channels", 1),
        num_classes=cfg_model.get("num_classes", 5),
        dropout_rates=dropout_rates,
    )

    total_params, trainable_params = get_model_param_counts(model)
    logger.info(f"Instantiated 1D CNN model: {total_params:,} total parameters ({trainable_params:,} trainable)")

    checkpoint_path = project_root / "artifacts" / "models" / "deep_learning" / "cnn_best.pt"

    # Train CNN baseline
    model, history_df, run_summary = train_deep_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    # Save training history & curves
    reports_tables_dir = project_root / "reports" / "tables"
    reports_figures_dir = project_root / "reports" / "figures"
    reports_pred_dir = project_root / "reports" / "predictions"

    reports_tables_dir.mkdir(parents=True, exist_ok=True)
    reports_figures_dir.mkdir(parents=True, exist_ok=True)
    reports_pred_dir.mkdir(parents=True, exist_ok=True)

    history_csv_path = reports_tables_dir / "cnn_training_history.csv"
    history_df.to_csv(history_csv_path, index=False)

    curves_png_path = reports_figures_dir / "cnn_training_curves.png"
    plot_training_curves(history_df, curves_png_path)
    logger.info(f"Saved training history to {history_csv_path} and curves to {curves_png_path}")

    # Model summary table
    checkpoint_size_mb = get_model_size_mb(checkpoint_path)
    
    # Latency benchmarking on test set
    latency_mets = measure_deep_inference_latency(model, test_loader, device)
    logger.info(f"Test Latency: {latency_mets['ms_per_beat']:.4f} ms/beat (Single Beat: {latency_mets['single_beat_ms']:.4f} ms)")

    summary_df = pd.DataFrame([{
        "model": "1D_CNN",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint_size_mb": checkpoint_size_mb,
        "inference_latency_ms_per_beat": latency_mets["ms_per_beat"],
        "single_beat_latency_ms": latency_mets["single_beat_ms"],
        "throughput_beats_per_sec": latency_mets["throughput_beats_per_sec"],
        "device": str(device),
    }])
    summary_df.to_csv(reports_tables_dir / "deep_model_summary.csv", index=False)

    # Training run summary table
    run_df = pd.DataFrame([{
        "model": "1D_CNN",
        "device": str(device),
        "seed": seed,
        "total_epochs_trained": run_summary["total_epochs"],
        "best_epoch": run_summary["best_epoch"],
        "best_val_macro_f1": run_summary["best_val_macro_f1"],
        "total_training_time_sec": run_summary["total_train_time_sec"],
        "avg_epoch_time_sec": run_summary["avg_epoch_time_sec"],
    }])
    run_df.to_csv(reports_tables_dir / "cnn_training_run.csv", index=False)

    # Evaluate on Validation and Test
    labels_list = [DEFAULT_LABEL_MAPPING[c] for c in DEFAULT_CLASS_NAMES]
    overall_results = []
    per_class_results = []

    for part_name, loader, dataset in [("val", val_loader, datasets["val"]), ("test", test_loader, datasets["test"])]:
        y_pred, y_prob = predict_deep_model(model, loader, device)
        y_true = dataset.labels
        meta_df = dataset.metadata_df

        # Overall metrics
        overall_mets = compute_overall_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            labels=labels_list,
            n_bins=config.get("evaluation", {}).get("ece_bins", 10),
        )
        overall_mets.update({
            "model": "1D_CNN",
            "partition": part_name,
            "inference_latency_ms_per_beat": latency_mets["ms_per_beat"],
            "model_size_mb": checkpoint_size_mb,
        })
        overall_results.append(overall_mets)

        # Per-class metrics
        per_class_df = compute_per_class_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=DEFAULT_CLASS_NAMES,
            label_mapping=DEFAULT_LABEL_MAPPING,
        )
        per_class_df["model"] = "1D_CNN"
        per_class_df["partition"] = part_name
        per_class_results.append(per_class_df)

        # Export predictions CSV
        reverse_map = {i: c for i, c in enumerate(DEFAULT_CLASS_NAMES)}
        pred_df = pd.DataFrame()
        pred_df["beat_id"] = meta_df.get("beat_id", np.arange(len(meta_df)))
        pred_df["record_id"] = meta_df.get("record_id", "")
        pred_df["sample_index"] = meta_df.get("sample_index", 0)
        pred_df["true_class"] = [reverse_map[y] for y in y_true]
        pred_df["predicted_class"] = [reverse_map[y] for y in y_pred]
        pred_df["confidence"] = np.max(y_prob, axis=1)
        for i, c in enumerate(DEFAULT_CLASS_NAMES):
            pred_df[f"probability_{c}"] = y_prob[:, i]

        pred_csv_path = reports_pred_dir / f"cnn_{part_name}.csv"
        pred_df.to_csv(pred_csv_path, index=False)

        # Confusion matrix plots
        cm_raw, cm_norm = compute_confusion_matrices(y_true, y_pred, labels=labels_list)
        cm_raw_path = reports_figures_dir / f"confusion_matrix_cnn_{part_name}.png"
        plot_confusion_matrix(
            cm=cm_raw,
            class_names=DEFAULT_CLASS_NAMES,
            title=f"Confusion Matrix: 1D CNN ({part_name.upper()})",
            output_path=cm_raw_path,
            fmt="d",
        )

        # CNN error summary table
        if part_name == "test":
            cm_df = pd.DataFrame(cm_raw, index=DEFAULT_CLASS_NAMES, columns=DEFAULT_CLASS_NAMES)
            err_rows = []
            for t_cls in DEFAULT_CLASS_NAMES:
                for p_cls in DEFAULT_CLASS_NAMES:
                    count = int(cm_df.loc[t_cls, p_cls])
                    err_rows.append({"true_class": t_cls, "predicted_class": p_cls, "count": count})
            pd.DataFrame(err_rows).to_csv(reports_tables_dir / "cnn_error_summary.csv", index=False)

    # Classical vs CNN Comparison Table
    classical_comp_path = reports_tables_dir / "classical_model_comparison.csv"
    if classical_comp_path.exists():
        class_df = pd.read_csv(classical_comp_path)
        cnn_df = pd.DataFrame(overall_results)
        combined_df = pd.concat([class_df, cnn_df], ignore_index=True)
        combined_df.to_csv(reports_tables_dir / "classical_vs_cnn.csv", index=False)
        logger.info(f"Saved classical vs CNN comparison table to {reports_tables_dir / 'classical_vs_cnn.csv'}")

    logger.info("\n==================================================")
    logger.info("MILESTONE 6 1D CNN BASELINE SUMMARY")
    logger.info("==================================================")
    print("\nOVERALL METRICS (TEST SET):")
    cnn_test_res = pd.DataFrame(overall_results)
    cnn_test_res = cnn_test_res[cnn_test_res["partition"] == "test"]
    print(cnn_test_res[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_pr_auc", "ece", "inference_latency_ms_per_beat"]].to_string(index=False))


if __name__ == "__main__":
    main()
