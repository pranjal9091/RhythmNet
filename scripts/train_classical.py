"""
CLI Script to train and evaluate Classical Machine Learning Baselines for ECG Arrhythmia Classification.

Usage:
    python scripts/train_classical.py --config configs/classical.yaml
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd

from ecg_arrhythmia.utils import get_logger, load_config, seed_everything, get_project_root
from ecg_arrhythmia.models import (
    build_classical_model,
    train_model,
    predict_model,
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_confusion_matrices,
    save_model,
    load_model,
    save_label_mapping,
    get_model_size_mb,
    measure_inference_latency,
    generate_predictions_dataframe,
    plot_confusion_matrix,
    plot_per_class_f1_comparison,
    DEFAULT_LABEL_MAPPING,
    DEFAULT_CLASS_NAMES,
)

logger = get_logger("train_classical")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Classical ML Baselines on Hand-Crafted ECG Features")
    parser.add_argument("--config", type=str, default="configs/classical.yaml", help="Path to classical ML config file")
    args = parser.parse_args()

    project_root = get_project_root()
    config_path = project_root / args.config

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    seed = config.get("reproducibility", {}).get("seed", 42)
    seed_everything(seed)

    logger.info(f"Loaded configuration from {config_path}")

    # Label mapping setup
    label_mapping = config.get("label_mapping", DEFAULT_LABEL_MAPPING)
    labels_list = [label_mapping[c] for c in DEFAULT_CLASS_NAMES]
    
    # Save label mapping artifact
    label_mapping_path = project_root / "artifacts" / "models" / "classical" / "label_mapping.json"
    save_label_mapping(label_mapping, label_mapping_path)
    logger.info(f"Saved label mapping to {label_mapping_path}")

    # Load feature matrices and metadata
    features_dir = project_root / "data" / "features"
    partitions = ["train", "val", "test"]
    data = {}

    for part in partitions:
        feat_path = features_dir / part / "features.npy"
        meta_path = features_dir / part / "metadata.csv"

        if not feat_path.exists() or not meta_path.exists():
            logger.error(f"Feature matrix or metadata missing for {part} in {features_dir / part}")
            sys.exit(1)

        X = np.load(feat_path)
        meta_df = pd.read_csv(meta_path)
        y = np.array([label_mapping[c] for c in meta_df["aami_class"].values], dtype=int)

        data[part] = {
            "X": X,
            "y": y,
            "meta": meta_df,
        }
        logger.info(f"Loaded {part}: features shape {X.shape}, labels shape {y.shape}")

    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["val"]["X"], data["val"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]

    models_config = config.get("models", {})
    enabled_models = [m for m, cfg in models_config.items() if cfg.get("enabled", True)]

    overall_results = []
    per_class_results = []
    
    models_dir = project_root / "artifacts" / "models" / "classical"
    reports_figures_dir = project_root / "reports" / "figures"
    reports_tables_dir = project_root / "reports" / "tables"
    reports_pred_dir = project_root / "reports" / "predictions"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_figures_dir.mkdir(parents=True, exist_ok=True)
    reports_tables_dir.mkdir(parents=True, exist_ok=True)
    reports_pred_dir.mkdir(parents=True, exist_ok=True)

    for model_name in enabled_models:
        m_cfg = models_config[model_name]
        logger.info(f"\n==================================================")
        logger.info(f"Training Model: {model_name}")
        logger.info(f"==================================================")

        # Build and train model
        model = build_classical_model(model_name, m_cfg)
        t0 = time.time()
        model = train_model(model, X_train, y_train)
        train_time = time.time() - t0
        logger.info(f"Training completed in {train_time:.2f}s")

        # Measure inference latency on test set
        _, latency_ms_per_beat = measure_inference_latency(model, X_test)
        logger.info(f"Inference latency on test set: {latency_ms_per_beat:.4f} ms/beat")

        # Serialize model artifact
        if model_name == "xgboost":
            model_file = models_dir / f"{model_name}.json"
        else:
            model_file = models_dir / f"{model_name}.joblib"
            
        save_model(model, model_file)
        model_size_mb = get_model_size_mb(model_file)
        logger.info(f"Saved model artifact to {model_file} ({model_size_mb:.2f} MB)")

        # Reload verification
        reloaded_model = load_model(model_file, model_type=model_name)
        reloaded_pred, reloaded_prob = predict_model(reloaded_model, X_val[:50])
        orig_pred, orig_prob = predict_model(model, X_val[:50])
        np.testing.assert_array_equal(orig_pred, reloaded_pred, err_msg="Reloaded model prediction mismatch!")
        logger.info("Model reload verification: PASSED (exact prediction match)")

        # Evaluate on Validation and Test
        for part_name, part_data in [("val", data["val"]), ("test", data["test"])]:
            X_part = part_data["X"]
            y_part = part_data["y"]
            meta_part = part_data["meta"]

            y_pred, y_prob = predict_model(model, X_part)

            # Overall metrics
            overall_mets = compute_overall_metrics(
                y_true=y_part,
                y_pred=y_pred,
                y_prob=y_prob,
                labels=labels_list,
                n_bins=config.get("evaluation", {}).get("ece_bins", 10),
            )
            overall_mets.update({
                "model": model_name,
                "partition": part_name,
                "inference_latency_ms_per_beat": latency_ms_per_beat,
                "model_size_mb": model_size_mb,
            })
            overall_results.append(overall_mets)

            # Per-class metrics
            per_class_df = compute_per_class_metrics(
                y_true=y_part,
                y_pred=y_pred,
                y_prob=y_prob,
                class_names=DEFAULT_CLASS_NAMES,
                label_mapping=label_mapping,
            )
            per_class_df["model"] = model_name
            per_class_df["partition"] = part_name
            per_class_results.append(per_class_df)

            # Save predictions CSV
            pred_df = generate_predictions_dataframe(
                metadata_df=meta_part,
                y_true=y_part,
                y_pred=y_pred,
                y_prob=y_prob,
                class_names=DEFAULT_CLASS_NAMES,
            )
            pred_csv_path = reports_pred_dir / f"{model_name}_{part_name}.csv"
            pred_df.to_csv(pred_csv_path, index=False)
            logger.info(f"Saved predictions CSV to {pred_csv_path} ({len(pred_df)} rows)")

            # Confusion matrices
            cm_raw, cm_norm = compute_confusion_matrices(y_part, y_pred, labels=labels_list)

            # Plot raw and normalized confusion matrices
            cm_raw_path = reports_figures_dir / f"confusion_matrix_{model_name}_{part_name}.png"
            plot_confusion_matrix(
                cm=cm_raw,
                class_names=DEFAULT_CLASS_NAMES,
                title=f"Confusion Matrix: {model_name.replace('_', ' ').title()} ({part_name.upper()})",
                output_path=cm_raw_path,
                fmt="d",
            )
            logger.info(f"Saved confusion matrix plot to {cm_raw_path}")

    # Combine metrics & export tables
    all_overall_df = pd.DataFrame(overall_results)
    cols_order = [
        "model", "partition", "accuracy", "balanced_accuracy",
        "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
        "macro_pr_auc", "macro_roc_auc", "ece",
        "inference_latency_ms_per_beat", "model_size_mb"
    ]
    all_overall_df = all_overall_df[cols_order]
    
    comp_csv_path = reports_tables_dir / "classical_model_comparison.csv"
    all_overall_df.to_csv(comp_csv_path, index=False)
    logger.info(f"\nSaved overall model comparison table to {comp_csv_path}")

    all_per_class_df = pd.concat(per_class_results, ignore_index=True)
    per_class_cols = ["model", "partition", "class", "precision", "recall", "f1", "support", "pr_auc"]
    all_per_class_df = all_per_class_df[per_class_cols]
    
    per_class_csv_path = reports_tables_dir / "classical_per_class_metrics.csv"
    all_per_class_df.to_csv(per_class_csv_path, index=False)
    logger.info(f"Saved per-class metrics table to {per_class_csv_path}")

    # Plot per-class F1 comparison figures
    for part_name in ["val", "test"]:
        f1_plot_path = reports_figures_dir / f"per_class_f1_comparison_{part_name}.png"
        plot_per_class_f1_comparison(all_per_class_df, part_name, f1_plot_path)
        logger.info(f"Saved per-class F1 comparison plot to {f1_plot_path}")
        
    # Also save main per_class_f1_comparison.png for test set
    main_f1_plot_path = reports_figures_dir / "per_class_f1_comparison.png"
    plot_per_class_f1_comparison(all_per_class_df, "test", main_f1_plot_path)

    # Export experiment config artifact
    exp_config_path = reports_tables_dir / "classical_experiment_config.csv"
    exp_config_data = []
    for m in enabled_models:
        exp_config_data.append({
            "model": m,
            "seed": seed,
            "n_features": X_train.shape[1],
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "parameters": str(models_config[m]),
        })
    pd.DataFrame(exp_config_data).to_csv(exp_config_path, index=False)
    logger.info(f"Saved experiment configuration to {exp_config_path}")

    logger.info("\n==================================================")
    logger.info("MILESTONE 5 CLASSICAL BASELINES SUMMARY")
    logger.info("==================================================")
    print("\nOVERALL METRICS (TEST SET):")
    test_summary = all_overall_df[all_overall_df["partition"] == "test"]
    print(test_summary[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_pr_auc", "ece", "inference_latency_ms_per_beat"]].to_string(index=False))


if __name__ == "__main__":
    main()
