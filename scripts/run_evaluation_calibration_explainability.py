"""
CLI Script to run Milestone 8 — Evaluation, Calibration & Explainability.

Usage:
    python scripts/run_evaluation_calibration_explainability.py \
        --config configs/evaluation_calibration_explainability.yaml
"""

from pathlib import Path
import argparse
import sys
import time
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.utils import get_logger, load_config, seed_everything, get_project_root
from ecg_arrhythmia.deep_learning import (
    ECGBeatDataset,
    ECG1DCNN,
    predict_deep_model,
    measure_deep_inference_latency,
)
from ecg_arrhythmia.calibration import (
    ModelWithTemperature,
    fit_temperature_scaling,
    save_temperature_artifact,
    compute_calibration_report,
    plot_reliability_diagram,
)
from ecg_arrhythmia.explainability import (
    compute_input_gradients,
    compute_integrated_gradients,
    GradCAM1D,
    select_explainability_examples,
    compute_aggregate_attributions,
    plot_beat_attribution,
    plot_aggregate_attribution,
)
from ecg_arrhythmia.models import DEFAULT_LABEL_MAPPING, DEFAULT_CLASS_NAMES

logger = get_logger("evaluation_calibration_explainability")


def load_model_checkpoint(model_name: str, project_root: Path, device: torch.device) -> Tuple[nn.Module, Path]:
    """Loads trained 1D CNN model checkpoint from M7 or M6 directory."""
    m7_path = project_root / "artifacts" / "models" / "deep_learning" / "m7" / f"{model_name}.pt"
    m6_path = project_root / "artifacts" / "models" / "deep_learning" / "cnn_best.pt"

    if m7_path.exists():
        ckpt_path = m7_path
    elif model_name == "cnn_unweighted" and m6_path.exists():
        ckpt_path = m6_path
    else:
        raise FileNotFoundError(f"Checkpoint for {model_name} not found at {m7_path} or {m6_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = ECG1DCNN(in_channels=1, num_classes=5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    return model, ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evaluation, Calibration & Explainability")
    parser.add_argument("--config", type=str, default="configs/evaluation_calibration_explainability.yaml")
    parser.add_argument("--model", type=str, default=None, help="Optionally filter by model name")
    parser.add_argument("--calibration-only", action="store_true", help="Run calibration only")
    parser.add_argument("--explainability-only", action="store_true", help="Run explainability only")
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

    # Load datasets
    data_dir = project_root / "data" / "processed"
    val_dataset = ECGBeatDataset(beats=data_dir / "val" / "beats.npy", metadata=data_dir / "val" / "metadata.csv")
    test_dataset = ECGBeatDataset(beats=data_dir / "test" / "beats.npy", metadata=data_dir / "test" / "metadata.csv")

    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    models_list = config.get("models", ["cnn_unweighted", "cnn_focal", "cnn_weighted_inverse", "cnn_weighted_sqrt_inverse", "cnn_oversampling"])
    if args.model:
        models_list = [m for m in models_list if m == args.model]

    # Directories
    artifacts_calib_dir = project_root / "artifacts" / "calibration"
    reports_tables_dir = project_root / "reports" / "tables"
    reports_fig_calib_dir = project_root / "reports" / "figures" / "calibration"
    reports_fig_exp_dir = project_root / "reports" / "figures" / "explainability"
    reports_tables_exp_dir = project_root / "reports" / "tables" / "explainability"

    artifacts_calib_dir.mkdir(parents=True, exist_ok=True)
    reports_tables_dir.mkdir(parents=True, exist_ok=True)
    reports_fig_calib_dir.mkdir(parents=True, exist_ok=True)
    reports_fig_exp_dir.mkdir(parents=True, exist_ok=True)
    reports_tables_exp_dir.mkdir(parents=True, exist_ok=True)

    calibration_comp_rows = []
    calibration_run_rows = []
    explainability_examples_rows = []
    explainability_summary_rows = []

    labels_list = [DEFAULT_LABEL_MAPPING[c] for c in DEFAULT_CLASS_NAMES]
    n_bins = config.get("calibration", {}).get("n_bins", 10)

    for model_name in models_list:
        logger.info(f"\n==================================================")
        logger.info(f"EVALUATING MODEL: {model_name}")
        logger.info(f"==================================================")

        model, ckpt_path = load_model_checkpoint(model_name, project_root, device)

        # PART A: CALIBRATION
        if not args.explainability_only:
            logger.info(f"Fitting temperature scaling strictly on Validation Set logits...")
            opt_temp, val_nll_before, val_nll_after = fit_temperature_scaling(model, val_loader, device)

            # Save temperature artifact
            temp_art_path = artifacts_calib_dir / f"{model_name}_temperature.json"
            save_temperature_artifact(
                model_name=model_name,
                temperature=opt_temp,
                val_nll_before=val_nll_before,
                val_nll_after=val_nll_after,
                output_path=temp_art_path,
                n_bins=n_bins,
                seed=seed,
            )

            model_temp = ModelWithTemperature(model, temperature=opt_temp)

            # Evaluate on Validation & Test before and after calibration
            for part_name, loader, dataset in [("val", val_loader, val_dataset), ("test", test_loader, test_dataset)]:
                # Uncalibrated probabilities
                y_pred_uncal, y_prob_uncal = predict_deep_model(model, loader, device)
                rep_uncal = compute_calibration_report(dataset.labels, y_prob_uncal, n_bins=n_bins, labels=labels_list)
                rep_uncal.update({
                    "model": model_name,
                    "calibration_method": "uncalibrated",
                    "temperature": 1.0,
                    "partition": part_name,
                })
                calibration_comp_rows.append(rep_uncal)

                # Temperature-scaled probabilities
                y_pred_cal, y_prob_cal = predict_deep_model(model_temp, loader, device)
                rep_cal = compute_calibration_report(dataset.labels, y_prob_cal, n_bins=n_bins, labels=labels_list)
                rep_cal.update({
                    "model": model_name,
                    "calibration_method": "temperature_scaling",
                    "temperature": opt_temp,
                    "partition": part_name,
                })
                calibration_comp_rows.append(rep_cal)

                if part_name == "test":
                    # Plot Reliability Diagrams
                    rel_fig_path = reports_fig_calib_dir / f"reliability_{model_name}.png"
                    plot_reliability_diagram(dataset.labels, y_prob_uncal, y_prob_cal, model_name, rel_fig_path, n_bins=n_bins)
                    logger.info(f"Saved reliability diagram to {rel_fig_path}")

            calibration_run_rows.append({
                "model": model_name,
                "temperature": opt_temp,
                "val_nll_before": val_nll_before,
                "val_nll_after": val_nll_after,
            })

        # PART B: EXPLAINABILITY
        if not args.calibration_only:
            logger.info(f"Running waveform explainability for {model_name}...")

            # 1. Benchmark explainability latencies
            sample_waveform = test_dataset[0]["waveform"]  # [1, 216]
            
            t0 = time.perf_counter()
            _ = predict_deep_model(model, DataLoader(test_dataset, batch_size=256), device)
            t_infer = (time.perf_counter() - t0) / len(test_dataset) * 1000.0

            t0 = time.perf_counter()
            _ = compute_input_gradients(model, sample_waveform, target_class=0, device=device)
            t_grad = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            _, _ = compute_integrated_gradients(model, sample_waveform, target_class=0, device=device, n_steps=50)
            t_ig = (time.perf_counter() - t0) * 1000.0

            gradcam_engine = GradCAM1D(model)
            t0 = time.perf_counter()
            _ = gradcam_engine.generate_heatmap(sample_waveform, target_class=0, device=device)
            t_gcam = (time.perf_counter() - t0) * 1000.0

            explainability_summary_rows.append({
                "model": model_name,
                "inference_ms_per_beat": t_infer,
                "input_gradient_ms_per_beat": t_grad,
                "integrated_gradients_ms_per_beat": t_ig,
                "gradcam_ms_per_beat": t_gcam,
            })

            # 2. Select deterministic representative beat examples
            test_preds_path = project_root / "reports" / "predictions" / "m7" / f"{model_name}_test.csv"
            if not test_preds_path.exists():
                test_preds_path = project_root / "reports" / "predictions" / f"{model_name}_test.csv"

            if test_preds_path.exists():
                pred_df = pd.read_csv(test_preds_path)
                selected_examples = select_explainability_examples(pred_df, seed=seed, max_examples=8)

                for ex in selected_examples:
                    idx = ex["dataset_index"]
                    w_tensor = test_dataset[idx]["waveform"]
                    w_np = w_tensor.numpy()[0]
                    target_cls_idx = DEFAULT_LABEL_MAPPING[ex["predicted_class"]]

                    # Calculate attributions
                    ig_grad = compute_input_gradients(model, w_tensor, target_class=target_cls_idx, device=device)
                    ig_attr, comp_err = compute_integrated_gradients(model, w_tensor, target_class=target_cls_idx, device=device, n_steps=50)
                    gcam_attr = gradcam_engine.generate_heatmap(w_tensor, target_class=target_cls_idx, device=device)

                    attributions_dict = {
                        "input_gradient": ig_grad,
                        "integrated_gradients": ig_attr,
                        "gradcam": gcam_attr,
                    }

                    # Plot individual beat attribution figure
                    fig_path = reports_fig_exp_dir / f"{model_name}_{ex['example_id']}.png"
                    plot_beat_attribution(w_np, attributions_dict, ex, fig_path)

                    # Export machine-readable attribution CSV
                    time_ms = (np.arange(216) - 72) / 360.0 * 1000.0
                    attr_df = pd.DataFrame({
                        "sample_index": np.arange(216),
                        "time_ms": time_ms,
                        "waveform_value": w_np,
                        "input_gradient": ig_grad,
                        "integrated_gradients": ig_attr,
                        "gradcam_importance": gcam_attr,
                    })
                    attr_df.to_csv(reports_tables_exp_dir / f"{model_name}_{ex['example_id']}_attributions.csv", index=False)

                    ex.update({
                        "model": model_name,
                        "split": "test",
                        "completeness_error": comp_err,
                    })
                    explainability_examples_rows.append(ex)

            # 3. Compute aggregate attributions for the model
            agg_dict = compute_aggregate_attributions(model, test_loader, device=device, max_examples_per_class=50, seed=seed)
            agg_fig_path = reports_fig_exp_dir / f"aggregate_attribution_{model_name}.png"
            plot_aggregate_attribution(agg_dict, agg_fig_path)
            logger.info(f"Saved aggregate attribution figure to {agg_fig_path}")

    # Combine tables
    if calibration_comp_rows:
        cal_comp_df = pd.DataFrame(calibration_comp_rows)
        cal_comp_df.to_csv(reports_tables_dir / "calibration_comparison.csv", index=False)
        pd.DataFrame(calibration_run_rows).to_csv(reports_tables_dir / "calibration_runs.csv", index=False)

    if explainability_examples_rows:
        pd.DataFrame(explainability_examples_rows).to_csv(reports_tables_dir / "explainability_examples.csv", index=False)
    if explainability_summary_rows:
        pd.DataFrame(explainability_summary_rows).to_csv(reports_tables_dir / "explainability_summary.csv", index=False)

    logger.info("\n==================================================")
    logger.info("MILESTONE 8 CALIBRATION & EXPLAINABILITY SUMMARY")
    logger.info("==================================================")
    if calibration_comp_rows:
        print("\nTEST SET CALIBRATION SUMMARY:")
        test_cal = cal_comp_df[cal_comp_df["partition"] == "test"]
        print(test_cal[["model", "calibration_method", "temperature", "nll", "ece", "brier", "accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
