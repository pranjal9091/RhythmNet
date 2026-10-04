"""
CLI Script for Milestone 9 — Model Selection, ONNX Export & INT8 Quantization.

Usage:
    python scripts/run_onnx_export_quantization.py --config configs/deployment.yaml
"""

from pathlib import Path
import argparse
import sys
import time
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.utils import get_logger, load_config, seed_everything, get_project_root
from ecg_arrhythmia.deep_learning import ECGBeatDataset, ECG1DCNN
from ecg_arrhythmia.deployment import (
    export_pytorch_to_onnx,
    verify_onnx_numerical_equivalence,
    quantize_onnx_model_int8,
    evaluate_deployment_model,
    benchmark_deployment_performance,
)

logger = get_logger("run_deployment")


def parse_args():
    parser = argparse.ArgumentParser(description="Milestone 9 — ONNX Export & INT8 Quantization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deployment.yaml",
        help="Path to deployment YAML config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model candidate (e.g. cnn_focal, cnn_unweighted)",
    )
    return parser.parse_args()


def load_model_checkpoint(model_name: str, project_root: Path, device: torch.device) -> Tuple[nn.Module, Path]:
    search_paths = [
        project_root / "artifacts" / "models" / "deep_learning" / "m7" / f"{model_name}.pt",
        project_root / "artifacts" / "models" / "deep_learning" / f"{model_name}_best.pt",
        project_root / "artifacts" / "models" / "deep_learning" / f"{model_name}.pt",
        project_root / "models" / model_name / f"{model_name}_best.pt",
        project_root / "models" / model_name / "best_model.pt",
    ]

    ckpt_path = None
    for path in search_paths:
        if path.exists():
            ckpt_path = path
            break

    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint for {model_name} not found in search paths: {search_paths}")

    model = ECG1DCNN(in_channels=1, num_classes=5)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    logger.info(f"Loaded checkpoint for {model_name} from {ckpt_path}")
    return model, ckpt_path


def main():
    args = parse_args()
    project_root = get_project_root()
    config_path = project_root / args.config

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    seed = config.get("seed", 42)
    seed_everything(seed)

    selected_model_name = args.model or config.get("selected_model", "cnn_focal")
    device = torch.device("cpu")  # ONNX Runtime CPU benchmarking

    logger.info(f"Starting Milestone 9 deployment pipeline for model: {selected_model_name}")

    # Output directories
    art_deploy_dir = project_root / "artifacts" / "deployment"
    rep_tables_dir = project_root / "reports" / "tables"
    art_deploy_dir.mkdir(parents=True, exist_ok=True)
    rep_tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    data_dir = project_root / "data" / "processed"
    train_dataset = ECGBeatDataset(data_dir / "train" / "beats.npy", data_dir / "train" / "metadata.csv")
    val_dataset = ECGBeatDataset(data_dir / "val" / "beats.npy", data_dir / "val" / "metadata.csv")
    test_dataset = ECGBeatDataset(data_dir / "test" / "beats.npy", data_dir / "test" / "metadata.csv")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 2. Load PyTorch Model
    pytorch_model, ckpt_path = load_model_checkpoint(selected_model_name, project_root, device)

    # 3. Export to FP32 ONNX
    fp32_onnx_path = art_deploy_dir / "model_fp32.onnx"
    opset_version = config.get("export", {}).get("opset_version", 17)
    export_pytorch_to_onnx(
        model=pytorch_model,
        output_path=fp32_onnx_path,
        opset_version=opset_version,
        device=device,
    )

    # 4. Numerical Equivalence Verification
    logger.info("Verifying FP32 ONNX numerical equivalence against PyTorch on Validation Set...")
    equiv_metrics = verify_onnx_numerical_equivalence(
        pytorch_model=pytorch_model,
        onnx_path=fp32_onnx_path,
        val_loader=val_loader,
        device=device,
        max_samples=1000,
    )

    onnx_val_df = pd.DataFrame([{
        "model": selected_model_name,
        "opset_version": opset_version,
        "max_abs_difference": equiv_metrics["max_abs_difference"],
        "mean_abs_difference": equiv_metrics["mean_abs_difference"],
        "max_rel_difference": equiv_metrics["max_rel_difference"],
        "samples_verified": equiv_metrics["samples_verified"],
        "validation_passed": equiv_metrics["max_abs_difference"] < 1e-4,
    }])
    onnx_val_csv = rep_tables_dir / "onnx_validation.csv"
    onnx_val_df.to_csv(onnx_val_csv, index=False)
    logger.info(f"Saved ONNX validation metrics to {onnx_val_csv}")

    # 5. INT8 Static Quantization using TRAIN calibration data ONLY
    int8_onnx_path = art_deploy_dir / "model_int8.onnx"
    n_calib = config.get("quantization", {}).get("n_calibration_samples", 500)
    logger.info(f"Applying INT8 Static Quantization using {n_calib} TRAIN samples ONLY...")

    quantize_onnx_model_int8(
        input_fp32_onnx=fp32_onnx_path,
        output_int8_onnx=int8_onnx_path,
        train_loader=train_loader,
        n_calibration_samples=n_calib,
    )

    quant_cfg_df = pd.DataFrame([{
        "model": selected_model_name,
        "quantization_method": "static_qdq",
        "calibration_split": "train",
        "n_calibration_samples": n_calib,
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "calibration_dataset_size": len(train_dataset),
        "test_labels_accessed": False,
    }])
    quant_cfg_csv = rep_tables_dir / "quantization_config.csv"
    quant_cfg_df.to_csv(quant_cfg_csv, index=False)
    logger.info(f"Saved quantization config to {quant_cfg_csv}")

    # 6. Accuracy Validation on Validation & Test Sets
    logger.info("Running accuracy validation across PyTorch FP32, ONNX FP32, and ONNX INT8...")
    eval_targets = [
        ("PyTorch FP32", pytorch_model, False),
        ("ONNX FP32", fp32_onnx_path, True),
        ("ONNX INT8", int8_onnx_path, True),
    ]

    deploy_comp_rows = []
    
    # Store PyTorch predictions for agreement check
    pt_val_preds = None
    pt_test_preds = None

    for model_label, model_obj, is_onnx in eval_targets:
        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            res = evaluate_deployment_model(model_obj, loader, device=device, is_onnx=is_onnx)
            
            if model_label == "PyTorch FP32":
                if split_name == "val":
                    pt_val_preds = res["y_pred"]
                else:
                    pt_test_preds = res["y_pred"]
                agreement_pct = 100.0
            else:
                ref_preds = pt_val_preds if split_name == "val" else pt_test_preds
                agreement_pct = float(np.mean(res["y_pred"] == ref_preds) * 100.0)

            row = {
                "selected_model": selected_model_name,
                "model_format": model_label,
                "split": split_name,
                "accuracy": res["accuracy"],
                "balanced_accuracy": res["balanced_accuracy"],
                "macro_f1": res["macro_f1"],
                "weighted_f1": res["weighted_f1"],
                "N_f1": res["per_class_f1"]["N"],
                "S_f1": res["per_class_f1"]["S"],
                "V_f1": res["per_class_f1"]["V"],
                "F_f1": res["per_class_f1"]["F"],
                "Q_f1": res["per_class_f1"]["Q"],
                "prediction_agreement_pct": agreement_pct,
            }
            deploy_comp_rows.append(row)

    # 7. Performance Benchmarking
    logger.info("Running latency and throughput benchmarks...")
    sample_wave = val_dataset[0]["waveform"]  # [1, 216]
    
    batch_waves = torch.stack([val_dataset[i]["waveform"] for i in range(256)])  # [256, 1, 216]

    perf_metrics = benchmark_deployment_performance(
        pytorch_model=pytorch_model,
        onnx_fp32_path=fp32_onnx_path,
        onnx_int8_path=int8_onnx_path,
        sample_waveform=sample_wave,
        batch_waveform=batch_waves,
        device=device,
        n_warmup=config.get("benchmark", {}).get("n_warmup", 20),
        n_runs=config.get("benchmark", {}).get("n_runs", 200),
    )

    # Attach performance metrics to rows
    for row in deploy_comp_rows:
        label = row["model_format"]
        if label in perf_metrics:
            row.update({
                "single_beat_ms": perf_metrics[label]["single_beat_ms"],
                "batch_ms": perf_metrics[label]["batch_ms"],
                "throughput_beats_per_sec": perf_metrics[label]["throughput"],
                "model_size_mb": perf_metrics[label]["size_mb"],
                "compression_ratio": perf_metrics[label]["compression_ratio"],
            })

    deploy_comp_df = pd.DataFrame(deploy_comp_rows)
    deploy_comp_csv = rep_tables_dir / "deployment_comparison.csv"
    deploy_comp_df.to_csv(deploy_comp_csv, index=False)
    logger.info(f"Saved deployment comparison table to {deploy_comp_csv}")

    # Summary Output
    logger.info(f"\n==================================================")
    logger.info(f"MILESTONE 9 DEPLOYMENT SUMMARY ({selected_model_name})")
    logger.info(f"==================================================")
    print("\nTEST SET DEPLOYMENT COMPARISON:")
    test_df = deploy_comp_df[deploy_comp_df["split"] == "test"]
    print(test_df[[
        "model_format", "accuracy", "balanced_accuracy", "macro_f1",
        "weighted_f1", "prediction_agreement_pct", "single_beat_ms",
        "throughput_beats_per_sec", "model_size_mb", "compression_ratio"
    ]].to_string(index=False))

    logger.info("\nMilestone 9 deployment pipeline completed successfully!")


if __name__ == "__main__":
    main()
