"""
Milestone 12 — Continuous ECG Benchmark & Error Audit Script.

Evaluates the deployment-time Pan-Tompkins R-peak detector, continuous beat window segmentation,
and frozen ONNX INT8 classifier across real MIT-BIH database records.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

from ecg_arrhythmia.evaluation.continuous_eval import (
    evaluate_record_continuous_pipeline,
    evaluate_r_peak_detection,
)
from ecg_arrhythmia.deployment.continuous import ContinuousInferencePipeline
from ecg_arrhythmia.deployment.service import ONNXInferenceService
from ecg_arrhythmia.preprocessing.filtering import select_ecg_lead, filter_ecg
from ecg_arrhythmia.data.annotations import is_heartbeat
from ecg_arrhythmia.utils import get_logger, get_project_root

logger = get_logger("evaluate_continuous_pipeline")

# DS2 Test Records (22 records per de Chazal benchmark split)
TEST_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234"
]


def plot_rpeak_overlay(
    raw_signal: np.ndarray,
    detected_peaks: np.ndarray,
    annotated_peaks: np.ndarray,
    fs: float,
    start_sample: int,
    end_sample: int,
    output_path: Path,
    record_name: str,
):
    """Generates waveform overlay plot comparing detected R-peaks vs reference annotations."""
    sub_sig = raw_signal[start_sample:end_sample]
    time_axis = np.arange(start_sample, end_sample) / fs

    det_sub = [p for p in detected_peaks if start_sample <= p < end_sample]
    ann_sub = [a for a in annotated_peaks if start_sample <= a < end_sample]

    plt.figure(figsize=(12, 4.5), dpi=300)
    plt.plot(time_axis, sub_sig, label="Filtered ECG Signal (0.5–40 Hz)", color="#1f77b4", linewidth=1.2)

    if det_sub:
        plt.plot(
            np.array(det_sub) / fs,
            sub_sig[np.array(det_sub) - start_sample],
            "^",
            color="#2ca02c",
            markersize=9,
            label="Pan-Tompkins Detected R-Peaks",
            zorder=4,
        )

    if ann_sub:
        plt.plot(
            np.array(ann_sub) / fs,
            sub_sig[np.array(ann_sub) - start_sample],
            "x",
            color="#d62728",
            markersize=9,
            markeredgewidth=2,
            label="PhysioNet Reference Annotations",
            zorder=5,
        )

    plt.title(f"Continuous ECG R-Peak Detection Overlay — Record {record_name} (Sample Window {start_sample}–{end_sample})", fontsize=12, pad=10)
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("ECG Amplitude (mV)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_timing_error_histogram(all_timing_errors_ms: List[float], output_path: Path):
    """Generates histogram of timing errors (detected - annotated) in ms."""
    plt.figure(figsize=(8, 4.5), dpi=300)
    errs = np.array(all_timing_errors_ms)

    plt.hist(errs, bins=50, color="#2ca02c", edgecolor="#1b661b", alpha=0.8, density=True)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero Timing Error")
    plt.axvline(np.mean(errs), color="#d62728", linestyle="-.", linewidth=1.5, label=f"Mean Error: {np.mean(errs):.2f} ms")
    plt.axvline(np.median(errs), color="#ff7f0e", linestyle=":", linewidth=1.5, label=f"Median Error: {np.median(errs):.2f} ms")

    plt.title("R-Peak Detection Timing Error Distribution (Detected - Annotated)", fontsize=12, pad=10)
    plt.xlabel("Timing Error (ms)", fontsize=10)
    plt.ylabel("Probability Density", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_boundary_rejection_example(
    raw_signal: np.ndarray,
    unclassified_peaks: List[Dict[str, Any]],
    output_path: Path,
    record_name: str,
):
    """Generates visual figure showing boundary-clipped R-peak rejection at signal boundaries."""
    plt.figure(figsize=(10, 4), dpi=300)
    
    # Plot first 500 samples (boundary region)
    show_samples = min(500, len(raw_signal))
    t_axis = np.arange(show_samples)
    plt.plot(t_axis, raw_signal[:show_samples], color="#1f77b4", label="Filtered ECG Signal")

    # Mark 72-sample boundary line
    plt.axvline(72, color="#d62728", linestyle="--", linewidth=1.5, label="216-Sample Pre-Window Boundary (72 samples / 200 ms)")

    boundary_peaks = [u["r_peak_sample"] for u in unclassified_peaks if u["r_peak_sample"] < show_samples]
    if boundary_peaks:
        plt.plot(
            boundary_peaks,
            raw_signal[boundary_peaks],
            "o",
            color="#d62728",
            markersize=10,
            label="Boundary-Rejected R-Peaks",
        )

    plt.title(f"Continuous Pipeline Boundary-Clipped Peak Rejection — Record {record_name}", fontsize=11, pad=10)
    plt.xlabel("Sample Index", fontsize=10)
    plt.ylabel("ECG Amplitude (mV)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate continuous ECG processing pipeline on MIT-BIH records.")
    parser.add_argument("--records", nargs="+", default=TEST_RECORDS, help="Record IDs to evaluate.")
    parser.add_argument("--tolerance", type=float, default=0.150, help="Matching tolerance window in seconds (default 0.150s).")
    args = parser.parse_args()

    project_root = get_project_root()
    data_dir = project_root / "data" / "raw" / "mitdb"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures" / "continuous_eval"
    tables_dir = reports_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing ONNXInferenceService for continuous benchmark evaluation...")
    service = ONNXInferenceService()

    record_summaries = []
    all_timing_errors = []
    total_start_time = time.perf_counter()

    logger.info(f"Evaluating continuous pipeline across {len(args.records)} records (tolerance={args.tolerance*1000:.0f} ms)...")

    for rec_id in args.records:
        rec_id_str = str(rec_id)
        try:
            summary = evaluate_record_continuous_pipeline(
                record_name=rec_id_str,
                data_dir=data_dir,
                service=service,
                tolerance_sec=args.tolerance,
            )
            record_summaries.append(summary)
            all_timing_errors.extend(summary["timing_errors_ms"])

            logger.info(
                f"Record {rec_id_str:3s} | Annotated: {summary['num_annotated_beats']:4d} | "
                f"Detected: {summary['num_detected_peaks']:4d} | TP: {summary['detector_tp']:4d} | "
                f"FP: {summary['detector_fp']:3d} | FN: {summary['detector_fn']:3d} | "
                f"Se: {summary['detector_sensitivity']*100:.2f}% | PPV: {summary['detector_ppv']*100:.2f}% | "
                f"F1: {summary['detector_f1']*100:.2f}% | MAE: {summary['detector_mae_ms']:.2f} ms"
            )
        except Exception as e:
            logger.error(f"Failed to evaluate record {rec_id_str}: {e}")

    total_eval_time = time.perf_counter() - total_start_time

    # Aggregate Metrics
    agg_annotated = sum(s["num_annotated_beats"] for s in record_summaries)
    agg_detected = sum(s["num_detected_peaks"] for s in record_summaries)
    agg_processed = sum(s["num_processed_beats"] for s in record_summaries)
    agg_rejected = sum(s["num_boundary_rejected"] for s in record_summaries)

    agg_tp = sum(s["detector_tp"] for s in record_summaries)
    agg_fp = sum(s["detector_fp"] for s in record_summaries)
    agg_fn = sum(s["detector_fn"] for s in record_summaries)

    agg_se = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
    agg_ppv = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
    agg_f1 = (2.0 * agg_se * agg_ppv / (agg_se + agg_ppv)) if (agg_se + agg_ppv) > 0 else 0.0

    agg_mae_ms = float(np.mean(np.abs(all_timing_errors))) if all_timing_errors else 0.0
    agg_median_err_ms = float(np.median(all_timing_errors)) if all_timing_errors else 0.0

    # Aggregate predicted class counts
    agg_class_counts = {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}
    for s in record_summaries:
        for k, v in s["predicted_class_counts"].items():
            agg_class_counts[k] += v

    # Export Table 1: R-Peak Detection Metrics per Record & Aggregate
    rpeak_rows = []
    for s in record_summaries:
        rpeak_rows.append({
            "record_id": s["record_name"],
            "lead": s["lead_name"],
            "annotated_beats": s["num_annotated_beats"],
            "detected_peaks": s["num_detected_peaks"],
            "tp": s["detector_tp"],
            "fp": s["detector_fp"],
            "fn": s["detector_fn"],
            "sensitivity_recall": round(s["detector_sensitivity"], 4),
            "ppv_precision": round(s["detector_ppv"], 4),
            "f1_score": round(s["detector_f1"], 4),
            "mae_timing_error_ms": round(s["detector_mae_ms"], 2),
            "median_timing_error_ms": round(s["detector_median_err_ms"], 2),
        })

    rpeak_rows.append({
        "record_id": "AGGREGATE_DS2",
        "lead": "MLII_PRIMARY",
        "annotated_beats": agg_annotated,
        "detected_peaks": agg_detected,
        "tp": agg_tp,
        "fp": agg_fp,
        "fn": agg_fn,
        "sensitivity_recall": round(agg_se, 4),
        "ppv_precision": round(agg_ppv, 4),
        "f1_score": round(agg_f1, 4),
        "mae_timing_error_ms": round(agg_mae_ms, 2),
        "median_timing_error_ms": round(agg_median_err_ms, 2),
    })

    df_rpeak = pd.DataFrame(rpeak_rows)
    df_rpeak.to_csv(tables_dir / "continuous_rpeak_eval_summary.csv", index=False)
    logger.info(f"Saved R-peak metrics table to {tables_dir / 'continuous_rpeak_eval_summary.csv'}")

    # Export Table 2: Segmentation Audit per Record & Aggregate
    seg_rows = []
    for s in record_summaries:
        seg_rows.append({
            "record_id": s["record_name"],
            "detected_peaks": s["num_detected_peaks"],
            "valid_216_beats": s["num_processed_beats"],
            "boundary_rejected": s["num_boundary_rejected"],
            "segmentation_success_rate_pct": round(s["segmentation_success_rate_pct"], 2),
        })

    seg_rows.append({
        "record_id": "AGGREGATE_DS2",
        "detected_peaks": agg_detected,
        "valid_216_beats": agg_processed,
        "boundary_rejected": agg_rejected,
        "segmentation_success_rate_pct": round(agg_processed / agg_detected * 100.0, 2) if agg_detected > 0 else 0.0,
    })

    df_seg = pd.DataFrame(seg_rows)
    df_seg.to_csv(tables_dir / "continuous_segmentation_audit.csv", index=False)
    logger.info(f"Saved segmentation audit table to {tables_dir / 'continuous_segmentation_audit.csv'}")

    # Export Table 3: Predicted Class Distribution on Continuous Beats
    class_rows = []
    for s in record_summaries:
        counts = s["predicted_class_counts"]
        total_p = s["num_processed_beats"]
        class_rows.append({
            "record_id": s["record_name"],
            "total_classified": total_p,
            "pred_N": counts["N"],
            "pred_S": counts["S"],
            "pred_V": counts["V"],
            "pred_F": counts["F"],
            "pred_Q": counts["Q"],
            "pct_N": round(counts["N"] / total_p * 100.0, 2) if total_p > 0 else 0.0,
            "pct_S": round(counts["S"] / total_p * 100.0, 2) if total_p > 0 else 0.0,
            "pct_V": round(counts["V"] / total_p * 100.0, 2) if total_p > 0 else 0.0,
            "pct_F": round(counts["F"] / total_p * 100.0, 2) if total_p > 0 else 0.0,
            "pct_Q": round(counts["Q"] / total_p * 100.0, 2) if total_p > 0 else 0.0,
        })

    class_rows.append({
        "record_id": "AGGREGATE_DS2",
        "total_classified": agg_processed,
        "pred_N": agg_class_counts["N"],
        "pred_S": agg_class_counts["S"],
        "pred_V": agg_class_counts["V"],
        "pred_F": agg_class_counts["F"],
        "pred_Q": agg_class_counts["Q"],
        "pct_N": round(agg_class_counts["N"] / agg_processed * 100.0, 2) if agg_processed > 0 else 0.0,
        "pct_S": round(agg_class_counts["S"] / agg_processed * 100.0, 2) if agg_processed > 0 else 0.0,
        "pct_V": round(agg_class_counts["V"] / agg_processed * 100.0, 2) if agg_processed > 0 else 0.0,
        "pct_F": round(agg_class_counts["F"] / agg_processed * 100.0, 2) if agg_processed > 0 else 0.0,
        "pct_Q": round(agg_class_counts["Q"] / agg_processed * 100.0, 2) if agg_processed > 0 else 0.0,
    })

    df_class = pd.DataFrame(class_rows)
    df_class.to_csv(tables_dir / "continuous_class_distribution.csv", index=False)
    logger.info(f"Saved class distribution table to {tables_dir / 'continuous_class_distribution.csv'}")

    # Generate Figures
    # 1. Overlay figure for Record 100
    rec_100_path = str(data_dir / "100")
    if Path(rec_100_path + ".dat").exists():
        rec100 = wfdb.rdrecord(rec_100_path)
        ann100 = wfdb.rdann(rec_100_path, "atr")
        sig100, _, _ = select_ecg_lead(rec100, primary_lead="MLII")
        filt100 = filter_ecg(sig100, fs=360, low_hz=0.5, high_hz=40.0)

        pipeline = ContinuousInferencePipeline(service=service)
        res100 = pipeline.process_signal(sig100, sampling_rate=360.0)

        det_peaks_100 = np.array([p["r_peak_sample"] for p in res100["predictions"]] + [u["r_peak_sample"] for u in res100["unclassified_peaks"]])
        ann_peaks_100 = np.array([int(s) for s, m in zip(ann100.sample, ann100.symbol) if is_heartbeat(m)])

        plot_rpeak_overlay(
            raw_signal=filt100,
            detected_peaks=det_peaks_100,
            annotated_peaks=ann_peaks_100,
            fs=360.0,
            start_sample=0,
            end_sample=3600,  # First 10 seconds
            output_path=figures_dir / "r_peak_detection_waveform_overlay.png",
            record_name="100",
        )

        plot_boundary_rejection_example(
            raw_signal=filt100,
            unclassified_peaks=res100["unclassified_peaks"],
            output_path=figures_dir / "boundary_rejection_examples.png",
            record_name="100",
        )

    # 2. Timing Error Histogram
    if all_timing_errors:
        plot_timing_error_histogram(
            all_timing_errors_ms=all_timing_errors,
            output_path=figures_dir / "timing_error_histogram.png",
        )

    # Save summary JSON
    summary_json = {
        "evaluation_records": args.records,
        "matching_tolerance_sec": args.tolerance,
        "matching_tolerance_ms": args.tolerance * 1000.0,
        "num_records": len(record_summaries),
        "total_annotated_beats": agg_annotated,
        "total_detected_peaks": agg_detected,
        "total_processed_beats": agg_processed,
        "total_boundary_rejected": agg_rejected,
        "segmentation_success_rate_pct": round(agg_processed / agg_detected * 100.0, 4) if agg_detected > 0 else 0.0,
        "aggregate_detector_metrics": {
            "tp": agg_tp,
            "fp": agg_fp,
            "fn": agg_fn,
            "sensitivity_recall": round(agg_se, 4),
            "ppv_precision": round(agg_ppv, 4),
            "f1_score": round(agg_f1, 4),
            "mean_abs_timing_error_ms": round(agg_mae_ms, 2),
            "median_timing_error_ms": round(agg_median_err_ms, 2),
        },
        "predicted_class_distribution": agg_class_counts,
        "inference_throughput_beats_per_sec": round(agg_processed / total_eval_time, 2) if total_eval_time > 0 else 0.0,
        "total_eval_time_sec": round(total_eval_time, 2),
    }

    with open(reports_dir / "continuous_eval_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    logger.info(f"Saved evaluation summary JSON to {reports_dir / 'continuous_eval_summary.json'}")

    print("\n" + "=" * 80)
    print("MILESTONE 12 — CONTINUOUS ECG BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Records Evaluated            : {len(record_summaries)} records (DS2 test set)")
    print(f"Matching Window Tolerance    : ±{args.tolerance*1000:.0f} ms (ANSI/AAMI EC57 standard)")
    print(f"Total Annotated Beats        : {agg_annotated:,}")
    print(f"Total Detected Peaks         : {agg_detected:,}")
    print(f"True Positives (TP)          : {agg_tp:,}")
    print(f"False Positives (FP)         : {agg_fp:,}")
    print(f"False Negatives (FN)         : {agg_fn:,}")
    print(f"Detector Sensitivity (Recall): {agg_se * 100:.2f}%")
    print(f"Detector PPV (Precision)     : {agg_ppv * 100:.2f}%")
    print(f"Detector F1-Score            : {agg_f1 * 100:.2f}%")
    print(f"Mean Abs Timing Error (MAE)  : {agg_mae_ms:.2f} ms")
    print(f"Median Timing Error          : {agg_median_err_ms:.2f} ms")
    print(f"Successfully Segmented Beats : {agg_processed:,} ({agg_processed / agg_detected * 100.2 if agg_detected else 0:.2f}%)")
    print(f"Boundary-Rejected Peaks      : {agg_rejected:,}")
    print(f"Predicted Class Distribution : N={agg_class_counts['N']:,}, S={agg_class_counts['S']:,}, V={agg_class_counts['V']:,}, F={agg_class_counts['F']:,}, Q={agg_class_counts['Q']:,}")
    print(f"End-to-End Processing Time   : {total_eval_time:.2f} s ({agg_processed / total_eval_time:.1f} beats/sec)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
