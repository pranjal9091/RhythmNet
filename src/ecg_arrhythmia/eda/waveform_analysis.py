"""Waveform morphology visual analysis, overlay summary plots, and non-ML exploratory statistics."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_arrhythmia.data.annotations import AAMI_CLASSES
from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def plot_waveforms_by_class(
    beats_dict: Dict[str, np.ndarray],
    dfs_dict: Dict[str, pd.DataFrame],
    out_path: Any,
    n_examples: int = 5,
    dpi: int = 200,
    fs: int = 360,
) -> None:
    """Plot multiple actual preprocessed beat waveforms for each AAMI class."""
    fig, axes = plt.subplots(len(AAMI_CLASSES), 1, figsize=(10, 12), sharex=True)

    # 216 samples = 600 ms at 360 Hz. Time axis in milliseconds centered at -200 ms to +400 ms
    pre_samples = 72
    time_ms = (np.arange(216) - pre_samples) * (1000.0 / fs)

    for idx, cls in enumerate(AAMI_CLASSES):
        ax = axes[idx]
        collected = []

        for split_name in ["train", "val", "test"]:
            if split_name in dfs_dict and split_name in beats_dict:
                df = dfs_dict[split_name]
                beats = beats_dict[split_name]

                indices = df[df["aami_class"] == cls].index.tolist()
                for i in indices:
                    collected.append((beats[i], df.loc[i, "record_id"], split_name))
                    if len(collected) >= n_examples:
                        break
            if len(collected) >= n_examples:
                break

        if collected:
            for wf, rid, split in collected:
                ax.plot(time_ms, wf, alpha=0.8, linewidth=1.2, label=f"Rec {rid} ({split})")
            ax.set_title(f"Class '{cls}' Beat Waveform Samples ({len(collected)} beats)")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_title(f"Class '{cls}' (No sample beats available)")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylabel("Amplitude (z-score)")

    axes[-1].set_xlabel("Time Relative to Annotation Peak (ms)")
    plt.suptitle(
        "Preprocessed ECG Heartbeat Waveform Morphology by AAMI Class\n"
        "(Normalized per-beat z-score, 216 samples @ 360 Hz)",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout()

    out_path = str(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved waveforms by class figure to: {out_path}")


def plot_waveform_overlays(
    beats_dict: Dict[str, np.ndarray],
    dfs_dict: Dict[str, pd.DataFrame],
    out_path: Any,
    n_overlay: int = 100,
    dpi: int = 200,
    fs: int = 360,
) -> None:
    """Plot overlaid sampled beats, class mean waveform, and ±1 std shaded region for each class."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    pre_samples = 72
    time_ms = (np.arange(216) - pre_samples) * (1000.0 / fs)

    for idx, cls in enumerate(AAMI_CLASSES):
        ax = axes_flat[idx]
        all_beats_list = []

        for split_name in ["train", "val", "test"]:
            if split_name in dfs_dict and split_name in beats_dict:
                df = dfs_dict[split_name]
                beats = beats_dict[split_name]
                indices = df[df["aami_class"] == cls].index.to_numpy()
                if len(indices) > 0:
                    all_beats_list.append(beats[indices])

        if all_beats_list:
            cls_beats = np.vstack(all_beats_list)
            n_total = len(cls_beats)

            # Sample up to n_overlay beats for background overlay
            sample_n = min(n_overlay, n_total)
            sampled_idx = np.random.choice(n_total, size=sample_n, replace=False)

            for i in sampled_idx:
                ax.plot(time_ms, cls_beats[i], color="gray", alpha=0.15, linewidth=0.8)

            mean_wf = np.mean(cls_beats, axis=0)
            std_wf = np.std(cls_beats, axis=0)

            ax.plot(time_ms, mean_wf, color="crimson", linewidth=2.0, label="Mean Waveform")
            ax.fill_between(
                time_ms,
                mean_wf - std_wf,
                mean_wf + std_wf,
                color="crimson",
                alpha=0.25,
                label="±1 Std Dev",
            )
            ax.set_title(f"Class '{cls}' (N={n_total} beats)")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_title(f"Class '{cls}' (0 beats)")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Normalized Amplitude")

    # Hide unused 6th subplot
    axes_flat[5].axis("off")

    plt.suptitle(
        "Class Waveform Morphology Overlay, Mean, and Standard Deviation Shading",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()

    out_path = str(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved class waveform overlays figure to: {out_path}")


def compute_morphology_statistics(
    beats_dict: Dict[str, np.ndarray], dfs_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Calculate non-ML exploratory waveform statistics aggregated by AAMI class."""
    stat_rows = []

    for split_name in ["train", "val", "test"]:
        if split_name not in dfs_dict or split_name not in beats_dict:
            continue

        df = dfs_dict[split_name]
        beats = beats_dict[split_name]

        if len(beats) == 0:
            continue

        # Per-beat amplitude statistics
        b_min = np.min(beats, axis=1)
        b_max = np.max(beats, axis=1)
        b_p2p = b_max - b_min
        b_rms = np.sqrt(np.mean(beats**2, axis=1))
        b_energy = np.sum(beats**2, axis=1)

        temp_df = pd.DataFrame(
            {
                "aami_class": df["aami_class"].values,
                "p2p": b_p2p,
                "rms": b_rms,
                "energy": b_energy,
                "min": b_min,
                "max": b_max,
            }
        )

        for cls in AAMI_CLASSES:
            cls_df = temp_df[temp_df["aami_class"] == cls]
            count = len(cls_df)

            if count == 0:
                continue

            stat_rows.append(
                {
                    "partition": split_name,
                    "aami_class": cls,
                    "count": count,
                    "p2p_mean": float(cls_df["p2p"].mean()),
                    "p2p_std": float(cls_df["p2p"].std()) if count > 1 else 0.0,
                    "p2p_median": float(cls_df["p2p"].median()),
                    "rms_mean": float(cls_df["rms"].mean()),
                    "rms_std": float(cls_df["rms"].std()) if count > 1 else 0.0,
                    "energy_mean": float(cls_df["energy"].mean()),
                    "min_mean": float(cls_df["min"].mean()),
                    "max_mean": float(cls_df["max"].mean()),
                }
            )

    return pd.DataFrame(stat_rows)
