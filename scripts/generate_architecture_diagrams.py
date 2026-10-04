#!/usr/bin/env python3
"""
Generate publication-quality architecture diagrams for:
1. Training & Evaluation Pipeline (Inter-Patient Split)
2. Production Inference Pipeline (FastAPI + ONNX INT8 + Streamlit)
3. Continuous ECG Processing Pipeline (Pan-Tompkins + Segmentation + Inference)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "reports" / "figures" / "architecture"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_training_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    boxes = [
        ("MIT-BIH Database\n(44 Records, Lead MLII)", "#e3f2fd", "#1565c0"),
        ("de Chazal Split\nDS1 (22 Train) / DS2 (22 Test)", "#e8eaf6", "#283593"),
        ("Signal Filtering\n(0.5–40Hz Bandpass)", "#e0f2f1", "#00695c"),
        ("Beat Segmentation\n(216-sample window)", "#e8f5e9", "#2e7d32"),
        ("z-score Normalization\n(Per-beat standardization)", "#fff8e1", "#f57f17"),
        ("Focal Loss 1D-CNN\n(Cost-Sensitive Loss)", "#fbe9e7", "#d84315"),
        ("Temperature Scaling\n(Calibrated T = 0.9854)", "#f3e5f5", "#6a1b9a"),
        ("DS2 Evaluation\n(49,712 Test Beats)", "#efebe9", "#4e342e"),
    ]

    x_start = 0.02
    box_width = 0.10
    box_height = 0.6
    y_center = 0.2

    for i, (text, bg_color, border_color) in enumerate(boxes):
        x = x_start + i * 0.12
        rect = mpatches.FancyBboxPatch(
            (x, y_center), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5, edgecolor=border_color, facecolor=bg_color
        )
        ax.add_patch(rect)
        ax.text(
            x + box_width / 2, y_center + box_height / 2, text,
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#212121",
            wrap=True
        )

        if i < len(boxes) - 1:
            ax.annotate(
                "", xy=(x + box_width + 0.02, y_center + box_height / 2),
                xytext=(x + box_width, y_center + box_height / 2),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#424242")
            )

    plt.title("Inter-Patient Model Training & Evaluation Pipeline (de Chazal Protocol)", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "training_evaluation_pipeline.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f" Saved: {output_path.relative_to(ROOT_DIR)}")


def create_production_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    components = [
        ("Client Request / Streamlit Dashboard", 0.05, 0.55, 0.25, 0.35, "#e3f2fd", "#1565c0", "Web Interface\n(Port 8501)"),
        ("FastAPI REST Endpoint\n/predict/beat | /predict/batch", 0.38, 0.55, 0.25, 0.35, "#e8f5e9", "#2e7d32", "API Gateway\n(Port 8000)"),
        ("Input Validation\n[N, 216] float32, NaN check", 0.38, 0.08, 0.25, 0.35, "#fff8e1", "#f57f17", "Validation Layer"),
        ("ONNX INT8 Session\nCPUExecutionProvider", 0.70, 0.55, 0.25, 0.35, "#fbe9e7", "#d84315", "Inference Engine\n(model_int8.onnx)"),
        ("Temperature Scaling\nSoftmax(logits / 0.9854)", 0.70, 0.08, 0.25, 0.35, "#f3e5f5", "#6a1b9a", "Calibration Layer"),
    ]

    for title, x, y, w, h, bg, border, label in components:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5, edgecolor=border, facecolor=bg
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=8, fontweight="bold", color="#212121")
        ax.text(x + w / 2, y + h * 0.25, f"({label})", ha="center", va="center", fontsize=7, fontstyle="italic", color="#616161")

    # Connectors
    ax.annotate("", xy=(0.38, 0.725), xytext=(0.30, 0.725), arrowprops=dict(arrowstyle="->", lw=1.5, color="#424242"))
    ax.annotate("", xy=(0.505, 0.43), xytext=(0.505, 0.55), arrowprops=dict(arrowstyle="->", lw=1.5, color="#424242"))
    ax.annotate("", xy=(0.70, 0.725), xytext=(0.63, 0.725), arrowprops=dict(arrowstyle="->", lw=1.5, color="#424242"))
    ax.annotate("", xy=(0.825, 0.43), xytext=(0.825, 0.55), arrowprops=dict(arrowstyle="->", lw=1.5, color="#424242"))

    plt.title("Production Inference Architecture (FastAPI + ONNX INT8 + Streamlit Container Stack)", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "production_inference_pipeline.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f" Saved: {output_path.relative_to(ROOT_DIR)}")


def create_continuous_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    boxes = [
        ("Raw Continuous ECG\n(360 Hz Single-Lead)", "#e8eaf6", "#283593"),
        ("Bandpass Filter\n(0.5–40 Hz 4th-order)", "#e0f2f1", "#00695c"),
        ("Pan-Tompkins Engine\n(Derivative/Square/Moving Avg)", "#fff8e1", "#f57f17"),
        ("R-Peak Detection\n(99.58% Sensitivity)", "#e8f5e9", "#2e7d32"),
        ("Beat Window Extraction\n(72 pre, 144 post)", "#f3e5f5", "#6a1b9a"),
        ("Boundary Check & z-score\n(99.96% Valid Beats)", "#fbe9e7", "#d84315"),
        ("Sub-Batch Chunking\n(500 beats / chunk)", "#e3f2fd", "#1565c0"),
        ("ONNX INT8 Classifier\n(17,908 beats/sec CPU)", "#efebe9", "#4e342e"),
    ]

    x_start = 0.02
    box_width = 0.10
    box_height = 0.6
    y_center = 0.2

    for i, (text, bg_color, border_color) in enumerate(boxes):
        x = x_start + i * 0.12
        rect = mpatches.FancyBboxPatch(
            (x, y_center), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5, edgecolor=border_color, facecolor=bg_color
        )
        ax.add_patch(rect)
        ax.text(
            x + box_width / 2, y_center + box_height / 2, text,
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#212121",
            wrap=True
        )

        if i < len(boxes) - 1:
            ax.annotate(
                "", xy=(x + box_width + 0.02, y_center + box_height / 2),
                xytext=(x + box_width, y_center + box_height / 2),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#424242")
            )

    plt.title("Continuous ECG Stream Processing & Inference Pipeline (Milestone 11-12 Architecture)", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "continuous_ecg_pipeline.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f" Saved: {output_path.relative_to(ROOT_DIR)}")


def main():
    print("Generating publication-quality architecture diagrams...")
    create_training_pipeline_diagram()
    create_production_pipeline_diagram()
    create_continuous_pipeline_diagram()
    print("All architecture diagrams created successfully.")


if __name__ == "__main__":
    main()
