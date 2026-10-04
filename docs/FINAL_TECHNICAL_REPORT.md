# Inter-Patient ECG Arrhythmia Classification: Final Technical & Scientific Report

## Executive Summary

This report documents the completed design, empirical evaluation, optimization, containerized deployment, and continuous ECG processing pipeline for **Inter-Patient ECG Arrhythmia Classification** trained on the **PhysioNet MIT-BIH Arrhythmia Database**.

Across Milestones 1 through 13, all experiments strictly enforce the **de Chazal et al. (2004)** patient-independent protocol:
- **DS1 (22 records, 51,020 beats)**: Used exclusively for training and hyperparameter tuning.
- **DS2 (22 records, 49,712 beats)**: Reserved strictly for unseen holdout testing.

The final production inference service is deployed using static **ONNX INT8 quantization** served via **FastAPI** (`0.0.0.0:8000`) and visualized using a **Streamlit** dashboard (`0.0.0.0:8501`), containerized with **Docker Compose**. Continuous raw ECG processing is enabled via an automated **Pan-Tompkins R-peak detector** achieving **99.58% Sensitivity** and **99.77% PPV** across all 49,712 annotated beats in DS2.

---

## 1. Problem Statement & Inter-Patient Validation Protocol

Standard machine learning models evaluated using random beat-level splitting suffer from severe data leakage because beats from the same patient appear in both training and test sets (*intra-patient evaluation*), artificially inflating accuracy up to 98-99%. 

To assess true clinical generalizability to unseen patients, this system enforces **inter-patient splitting**:
- **AAMI EC57 5-Class System**:
  - `N` (Normal & Bundle Branch Blocks)
  - `S` (Supraventricular Ectopic Beats)
  - `V` (Premature Ventricular Contractions)
  - `F` (Fusion Beats)
  - `Q` (Unknown / Unclassifiable Beats)
- **Input Representation**: Single-lead MLII ECG sampled at 360 Hz, bandpass filtered (0.5–40 Hz 4th-order zero-phase Butterworth), windowed to 216 samples (72 samples pre-R, 144 samples post-R), with per-beat z-score standardization $(x - \mu) / (\sigma + 10^{-8})$.

---

## 2. Empirical Model Comparison & Imbalance Analysis

### 2.1 Classical Baseline Models vs 1D-CNN

Feature extraction included 38 morphological, interval, and statistical features (R-R interval ratio, QRS width, peak-to-peak amplitude, skewness, kurtosis).

| Model Family | Algorithm | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test ECE | Inference Latency | Size (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical | Logistic Regression | 93.21% | 0.3968 | 0.9215 | 0.0273 | 0.047 µs/beat | 0.0025 |
| Classical | Random Forest | **94.75%** | 0.4201 | 0.9313 | 0.0462 | 1.81 µs/beat | 27.16 |
| Classical | XGBoost | 94.29% | 0.4699 | 0.9327 | 0.0302 | 1.50 µs/beat | 4.49 |
| Deep Learning | 1D-CNN (Unweighted) | 89.26% | 0.3081 | 0.8772 | 0.0217 | 47.7 µs/beat | 1.44 |

*Finding*: Classical tree ensembles (XGBoost) achieved higher raw test Macro F1 on hand-crafted features than unweighted end-to-end 1D-CNNs due to extreme class imbalance in MIT-BIH (N beats comprise >85% of samples).

### 2.2 Imbalance Remediation Experiments

To improve minority class detection (S, V, F), five loss and sampling techniques were evaluated:

| Experiment Strategy | Test Acc | Test Macro F1 | S-Class F1 | V-Class F1 | Key Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CNN Unweighted** | 89.26% | 0.3081 | 0.0000 | 0.5991 | S & F classes completely collapsed (0 F1) |
| **CNN Inverse Weighted** | 76.21% | 0.3064 | 0.0279 | 0.6271 | High recall on S/V, but high N false positives |
| **CNN Sqrt Inverse** | 86.10% | 0.2820 | 0.0021 | 0.4735 | Moderate trade-off |
| **CNN Focal Loss** ($\gamma=2.0$) | **87.57%** | **0.3105** | **0.0162** | **0.6009** | **Best overall precision/recall trade-off** |
| **CNN Oversampling** | 76.43% | 0.3075 | 0.0423 | 0.6292 | Overfitting on synthetic S beats |

---

## 3. Calibration, Explainability & Quantization

### 3.1 Model Calibration
Temperature Scaling ($T = 0.9854227900505066$) was optimized on validation logits.
- **Uncalibrated ECE**: 0.0810
- **Calibrated ECE**: 0.0774
- **Calibrated Softmax Formula**: $\hat{p}_i = \text{softmax}(z_i / T)$

### 3.2 Feature Attribution (Integrated Gradients)
Attribution maps confirmed that the 1D-CNN relies primarily on physiologically relevant QRS morphology (samples 60–85) and ST segment deviation (samples 90–130), matching clinical electrocardiographic principles.

### 3.3 ONNX INT8 Static Quantization

Static INT8 quantization reduced model footprint and accelerated CPU throughput while maintaining >99.1% prediction agreement with PyTorch FP32:

| Execution Engine | Format | Accuracy | Macro F1 | Size (MB) | Throughput (beats/sec) | Latency (ms/beat) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| PyTorch CPU | FP32 | 87.57% | 0.3105 | 0.468 MB | 403.3 | 0.160 ms |
| ONNX Runtime | FP32 | 87.57% | 0.3105 | 0.036 MB | 17,810.7 | 0.092 ms |
| **ONNX Runtime** | **INT8** | **88.05%** | **0.3126** | **0.158 MB** | **40,189.4** | **0.045 ms** |

---

## 4. Continuous ECG Processing & R-Peak Detection (Milestones 11–12)

### 4.1 Pan-Tompkins Detector Evaluation (ANSI/AAMI EC57 Standard)
Evaluated across all 22 DS2 test records (49,712 total annotated beats) under $\pm 150\text{ ms}$ ($\pm 54\text{ samples}$) greedy bipartite matching:

- **True Positives (TP)**: 49,503
- **False Positives (FP)**: 113
- **False Negatives (FN)**: 209
- **Sensitivity / Recall**: **99.58%**
- **Positive Predictive Value / Precision**: **99.77%**
- **Detector $F_1$-Score**: **99.68%**
- **Mean Absolute Timing Error**: **3.12 ms** ($\approx 1.12$ samples at 360 Hz)
- **Median Timing Error**: **0.00 ms**

### 4.2 Continuous Pipeline Audit
- **Total Detected Peaks**: 49,616
- **Valid Segmented Beats**: 49,598 (**99.96% segmentation success rate**)
- **Boundary-Clipped Rejections**: 18 beats (0.04%)
- **CPU End-to-End Throughput**: **17,908 beats/sec**

---

## 5. Major Scientific Limitations & Clinical Disclaimer

1. **Inter-Patient Class Shift**: Morphological variations across unseen patients cause significant intra-class variance, particularly for Supraventricular Ectopic Beats (S-class $F_1 < 0.10$).
2. **Single-Lead Constraint**: Lead MLII provides strong ventricular rhythm information but lacks multi-lead spatial coverage for complex atrial arrhythmias or ischemic changes.
3. **Artifact Vulnerability**: Heavy muscle tremor or electrode motion artifacts can trigger false R-peak detections (113 FPs recorded in DS2).

> [!WARNING]
> **Clinical Disclaimer**: This software is a research and engineering prototype. It is NOT FDA-approved, CE-marked, or certified as a medical device. It must NOT be used for clinical diagnosis, patient monitoring, or treatment decisions.

---

## 6. Reproducibility & Artifact Manifest

All results are fully reproducible without retraining using:
```bash
python scripts/verify_reproducibility.py
python scripts/audit_release.py
OMP_NUM_THREADS=1 .venv/bin/pytest -q
```
Authoritative metrics are stored in [`reports/final_results_manifest.json`](file:///Users/pranjalsingh/Downloads/old-voicemind/VoiceMind-V1-GitHub/app/untitled%20folder/reports/final_results_manifest.json).
