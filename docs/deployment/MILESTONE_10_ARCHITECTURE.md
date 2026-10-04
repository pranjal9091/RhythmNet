# Milestone 10 — Production Web Deployment Architecture Specification

> **Clinical Disclaimer**: This system is a research prototype for ECG arrhythmia classification according to AAMI EC57 standard and is NOT certified for primary clinical diagnosis or standalone patient monitoring.

---

## 1. Current Model & Production Inference Contract

### Target Model
- **Deployed Model Artifact**: `artifacts/deployment/model_int8.onnx` (Static INT8 Quantized 1D CNN derived from `cnn_focal`).
- **Temperature Calibration Artifact**: `artifacts/calibration/cnn_focal_temperature.json` ($T = 0.9854227900505066$).
- **Execution Provider**: ONNX Runtime (`CPUExecutionProvider`).

### Exact ONNX IO Contract
- **Input Node Name**: `"input"`
- **Input Shape**: `[batch_size, 1, 216]` (Dynamic batch dimension, 1 channel, 216 time samples at 360 Hz).
- **Input Data Type**: `float32` (`numpy.float32`).
- **Output Node Name**: `"output"`
- **Output Shape**: `[batch_size, 5]` (Dynamic batch dimension, 5 raw unnormalized output logits).
- **Output Data Type**: `float32` (`numpy.float32`).

### Post-Processing & Calibration Contract
1. **Raw Logits**: $\mathbf{z} = \text{ONNX\_Inference}(\mathbf{x})$ of shape `[batch_size, 5]`.
2. **Temperature Rescaling**: $\mathbf{z}_{\text{scaled}} = \mathbf{z} / T$, where $T = 0.9854227900505066$.
3. **Calibrated Probabilities**: $\mathbf{p} = \text{softmax}(\mathbf{z}_{\text{scaled}})_k = \frac{\exp(z_{\text{scaled}, k})}{\sum_{j=0}^4 \exp(z_{\text{scaled}, j})}$.
4. **Predicted Class**: $\hat{y} = \arg\max_{k \in \{0..4\}} p_k$.
5. **Confidence Score**: $c = \max_{k \in \{0..4\}} p_k$.

---

## 2. Existing Preprocessing Pipeline

The deployment pipeline MUST execute the exact signal transformation sequence used during model training:

```text
Raw 1D ECG Signal (fs = 360 Hz)
               │
               ▼
1. Sampling Frequency Validation (360 Hz)
               │
               ▼
2. Zero-Phase Bandpass Filter (0.5 – 40 Hz, Order 4 Butterworth via scipy.signal.filtfilt)
               │
               ▼
3. Peak Detection / Beat Window Segmentation (216 samples = 600 ms window)
   [72 samples (200 ms) pre-R-peak, 144 samples (400 ms) post-R-peak]
               │
               ▼
4. Per-Beat Z-Score Normalization: x_norm = (x - μ) / (σ + 1e-8)
               │
               ▼
5. Reshape to Tensor [batch_size, 1, 216] (float32)
```

---

## 3. Required Production Input Formats

The production service will accept two primary input payloads:

### A. Raw Single/Multi-Beat Waveform Array
- Direct array of raw or pre-filtered amplitudes length 216.
- Application of per-beat z-score normalization: $(\mathbf{x} - \mu) / (\sigma + 10^{-8})$.

### B. Continuous Continuous 1D ECG Signal Stream
- Continuous raw ECG signal array at $360\text{ Hz}$.
- Application of $0.5\text{--}40\text{ Hz}$ zero-phase Butterworth bandpass filter.
- Automated R-peak detection (e.g. Scipy peak finding / Pan-Tompkins algorithm).
- Extraction of 216-sample beat windows $[-200\text{ ms}, +400\text{ ms}]$ ($[-72, +144]$ samples around peak).
- Per-beat z-score normalization.

---

## 4. Output Format (API & App Response)

The API JSON output payload will follow a strict schema:

```json
{
  "status": "success",
  "model_version": "cnn_focal_int8_v1",
  "num_beats": 1,
  "predictions": [
    {
      "beat_index": 0,
      "r_peak_sample": 72,
      "predicted_class": "V",
      "class_name": "Ventricular Ectopic Beat (VEB)",
      "confidence": 0.9864,
      "calibrated_probabilities": {
        "N": 0.0051,
        "S": 0.0032,
        "V": 0.9864,
        "F": 0.0048,
        "Q": 0.0005
      }
    }
  ]
}
```

---

## 5. AAMI 5-Class Label Mapping

The model outputs 5 logits corresponding strictly to integer indices 0 through 4:

| Class Index | AAMI Code | Full Category Name | Included MIT-BIH Symbols |
| :--- | :--- | :--- | :--- |
| **0** | **N** | Non-Ectopic Beat (Normal / Bundle Branch) | `N`, `L`, `R`, `e`, `j`, `B` |
| **1** | **S** | Supraventricular Ectopic Beat (SVEB) | `A`, `a`, `J`, `S` |
| **2** | **V** | Ventricular Ectopic Beat (VEB) | `V`, `E` |
| **3** | **F** | Fusion Beat (Ventricular + Normal) | `F` |
| **4** | **Q** | Unknown / Paced / Unclassifiable | `/`, `f`, `Q`, `?` |

---

## 6. Hardened FastAPI Architecture (Milestone 10B Implementation)

- **Framework**: `FastAPI` + `Uvicorn` with async lifespan context manager initializing `ONNXInferenceService`.
- **Inference Engine**: Reusable `ONNXInferenceService` (`src/ecg_arrhythmia/deployment/service.py`) with vectorized INT8 ONNX Runtime session and temperature scaling ($T = 0.9854227900505066$).
- **Endpoints**:
  - `GET /health`: Returns service metadata, loaded model name (`cnn_focal`), ONNX runtime info, and class mapping.
  - `GET /ready`: Readiness probe returning `200 OK` when ONNX session is initialized and ready, or `503 Service Unavailable` if unready.
  - `POST /predict/beat`: Predict single pre-segmented 216-sample beat waveform with strict schema validation ($N=216$, non-NaN/Inf float values).
  - `POST /predict/batch`: Vectorized batch prediction accepting $1 \le N \le 1000$ 216-sample beat arrays, applying vectorized z-score normalization and temperature scaling.
- **Observability & Logging**:
  - HTTP middleware (`log_requests_middleware`) recording HTTP method, request path, status code, and latency in ms per request.
  - Service logging recording batch size and inference execution time.
  - **Privacy Guarantee**: Raw ECG sample values, array floats, and patient identifiers are strictly excluded from logs.

---

## 7. Streamlit Dashboard Architecture (Step 3A Implementation)

- **Entry Point**: `src/ecg_arrhythmia/deployment/dashboard.py`
- **Launch Command**: `streamlit run src/ecg_arrhythmia/deployment/dashboard.py`
- **Supported Input Modes**:
  1. **MODE A — Test Dataset**: Select beat sample from preprocessed MIT-BIH test dataset (`data/processed/test/beats.npy` and `metadata.csv`). Supports ground-truth label display, AAMI class filtering, beat index selection, and Match/Mismatch evaluation.
  2. **MODE B — Manual Beat Input**: Upload CSV file or enter raw numeric text containing exactly 216 float samples. Validated via `validate_beat_array()` and `parse_csv_beat_input()`.
- **Inference Integration**:
  - Consumes `ONNXInferenceService` directly from `src/ecg_arrhythmia/deployment/service.py`.
  - Service instance is cached per process using `@st.cache_resource`.
  - Reuses central preprocessing (per-beat z-score), temperature scaling ($T = 0.9854227900505066$), and ONNX INT8 session without code duplication.
- **Visualization**:
  - Interactive Plotly line chart (`plotly.graph_objects.Figure`).
  - X-axis: Sample Index (0 to 215 @ 360 Hz).
  - Y-axis: Normalized ECG Amplitude.
  - Includes zero reference dashed line and clean grid styling. No fabricated R-peaks.
- **Error Handling**:
  - Gracefully captures missing model/dataset files, invalid input lengths ($\neq 216$), non-numeric values, NaN/Inf, and ONNX Runtime errors.
  - Displays user-friendly error banners via `st.error()` without exposing raw Python stack traces.
- **Explainability Limitation**:
  - Production inference uses frozen INT8 ONNX model.
  - Dedicated Explainability placeholder displayed: *"Gradient-based attribution is planned as a separate explainability pipeline. Production inference currently uses the frozen INT8 ONNX model."*
  - No fake heatmaps or fake Grad-CAM attributions generated.

---

## 8. Docker Architecture (Step 4 Implementation)

- **Base Image**: `python:3.12-slim`.
- **FastAPI Container (`Dockerfile.api`)**:
  - Exposes port `8000`.
  - Runs as non-root user `appuser` (UID `10001`).
  - Includes `artifacts/deployment/model_int8.onnx`, `artifacts/calibration/cnn_focal_temperature.json`, `configs/`, `README.md`, `pyproject.toml`, and `src/`.
  - Healthcheck: `curl -f http://localhost:8000/ready || exit 1` (interval 10s, timeout 5s, retries 3, start period 10s).
- **Streamlit Dashboard Container (`Dockerfile.dashboard`)**:
  - Exposes port `8501`.
  - Runs as non-root user `appuser` (UID `10001`).
  - Includes `data/processed/test/` for Mode A dataset inspection and test sample selection.
  - Configured with `API_URL=http://api:8000` via environment variable to query the containerized FastAPI inference service over the internal network.
  - Healthcheck: `curl -f http://localhost:8501/_stcore/health || exit 1`.
- **Docker Compose (`docker-compose.yml`)**:
  - `api` service mapping port `8000:8000`.
  - `dashboard` service mapping port `8501:8501`.
  - Service-to-service networking on internal bridge network `ecg_network`.
  - Health-aware startup dependency: `dashboard` service waits for `api` container condition `service_healthy`.
- **Docker Ignore (`.dockerignore`)**:
  - Excludes virtual environments (`.venv`), `.git`, `.pytest_cache`, raw records (`data/raw/`), large checkpoints (`artifacts/models/`), and unneeded temporary build files.

---

## 9. Error-Handling Requirements

- **Signal Length Validation**: Reject waveforms $< 216$ samples or continuous signals with invalid sampling rates.
- **NaN / Inf Filtering**: Detect and raise HTTP 422 errors if input array contains NaN or Infinite amplitude values.
- **Flatline Signal Handling**: Prevent division-by-zero errors in z-score normalization via $\epsilon = 10^{-8}$.
- **ONNX Session Resilience**: Catch ONNX Runtime exceptions and return structured JSON error messages with HTTP 500 status.

---

## 10. Testing Strategy & Deployment QA (Step 5 Implementation)

- **Automated Deployment QA Suite**: `tests/test_deployment_qa.py` (run via `OMP_NUM_THREADS=1 pytest tests/test_deployment_qa.py -v`).
- **Live Docker API E2E Testing**: Validates running Docker container REST endpoints (`GET /health`, `GET /ready`, `POST /predict/beat`, `POST /predict/batch`) over HTTP without mocking.
- **Numerical Prediction Consistency**: Guarantees identical class predictions and probability matching within $< 10^{-5}$ tolerance across local `ONNXInferenceService`, Docker `/predict/beat`, and Docker `/predict/batch`.
- **Security & Hygiene Audit**: Verifies non-root container user (`appuser`), `.dockerignore` exclusions (`.venv`, `.git`, `data/raw/`, `artifacts/models/`), no hardcoded secrets, and restricted port exposure (8000, 8501).
- **Full Test Suite**: 121 total unit and integration tests passing (`OMP_NUM_THREADS=1 pytest -q`).

---

## 11. Security & Safety Considerations

- **Input Sanitization**: Enforce payload size limits (e.g. max 10 MB per signal upload) to prevent Denial of Service (DoS) memory exhaustion.
- **CORS Configuration**: Restrict FastAPI Cross-Origin Resource Sharing (CORS) to specified domains in production settings.
- **Non-Clinical Disclaimer**: Include mandatory legal/clinical disclaimers in all API responses and Streamlit UI headers.

---

## 12. Performance Considerations

- **Low-Latency CPU Execution**: ONNX INT8 model runs in $\approx 0.0445\text{ ms/beat}$ ($\approx 40,000\text{ beats/sec}$), ensuring near-instantaneous processing for full 30-minute Holter records ($2,000\text{--}3,000$ beats) in $< 100\text{ ms}$.
- **Memory Footprint**: Total memory allocation per container $< 250\text{ MB}$.

---

## 13. Observability & Health-Check Requirements

- **Health Endpoint (`GET /health`)**: Returns JSON metadata:
  ```json
  {
    "status": "healthy",
    "model_name": "cnn_focal",
    "format": "ONNX_INT8",
    "temperature": 0.9854227900505066,
    "uptime_seconds": 3600
  }
  ```
- **Logging**: Structured JSON logging capturing request latency, beat counts, and prediction distribution without logging raw patient data.

---

## 14. Continuous ECG Benchmark & R-Peak Evaluation (Milestone 12 Implementation)

- **Matching Protocol**: Evaluated using greedy bipartite matching under ANSI/AAMI EC57 standard tolerance window ($\pm 150\text{ ms} = \pm 54\text{ samples}$ at $360\text{ Hz}$).
- **Evaluated Dataset**: All 22 de Chazal DS2 benchmark test records ($49,712$ total annotated beats).
- **R-Peak Detection Metrics**:
  - Sensitivity / Recall ($\text{Se}$): **$99.58\%$** ($49,503$ True Positives, $209$ False Negatives).
  - Positive Predictive Value / Precision ($\text{PPV}$): **$99.77\%$** ($113$ False Positives).
  - $F_1$-Score: **$99.68\%$**.
  - Mean Absolute Timing Error ($\text{MAE}$): **$3.12\text{ ms}$** ($\approx 1.12$ samples).
  - Median Timing Error: **$0.00\text{ ms}$**.
- **Segmentation Audit**:
  - Total Detected R-Peaks: $49,616$.
  - Valid 216-Sample Segmented Beats: $49,598$ (**$99.96\%$ segmentation success rate**).
  - Boundary-Clipped Rejected Peaks: $18$ ($0.04\%$).
- **End-to-End ONNX INT8 Inference**:
  - Processed $49,598$ continuous beats in $2.77\text{ seconds}$.
  - CPU Throughput: **$17,908\text{ beats/sec}$**.
  - Predicted Class Distribution: $N=43,425$ ($87.55\%$), $S=417$ ($0.84\%$), $V=5,596$ ($11.28\%$), $F=160$ ($0.32\%$), $Q=0$.
- **Generated Artifacts**:
  - Summary CSV Tables: `reports/tables/continuous_rpeak_eval_summary.csv`, `continuous_segmentation_audit.csv`, `continuous_class_distribution.csv`.
  - Figures: `reports/figures/continuous_eval/r_peak_detection_waveform_overlay.png`, `timing_error_histogram.png`, `boundary_rejection_examples.png`.
  - Detailed Metrics JSON: `reports/continuous_eval_summary.json`.

---

## DO NOT CHANGE

The following components and configurations are strictly **FROZEN** and MUST NOT be modified during Milestone 10 implementation:

1. **Frozen Patient Split**: The patient-independent de Chazal DS1/DS2 record split (Train: 17 records, Val: 5 records, Test: 22 records) remains untouched.
2. **Trained Model Artifacts**: `artifacts/deployment/model_int8.onnx`, `artifacts/deployment/model_fp32.onnx`, and `artifacts/calibration/cnn_focal_temperature.json` are frozen.
3. **Model Architecture**: The 1D CNN architecture (kernel sizes 7, 5, 3; 121,765 parameters) must not be altered or retrained.
4. **Training Data**: No new training, retraining, fine-tuning, or calibration parameter refitting may take place.
5. **Preprocessing Semantics**: Filtering cutoffs ($0.5\text{--}40\text{ Hz}$ 4th-order zero-phase Butterworth filter at $fs = 360\text{ Hz}$), beat window length (216 samples, pre: 72, post: 144), and per-beat z-score normalization formula must remain identical.
6. **AAMI 5-Class Label Mapping**: Class ordering `{"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}` must remain fixed.

---
