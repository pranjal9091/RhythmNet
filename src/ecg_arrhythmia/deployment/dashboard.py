"""
Streamlit ECG Arrhythmia Analysis Dashboard.

Provides an interactive research and deployment web interface for inspecting
individual 216-sample ECG beats, evaluating ONNX INT8 model predictions, and
analyzing temperature-scaled class probabilities.
"""

from typing import Any, Tuple, Optional, Dict, List, Union
from pathlib import Path
import os
import io
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ecg_arrhythmia.deployment.service import (
    ONNXInferenceService,
    AAMI_CLASS_MAPPING,
    AAMI_CLASS_NAMES_FULL,
)
from ecg_arrhythmia.utils import get_logger, get_project_root

logger = get_logger("dashboard")

# Disclaimer banner text
DISCLAIMER_TEXT = (
    "Research prototype — not for clinical diagnosis or primary patient monitoring."
)

EXPLAINABILITY_TEXT = (
    "Gradient-based attribution is planned as a separate explainability pipeline. "
    "Production inference currently uses the frozen INT8 ONNX model."
)

# Model metadata constants for display
MODEL_METADATA = {
    "Model": "cnn_focal",
    "Runtime": "ONNX Runtime (CPUExecutionProvider)",
    "Quantization": "INT8 static QDQ",
    "Input Shape": "[1, 1, 216]",
    "Sampling Frequency": "360 Hz",
    "Preprocessing": "0.5–40 Hz Butterworth bandpass + Per-beat z-score",
    "Temperature Factor": "0.9854227900505066",
    "Classes": "N / S / V / F / Q",
}


# ==============================================================================
# PURE HELPER FUNCTIONS (Testable independently of Streamlit UI runtime)
# ==============================================================================

def get_api_base_url() -> str:
    """Returns configured REST API base URL from environment variables, or empty string for in-process ONNX."""
    return os.getenv("ECG_API_URL", os.getenv("API_URL", "")).strip()


def validate_beat_array(signal: Any) -> Tuple[bool, str]:
    """
    Validates whether the input signal can be cast to a 1D float array of 216 samples.
    
    Parameters
    ----------
    signal : Any
        Input signal (list, numpy array, etc.)
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if signal is None:
        return False, "No signal data provided."

    try:
        arr = np.asarray(signal, dtype=np.float64)
    except Exception as e:
        return False, f"Failed to convert signal to numeric array: {e}"

    if arr.ndim != 1:
        return False, f"Signal must be 1-dimensional, got shape {arr.shape}."

    if len(arr) != 216:
        return False, f"Expected exactly 216 ECG samples, but got {len(arr)} samples."

    if np.isnan(arr).any():
        return False, "Signal contains invalid NaN (Not a Number) values."

    if np.isinf(arr).any():
        return False, "Signal contains invalid Infinite values."

    return True, ""


def parse_csv_beat_input(csv_content: Union[str, bytes]) -> np.ndarray:
    """
    Parses a CSV string or byte stream into a 1D numpy array of 216 float samples.
    
    Supports:
    - Single column of numbers with or without a header
    - Single row of numbers (comma, tab, space, or newline delimited)
    
    Parameters
    ----------
    csv_content : Union[str, bytes]
        CSV data string or byte buffer
        
    Returns
    -------
    np.ndarray
        1D array of shape (216,) with dtype float64
        
    Raises
    ------
    ValueError
        If content cannot be parsed into exactly 216 numeric samples
    """
    if isinstance(csv_content, bytes):
        csv_str = csv_content.decode("utf-8", errors="replace")
    else:
        csv_str = str(csv_content)

    csv_str = csv_str.strip()
    if not csv_str:
        raise ValueError("Input CSV content is empty.")

    # Try pandas read_csv first
    try:
        df = pd.read_csv(io.StringIO(csv_str), header=None)
        vals = df.values.flatten()
        clean_vals = []
        for v in vals:
            if pd.notna(v):
                try:
                    clean_vals.append(float(v))
                except (ValueError, TypeError):
                    pass  # Skip non-numeric header cells
        if len(clean_vals) == 216:
            return np.array(clean_vals, dtype=np.float64)
    except Exception:
        pass

    # Fallback parsing: split by common delimiters
    raw_tokens = csv_str.replace("\n", ",").replace("\r", ",").replace("\t", ",").replace(" ", ",").split(",")
    clean_vals = []
    for token in raw_tokens:
        token_str = token.strip()
        if not token_str:
            continue
        try:
            val = float(token_str)
            clean_vals.append(val)
        except ValueError:
            continue

    if len(clean_vals) != 216:
        raise ValueError(
            f"Parsed {len(clean_vals)} numeric samples from input. "
            f"Expected exactly 216 samples."
        )

    return np.array(clean_vals, dtype=np.float64)


def load_test_dataset_sample(
    beats_path: Union[str, Path],
    metadata_path: Union[str, Path],
    index: int,
) -> Dict[str, Any]:
    """
    Loads a specific beat sample from dataset files beats.npy and metadata.csv.
    
    Parameters
    ----------
    beats_path : Union[str, Path]
        Path to beats.npy
    metadata_path : Union[str, Path]
        Path to metadata.csv
    index : int
        Zero-based index of the beat sample to load
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - beat_signal: 1D np.ndarray (216,)
        - record_id: int/str
        - sample_index: int
        - ground_truth: str (AAMI class)
        - original_symbol: str
        - beat_id: str
        - num_total_beats: int
    """
    beats_path = Path(beats_path)
    metadata_path = Path(metadata_path)

    if not beats_path.exists():
        raise FileNotFoundError(f"Test beats dataset file not found at {beats_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Test metadata dataset file not found at {metadata_path}")

    beats_arr = np.load(beats_path, mmap_mode="r")
    num_total = len(beats_arr)

    if index < 0 or index >= num_total:
        raise IndexError(f"Beat index {index} out of bounds for dataset of size {num_total}.")

    signal = np.array(beats_arr[index], dtype=np.float64)

    df_meta = pd.read_csv(metadata_path)
    if index >= len(df_meta):
        meta_row = {}
    else:
        meta_row = df_meta.iloc[index].to_dict()

    return {
        "beat_signal": signal,
        "record_id": meta_row.get("record_id", "N/A"),
        "sample_index": meta_row.get("sample_index", "N/A"),
        "ground_truth": meta_row.get("aami_class", None),
        "original_symbol": meta_row.get("original_symbol", "N/A"),
        "beat_id": meta_row.get("beat_id", f"beat_{index:06d}"),
        "num_total_beats": num_total,
    }


def format_ground_truth_comparison(
    ground_truth: Optional[str], pred_label: str
) -> Dict[str, Any]:
    """
    Formats comparison between ground-truth label and model prediction.
    
    Parameters
    ----------
    ground_truth : Optional[str]
        Ground-truth AAMI class label (or None)
    pred_label : str
        Model predicted AAMI class label
        
    Returns
    -------
    Dict[str, Any]
        Comparison dictionary with keys: ground_truth, prediction, is_match, result_text
    """
    if not ground_truth or str(ground_truth).strip() == "":
        return {
            "ground_truth": "Unknown",
            "prediction": pred_label,
            "is_match": None,
            "result_text": "Ground truth not available",
        }

    gt_clean = str(ground_truth).strip().upper()
    pred_clean = str(pred_label).strip().upper()

    is_match = gt_clean == pred_clean
    result_text = "Match" if is_match else "Mismatch"

    return {
        "ground_truth": gt_clean,
        "prediction": pred_clean,
        "is_match": is_match,
        "result_text": result_text,
    }


def create_ecg_plot(
    signal: np.ndarray,
    title: str = "216-Sample ECG Waveform Window",
    sample_rate: float = 360.0,
) -> go.Figure:
    """
    Creates a clean, interactive Plotly line chart for the 216-sample ECG waveform.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array of 216 sample values
    title : str
        Chart title
    sample_rate : float
        ECG sampling frequency in Hz (default 360 Hz)
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    x_indices = np.arange(len(signal))
    
    fig = go.Figure()

    # Zero reference line
    fig.add_shape(
        type="line",
        x0=0,
        x1=len(signal) - 1,
        y0=0,
        y1=0,
        line=dict(color="rgba(150, 150, 150, 0.4)", width=1, dash="dash"),
    )

    # ECG trace
    fig.add_trace(
        go.Scatter(
            x=x_indices,
            y=signal,
            mode="lines",
            name="ECG Signal",
            line=dict(color="#0066cc", width=2.2),
            hovertemplate="Sample: %{x}<br>Amplitude: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1e293b")),
        xaxis=dict(
            title="Sample Index (0–215 @ 360 Hz)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=False,
            dtick=20,
        ),
        yaxis=dict(
            title="ECG Amplitude (Normalized / mV)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            zeroline=False,
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=380,
        hovermode="x unified",
    )

    return fig


# ==============================================================================
# STREAMLIT APPLICATION (Cached resources & UI layout)
# ==============================================================================

@st.cache_resource
def get_cached_inference_service() -> ONNXInferenceService:
    """
    Loads and caches the ONNXInferenceService instance once per process.
    Reuses the existing inference service abstraction without duplicating logic.
    """
    logger.info("Initializing cached ONNXInferenceService for Streamlit dashboard.")
    return ONNXInferenceService()


def render_dashboard() -> None:
    """Main Streamlit UI renderer."""
    st.set_page_config(
        page_title="ECG Arrhythmia Analysis Dashboard",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title & Subtitle Header
    st.title("ECG Arrhythmia Classification")
    st.caption("Research & Deployment Dashboard | ONNX INT8 Inference Engine")

    # Important Medical Disclaimer Banner
    st.warning(f"⚠️ **Disclaimer:** {DISCLAIMER_TEXT}")

    api_url = get_api_base_url()

    service: Optional[ONNXInferenceService] = None
    health_meta: Dict[str, Any] = {}

    if not api_url:
        try:
            service = get_cached_inference_service()
            health_meta = service.get_health_metadata()
            health_meta["connection_mode"] = "Direct In-Process ONNX Runtime"
        except Exception as e:
            st.error(f"❌ Failed to initialize ONNX Inference Service: {e}")
            st.stop()
            return
    else:
        try:
            resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=5.0)
            if resp.status_code == 200:
                health_meta = resp.json()
                health_meta["connection_mode"] = f"HTTP REST API ({api_url})"
            else:
                st.error(f"❌ HTTP API at `{api_url}` returned health status {resp.status_code}")
                st.stop()
                return
        except Exception as e:
            st.error(f"❌ Failed to connect to HTTP API at `{api_url}`: {e}")
            st.stop()
            return

    # Sidebar Controls & Information
    st.sidebar.header("🕹️ Data Input Mode")
    mode = st.sidebar.radio(
        "Select Input Source:",
        ["MODE A — Test Dataset", "MODE B — Manual Beat Input"],
        help="Choose between selecting from preprocessed MIT-BIH test set or uploading/entering a custom 216-sample beat.",
    )

    project_root = get_project_root()
    beats_path = project_root / "data" / "processed" / "test" / "beats.npy"
    metadata_path = project_root / "data" / "processed" / "test" / "metadata.csv"

    selected_signal: Optional[np.ndarray] = None
    sample_info: Dict[str, Any] = {}

    if mode == "MODE A — Test Dataset":
        st.sidebar.subheader("Dataset Controls")
        if not beats_path.exists() or not metadata_path.exists():
            st.error(f"Test dataset files not found at `{beats_path}`. Please verify dataset files.")
            st.stop()
            return

        try:
            df_meta = pd.read_csv(metadata_path)
            total_beats = len(df_meta)
            
            class_filter = st.sidebar.selectbox(
                "Filter by Ground-Truth Class:",
                ["All Classes"] + sorted(list(df_meta["aami_class"].unique())),
            )

            if class_filter != "All Classes":
                filtered_indices = df_meta[df_meta["aami_class"] == class_filter].index.tolist()
                selected_sub_idx = st.sidebar.selectbox(
                    f"Select Sample (Total matching '{class_filter}': {len(filtered_indices)}):",
                    options=range(len(filtered_indices)),
                    format_func=lambda i: f"Idx {filtered_indices[i]} (Record {df_meta.iloc[filtered_indices[i]]['record_id']})",
                )
                beat_idx = filtered_indices[selected_sub_idx]
            else:
                beat_idx = st.sidebar.number_input(
                    f"Beat Index (0 to {total_beats - 1}):",
                    min_value=0,
                    max_value=total_beats - 1,
                    value=0,
                    step=1,
                )

            sample_info = load_test_dataset_sample(beats_path, metadata_path, beat_idx)
            selected_signal = sample_info["beat_signal"]

        except Exception as e:
            st.error(f"Failed to load dataset sample: {e}")
            st.stop()
            return

    else:  # MODE B — Manual Beat Input
        st.sidebar.subheader("Manual Input Controls")
        input_type = st.sidebar.radio("Input Method:", ["CSV File Upload", "Raw Text Input"])
        
        raw_input_text = ""
        uploaded_file = None

        if input_type == "CSV File Upload":
            uploaded_file = st.sidebar.file_uploader(
                "Upload 216-sample CSV file:",
                type=["csv", "txt"],
                help="File should contain exactly 216 comma or newline separated numeric values.",
            )
            if uploaded_file is not None:
                raw_input_text = uploaded_file.getvalue()
        else:
            raw_input_text = st.sidebar.text_area(
                "Enter 216 comma-separated values:",
                height=150,
                placeholder="0.01, -0.05, 0.12, ... (216 float numbers)",
            )

        if raw_input_text:
            try:
                selected_signal = parse_csv_beat_input(raw_input_text)
                sample_info = {
                    "beat_id": "manual_input",
                    "record_id": "User Input",
                    "sample_index": "N/A",
                    "ground_truth": None,
                }
            except Exception as e:
                st.sidebar.error(f"❌ Input Parsing Error: {e}")
                st.info("Please upload or enter a valid CSV containing exactly 216 numeric ECG samples.")
                st.stop()
                return
        else:
            st.info("👈 Please upload a CSV file or enter 216 numeric samples in the sidebar to run inference.")
            st.stop()
            return

    # Validate signal array
    is_valid, err_msg = validate_beat_array(selected_signal)
    if not is_valid:
        st.error(f"❌ Invalid Beat Signal: {err_msg}")
        st.stop()
        return

    # Execute ONNX INT8 Inference with wall-clock timing measurement
    try:
        if api_url:
            endpoint = f"{api_url.rstrip('/')}/predict/beat"
            t_start = time.perf_counter()
            resp = requests.post(endpoint, json={"signal": selected_signal.tolist()}, timeout=10.0)
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
            if resp.status_code == 200:
                pred_result = resp.json()
            else:
                st.error(f"❌ API Request to `{endpoint}` failed with status {resp.status_code}: {resp.text}")
                st.stop()
                return
        else:
            if service is None:
                service = get_cached_inference_service()
            t_start = time.perf_counter()
            pred_result = service.predict_beat(selected_signal)
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
    except Exception as e:
        st.error(f"❌ Inference Execution Error: {e}")
        st.stop()
        return

    # Layout: Top Summary Info Bar
    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.metric("Data Source", "MIT-BIH Test Set" if mode.startswith("MODE A") else "Manual Input")
    with col_meta2:
        st.metric("Beat ID / Index", str(sample_info.get("beat_id", sample_info.get("sample_index", "N/A"))))
    with col_meta3:
        st.metric("Record ID", str(sample_info.get("record_id", "N/A")))
    with col_meta4:
        st.metric("Inference Latency", f"{latency_ms:.2f} ms", help="Local wall-clock execution time for predict_beat call")

    st.markdown("---")

    # Section 1: ECG Waveform Visualization
    st.subheader("📈 ECG Waveform Window")
    fig = create_ecg_plot(selected_signal, title=f"ECG Beat Window (216 samples @ 360 Hz) — {sample_info.get('beat_id', '')}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Section 2: Model Prediction & Probability Panel
    col_pred, col_probs = st.columns([1, 1])

    pred = pred_result["prediction"]
    probs = pred_result["probabilities"]

    with col_pred:
        st.subheader("🎯 Model Prediction")
        st.markdown(
            f"<div style='background-color:#f8fafc; padding: 18px; border-radius: 8px; border: 1px solid #e2e8f0;'>"
            f"<h2 style='color:#0f172a; margin:0;'>Class {pred['class_label']} — {pred['confidence']*100:.1f}%</h2>"
            f"<p style='color:#475569; font-size:16px; margin-top:6px;'><b>Full Name:</b> {pred['full_name']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.write("")

        # Ground truth comparison box if available
        gt_label = sample_info.get("ground_truth", None)
        gt_comp = format_ground_truth_comparison(gt_label, pred["class_label"])

        if gt_comp["ground_truth"] != "Unknown":
            match_color = "#15803d" if gt_comp["is_match"] else "#b45309"
            match_bg = "#f0fdf4" if gt_comp["is_match"] else "#fffbeb"
            match_badge = "✅ MATCH" if gt_comp["is_match"] else "⚠️ MISMATCH"

            st.markdown(
                f"<div style='background-color:{match_bg}; padding: 14px; border-radius: 8px; border: 1px solid {match_color}33;'>"
                f"<span style='color:{match_color}; font-weight:bold;'>{match_badge}</span><br>"
                f"<b>Ground Truth:</b> {gt_comp['ground_truth']} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Prediction:</b> {gt_comp['prediction']}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption("ℹ️ *Single beat comparison is for demonstration purposes and does not imply clinical performance validation.*")

    with col_probs:
        st.subheader("📊 Class Probabilities")
        st.caption("Temperature-scaled probabilities / confidence estimates ($T = 0.9854$):")

        for cls_code in ["N", "S", "V", "F", "Q"]:
            prob_val = probs.get(cls_code, 0.0)
            cls_full = AAMI_CLASS_NAMES_FULL[cls_code].split("(")[0].strip()
            
            st.write(f"**{cls_code}** ({cls_full}): `{prob_val * 100:5.1f}%`")
            st.progress(min(max(float(prob_val), 0.0), 1.0))

    st.markdown("---")

    # Section 3 & Section 4: Model Info & Explainability Placeholder
    col_info, col_explain = st.columns([1, 1])

    with col_info:
        st.subheader("⚙️ Model & Service Metadata")
        with st.expander("Show Technical Configuration", expanded=True):
            st.json({
                "model": health_meta.get("model", "cnn_focal"),
                "runtime": health_meta.get("runtime", "onnxruntime"),
                "quantization": health_meta.get("quantization", "int8_static_qdq"),
                "execution_provider": health_meta.get("execution_provider", "CPUExecutionProvider"),
                "input_shape": health_meta.get("input_shape", [1, 1, 216]),
                "sampling_frequency": "360 Hz",
                "preprocessing": "0.5–40 Hz Butterworth bandpass + Per-beat z-score",
                "temperature_scaling_factor": health_meta.get("temperature", 0.9854227900505066),
                "num_classes": health_meta.get("num_classes", 5),
            })

    with col_explain:
        st.subheader("🔍 Explainability")
        with st.expander("Attribution Status", expanded=True):
            st.info(EXPLAINABILITY_TEXT)
            st.caption("Gradient-based attribution (Grad-CAM 1D) requires a PyTorch gradient path and will be added in a separate dedicated explainability module.")


if __name__ == "__main__":
    render_dashboard()
