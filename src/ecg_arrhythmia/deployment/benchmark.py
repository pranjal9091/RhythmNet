"""
Deployment Accuracy & Performance Benchmarking Module for PyTorch and ONNX Models.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Union, List
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import onnxruntime as ort

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from ecg_arrhythmia.utils import get_logger

logger = get_logger("benchmark")

AAMI_CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def evaluate_deployment_model(
    model_or_path: Union[nn.Module, str, Path],
    loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    is_onnx: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates PyTorch model or ONNX Runtime model on a dataset partition.
    
    Parameters
    ----------
    model_or_path : Union[nn.Module, str, Path]
        PyTorch nn.Module or Path to ONNX model.
    loader : DataLoader
        Dataset DataLoader.
    device : torch.device, default=cpu
        Torch device.
    is_onnx : bool, default=False
        Whether model_or_path is an ONNX file path.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing predictions, probabilities, targets, accuracy, balanced_accuracy,
        macro_f1, weighted_f1, per_class_f1, and confusion_matrix.
    """
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    if is_onnx:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(model_or_path), session_options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        for batch in loader:
            waveforms = batch["waveform"]
            labels = batch["label"]
            if waveforms.dim() == 2:
                waveforms = waveforms.unsqueeze(1)

            wave_np = waveforms.numpy().astype(np.float32)
            logits = session.run(None, {input_name: wave_np})[0]
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            preds = np.argmax(probs, axis=-1)

            y_true_list.append(labels.numpy())
            y_pred_list.append(preds)
            y_prob_list.append(probs)

    else:
        model = model_or_path
        model.eval()
        model.to(device)

        with torch.no_grad():
            for batch in loader:
                waveforms = batch["waveform"].to(device)
                labels = batch["label"].to(device)
                if waveforms.dim() == 2:
                    waveforms = waveforms.unsqueeze(1)

                logits = model(waveforms)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                y_true_list.append(labels.cpu().numpy())
                y_pred_list.append(preds)
                y_prob_list.append(probs)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_prob = np.vstack(y_prob_list)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else acc
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted", zero_division=0)
    
    per_class_f1_arr = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    per_class_f1 = {name: float(per_class_f1_arr[i]) for i, name in enumerate(AAMI_CLASS_NAMES)}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def benchmark_deployment_performance(
    pytorch_model: nn.Module,
    onnx_fp32_path: Union[str, Path],
    onnx_int8_path: Union[str, Path],
    sample_waveform: torch.Tensor,
    batch_waveform: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    n_warmup: int = 20,
    n_runs: int = 200,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmarks single-beat latency, batch latency, throughput, model size, and compression ratio.
    """
    onnx_fp32_path = Path(onnx_fp32_path)
    onnx_int8_path = Path(onnx_int8_path)

    if sample_waveform.dim() == 2:
        sample_waveform = sample_waveform.unsqueeze(1)
    if batch_waveform.dim() == 2:
        batch_waveform = batch_waveform.unsqueeze(1)

    single_np = sample_waveform.numpy().astype(np.float32)
    batch_np = batch_waveform.numpy().astype(np.float32)
    batch_size = len(batch_np)

    results = {}

    # 1. PyTorch FP32 Benchmark
    pytorch_model.eval()
    pytorch_model.to(device)
    single_torch = sample_waveform.to(device)
    batch_torch = batch_waveform.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = pytorch_model(single_torch)
            _ = pytorch_model(batch_torch)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = pytorch_model(single_torch)
    pt_single_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = pytorch_model(batch_torch)
    pt_batch_ms = (time.perf_counter() - t0) / n_runs * 1000.0
    pt_throughput = (batch_size * n_runs) / (pt_batch_ms * n_runs / 1000.0)

    # Calculate PyTorch params size in MB
    param_size = sum(p.numel() * p.element_size() for p in pytorch_model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in pytorch_model.buffers())
    pt_size_mb = (param_size + buffer_size) / (1024 * 1024)

    results["PyTorch FP32"] = {
        "single_beat_ms": pt_single_ms,
        "batch_ms": pt_batch_ms,
        "throughput": pt_throughput,
        "size_mb": pt_size_mb,
        "compression_ratio": 1.0,
    }

    # 2. ONNX FP32 & INT8 Benchmarks
    onnx_targets = [
        ("ONNX FP32", onnx_fp32_path),
        ("ONNX INT8", onnx_int8_path),
    ]

    for model_label, onnx_p in onnx_targets:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(onnx_p), session_options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(n_warmup):
            _ = session.run(None, {input_name: single_np})
            _ = session.run(None, {input_name: batch_np})

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = session.run(None, {input_name: single_np})
        single_ms = (time.perf_counter() - t0) / n_runs * 1000.0

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = session.run(None, {input_name: batch_np})
        batch_ms = (time.perf_counter() - t0) / n_runs * 1000.0
        throughput = (batch_size * n_runs) / (batch_ms * n_runs / 1000.0)

        file_size_mb = os.path.getsize(str(onnx_p)) / (1024 * 1024)
        compression_ratio = pt_size_mb / file_size_mb if file_size_mb > 0 else 1.0

        results[model_label] = {
            "single_beat_ms": single_ms,
            "batch_ms": batch_ms,
            "throughput": throughput,
            "size_mb": file_size_mb,
            "compression_ratio": compression_ratio,
        }

    return results
