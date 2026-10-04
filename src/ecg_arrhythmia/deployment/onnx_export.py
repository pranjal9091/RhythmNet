"""
ONNX Export and Numerical Verification Module for 1D CNN ECG Classifier.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Union
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort

from ecg_arrhythmia.utils import get_logger

logger = get_logger("onnx_export")


def export_pytorch_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    opset_version: int = 17,
    input_shape: Tuple[int, ...] = (1, 1, 216),
    device: torch.device = torch.device("cpu"),
) -> Path:
    """
    Exports PyTorch ECG 1D CNN model to ONNX format with dynamic batch dimension.
    
    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model in eval mode.
    output_path : Union[str, Path]
        Target file path for FP32 ONNX model.
    opset_version : int, default=17
        ONNX opset version.
    input_shape : Tuple[int, ...], default=(1, 1, 216)
        Single sample dummy input shape.
    device : torch.device, default=cpu
        Torch device.
        
    Returns
    -------
    Path
        Path to exported ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    dummy_input = torch.randn(*input_shape, device=device)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    logger.info(f"Exporting PyTorch model to ONNX: {output_path} (opset {opset_version})...")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # Validate ONNX graph structure
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph validation passed successfully.")

    return output_path


def verify_onnx_numerical_equivalence(
    pytorch_model: nn.Module,
    onnx_path: Union[str, Path],
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    max_samples: int = 500,
) -> Dict[str, float]:
    """
    Verifies numerical equivalence between PyTorch model and ONNX Runtime session.
    
    Parameters
    ----------
    pytorch_model : nn.Module
        PyTorch model instance.
    onnx_path : Union[str, Path]
        Path to exported ONNX model.
    val_loader : DataLoader
        Validation set DataLoader.
    device : torch.device, default=cpu
        Compute device for PyTorch.
    max_samples : int, default=500
        Maximum samples to compare.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing max_abs_difference, mean_abs_difference, and max_rel_difference.
    """
    pytorch_model.eval()
    pytorch_model.to(device)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), session_options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name

    torch_logits_list = []
    onnx_logits_list = []

    count = 0
    with torch.no_grad():
        for batch in val_loader:
            waveforms = batch["waveform"]  # [B, 1, 216] or [B, 216]
            if waveforms.dim() == 2:
                waveforms = waveforms.unsqueeze(1)

            # PyTorch forward
            pt_logits = pytorch_model(waveforms.to(device)).cpu().numpy()

            # ONNX forward
            ort_inputs = {input_name: waveforms.numpy().astype(np.float32)}
            ort_logits = session.run(None, ort_inputs)[0]

            torch_logits_list.append(pt_logits)
            onnx_logits_list.append(ort_logits)

            count += len(waveforms)
            if count >= max_samples:
                break

    all_pt = np.vstack(torch_logits_list)
    all_ort = np.vstack(onnx_logits_list)

    abs_diff = np.abs(all_pt - all_ort)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    rel_diff = abs_diff / (np.abs(all_pt) + 1e-7)
    max_rel_diff = float(np.max(rel_diff))

    logger.info(
        f"ONNX Equivalence Verification (N={len(all_pt)}): "
        f"Max Abs Diff = {max_abs_diff:.6e}, Mean Abs Diff = {mean_abs_diff:.6e}"
    )

    return {
        "max_abs_difference": max_abs_diff,
        "mean_abs_difference": mean_abs_diff,
        "max_rel_difference": max_rel_diff,
        "samples_verified": len(all_pt),
    }
