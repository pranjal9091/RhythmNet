"""
INT8 Static Quantization Module using ONNX Runtime with Train Calibration Data.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

from ecg_arrhythmia.utils import get_logger

logger = get_logger("quantization")


class TrainCalibrationDataReader(CalibrationDataReader):
    """
    ONNX Runtime CalibrationDataReader supplying representative waveform batches
    sourced EXCLUSIVELY from the Training dataset.
    
    Parameters
    ----------
    train_loader : DataLoader
        DataLoader for TRAIN set only.
    input_name : str, default='input'
        Name of ONNX graph input node.
    max_samples : int, default=500
        Maximum calibration samples to yield.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        input_name: str = "input",
        max_samples: int = 500,
    ):
        self.input_name = input_name
        self.max_samples = max_samples
        self.data_batches = []

        count = 0
        for batch in train_loader:
            waveforms = batch["waveform"]  # [B, 1, 216] or [B, 216]
            if waveforms.dim() == 2:
                waveforms = waveforms.unsqueeze(1)
            
            wave_np = waveforms.numpy().astype(np.float32)
            self.data_batches.append({self.input_name: wave_np})
            
            count += len(waveforms)
            if count >= self.max_samples:
                break

        self.enum_data = iter(self.data_batches)
        logger.info(
            f"Initialized TrainCalibrationDataReader with {count} training calibration samples "
            f"across {len(self.data_batches)} batches."
        )

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        return next(self.enum_data, None)

    def rewind(self) -> None:
        self.enum_data = iter(self.data_batches)


def quantize_onnx_model_int8(
    input_fp32_onnx: Union[str, Path],
    output_int8_onnx: Union[str, Path],
    train_loader: DataLoader,
    n_calibration_samples: int = 500,
    quant_format: QuantFormat = QuantFormat.QDQ,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
) -> Path:
    """
    Applies INT8 static quantization to FP32 ONNX model using Training calibration data ONLY.
    
    Parameters
    ----------
    input_fp32_onnx : Union[str, Path]
        Path to input FP32 ONNX model.
    output_int8_onnx : Union[str, Path]
        Target path for quantized INT8 ONNX model.
    train_loader : DataLoader
        DataLoader for TRAIN set only.
    n_calibration_samples : int, default=500
        Number of representative training calibration samples.
    quant_format : QuantFormat, default=QuantFormat.QDQ
        Quantization format (QDQ or QOperator).
    activation_type : QuantType, default=QuantType.QInt8
        Activation quantization data type.
    weight_type : QuantType, default=QuantType.QInt8
        Weight quantization data type.
        
    Returns
    -------
    Path
        Path to quantized INT8 ONNX model.
    """
    input_fp32_onnx = Path(input_fp32_onnx)
    output_int8_onnx = Path(output_int8_onnx)
    output_int8_onnx.parent.mkdir(parents=True, exist_ok=True)

    # Initialize calibration reader using TRAIN split only
    calibration_reader = TrainCalibrationDataReader(
        train_loader=train_loader,
        input_name="input",
        max_samples=n_calibration_samples,
    )

    logger.info(f"Applying INT8 Static Quantization: {input_fp32_onnx} -> {output_int8_onnx}...")

    quantize_static(
        model_input=str(input_fp32_onnx),
        model_output=str(output_int8_onnx),
        calibration_data_reader=calibration_reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=True,
    )

    # Verify quantized ONNX graph loads
    quant_model = onnx.load(str(output_int8_onnx))
    logger.info("INT8 ONNX model graph loaded and validated successfully.")

    return output_int8_onnx
