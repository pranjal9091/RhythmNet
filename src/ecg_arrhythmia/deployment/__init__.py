"""
Deployment module for ONNX export, INT8 static quantization, latency benchmarking, and FastAPI REST API.
"""

try:
    from ecg_arrhythmia.deployment.onnx_export import (
        export_pytorch_to_onnx,
        verify_onnx_numerical_equivalence,
    )
except ImportError:
    export_pytorch_to_onnx = None
    verify_onnx_numerical_equivalence = None

try:
    from ecg_arrhythmia.deployment.quantization import (
        TrainCalibrationDataReader,
        quantize_onnx_model_int8,
    )
except ImportError:
    TrainCalibrationDataReader = None
    quantize_onnx_model_int8 = None

try:
    from ecg_arrhythmia.deployment.benchmark import (
        evaluate_deployment_model,
        benchmark_deployment_performance,
    )
except ImportError:
    evaluate_deployment_model = None
    benchmark_deployment_performance = None

from ecg_arrhythmia.deployment.service import (
    ONNXInferenceService,
    AAMI_CLASS_MAPPING,
    AAMI_CLASS_NAMES_FULL,
)
from ecg_arrhythmia.deployment.api import app, get_inference_service
from ecg_arrhythmia.deployment.continuous import (
    detect_r_peaks,
    segment_continuous_ecg,
    ContinuousInferencePipeline,
)

__all__ = [
    "export_pytorch_to_onnx",
    "verify_onnx_numerical_equivalence",
    "TrainCalibrationDataReader",
    "quantize_onnx_model_int8",
    "evaluate_deployment_model",
    "benchmark_deployment_performance",
    "ONNXInferenceService",
    "AAMI_CLASS_MAPPING",
    "AAMI_CLASS_NAMES_FULL",
    "app",
    "get_inference_service",
    "detect_r_peaks",
    "segment_continuous_ecg",
    "ContinuousInferencePipeline",
]
