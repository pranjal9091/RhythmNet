"""
Production Inference Service wrapping INT8 ONNX Model with Temperature Scaling.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
import json
import time
import numpy as np
import onnxruntime as ort

from ecg_arrhythmia.utils import get_logger, get_project_root

logger = get_logger("inference_service")

AAMI_CLASS_MAPPING = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

AAMI_CLASS_NAMES_FULL = {
    "N": "Non-Ectopic Beat (Normal / Bundle Branch)",
    "S": "Supraventricular Ectopic Beat (SVEB)",
    "V": "Ventricular Ectopic Beat (VEB)",
    "F": "Fusion Beat (Ventricular + Normal)",
    "Q": "Unknown / Paced / Unclassifiable Beat",
}

MAX_BATCH_SIZE = 1000


class ONNXInferenceService:
    """
    Production ONNX Runtime Inference Service.
    
    Loads INT8 ONNX model and temperature scaling metadata once at startup,
    validates ONNX graph input/output contract, and provides calibrated
    per-beat and batch prediction routines.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        temperature_path: Optional[Union[str, Path]] = None,
        model_name: str = "cnn_focal",
    ):
        project_root = get_project_root()

        if model_path is None:
            model_path = project_root / "artifacts" / "deployment" / "model_int8.onnx"
        if temperature_path is None:
            temperature_path = project_root / "artifacts" / "calibration" / f"{model_name}_temperature.json"

        self.model_path = Path(model_path)
        self.temperature_path = Path(temperature_path)
        self.model_name = model_name

        self.session: Optional[ort.InferenceSession] = None
        self.temperature: float = 1.0
        self.input_name: str = ""
        self.output_name: str = ""
        self.input_shape: List[Any] = []
        self.output_shape: List[Any] = []
        self.is_initialized: bool = False

        self._initialize()

    def _initialize(self) -> None:
        """Loads ONNX model and calibration parameters, verifying graph contract."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model artifact not found at {self.model_path}")

        # Load temperature scaling parameter
        if self.temperature_path.exists():
            with open(self.temperature_path, "r") as f:
                calib_meta = json.load(f)
            self.temperature = float(calib_meta.get("temperature", 1.0))
            logger.info(f"Loaded temperature scaling factor T = {self.temperature:.6f} from {self.temperature_path.name}")
        else:
            logger.warning(f"Temperature artifact not found at {self.temperature_path}. Defaulting T = 1.0")
            self.temperature = 1.0

        # Initialize ONNX Runtime InferenceSession
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            logger.error(f"Failed to create ONNX InferenceSession: {e}")
            raise RuntimeError(f"ONNX InferenceSession creation failed: {e}") from e

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if not inputs or not outputs:
            raise RuntimeError("ONNX model has invalid or empty inputs/outputs.")

        input_node = inputs[0]
        output_node = outputs[0]

        self.input_name = input_node.name
        self.output_name = output_node.name
        self.input_shape = input_node.shape
        self.output_shape = output_node.shape

        # Startup Contract Validation
        if self.input_name != "input":
            raise ValueError(f"Startup contract violation: expected input name 'input', got '{self.input_name}'")
        if self.output_name != "output":
            raise ValueError(f"Startup contract violation: expected output name 'output', got '{self.output_name}'")
        if input_node.type != "tensor(float)":
            raise ValueError(f"Startup contract violation: expected input dtype 'tensor(float)', got '{input_node.type}'")
        if output_node.type != "tensor(float)":
            raise ValueError(f"Startup contract violation: expected output dtype 'tensor(float)', got '{output_node.type}'")

        # Check dimensions
        if len(self.input_shape) != 3 or self.input_shape[-1] != 216:
            raise ValueError(f"Startup contract violation: expected trailing input dim 216, got shape {self.input_shape}")
        if len(self.output_shape) != 2 or self.output_shape[-1] != 5:
            raise ValueError(f"Startup contract violation: expected trailing output dim 5, got shape {self.output_shape}")

        self.is_initialized = True
        logger.info(
            f"ONNXInferenceService initialized successfully. "
            f"Model: {self.model_name} | Input: '{self.input_name}' {self.input_shape} | "
            f"Output: '{self.output_name}' {self.output_shape}"
        )

    def is_ready(self) -> bool:
        """Returns True if the service is fully initialized and ONNX session is ready."""
        return self.is_initialized and self.session is not None

    def get_health_metadata(self) -> Dict[str, Any]:
        """Returns service metadata required by GET /health endpoint."""
        if not self.is_ready():
            return {"status": "unhealthy", "error": "Service not initialized"}

        return {
            "status": "ok",
            "model": self.model_name,
            "runtime": "onnxruntime",
            "quantization": "int8_static_qdq",
            "execution_provider": "CPUExecutionProvider",
            "input_shape": [1, 1, 216],
            "num_classes": 5,
            "temperature": self.temperature,
            "classes": {str(k): v for k, v in AAMI_CLASS_MAPPING.items()},
        }

    def _validate_raw_logits(self, raw_logits: np.ndarray, expected_num_beats: int) -> None:
        """Validates ONNX output tensor shape and finite numeric values."""
        if not isinstance(raw_logits, np.ndarray):
            raise RuntimeError(f"ONNX output must be a numpy ndarray, got {type(raw_logits)}")
        
        expected_shape = (expected_num_beats, 5)
        if raw_logits.shape != expected_shape:
            raise RuntimeError(f"ONNX output shape mismatch: expected {expected_shape}, got {raw_logits.shape}")

        if not np.isfinite(raw_logits).all():
            raise RuntimeError("ONNX output contains non-finite (NaN or Inf) logit values.")

    def predict_beat(self, signal: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Runs calibrated ONNX INT8 inference on a single 216-sample ECG beat.
        """
        batch_res = self.predict_batch([signal])
        single_pred = batch_res["predictions"][0]

        return {
            "prediction": {
                "class_index": single_pred["class_index"],
                "class_label": single_pred["class_label"],
                "full_name": single_pred["full_name"],
                "confidence": single_pred["confidence"],
            },
            "probabilities": single_pred["probabilities"],
            "raw_logits": single_pred["raw_logits"],
            "model": self.model_name,
            "runtime": "onnx_int8",
        }

    def predict_batch(
        self, signals: Union[List[Union[List[float], np.ndarray]], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Runs calibrated ONNX INT8 batch inference on N (1 to 1000) 216-sample ECG beats.
        """
        if not self.is_ready() or self.session is None:
            raise RuntimeError("Inference session is not initialized.")

        t_start = time.perf_counter()

        if isinstance(signals, np.ndarray):
            if signals.ndim == 1 and len(signals) == 216:
                arr = signals.reshape(1, 216)
            elif signals.ndim == 2 and signals.shape[1] == 216:
                arr = signals
            else:
                raise ValueError(f"Batch signals array shape must be (N, 216), got {signals.shape}")
        else:
            if not isinstance(signals, list) or len(signals) == 0:
                raise ValueError("Batch signals list cannot be empty.")
            
            try:
                arr = np.array(signals, dtype=np.float64)
            except Exception as e:
                raise ValueError(f"Failed to convert batch input to numeric array: {e}")

            if arr.ndim == 1 and len(arr) == 216:
                arr = arr.reshape(1, 216)
            elif arr.ndim != 2 or arr.shape[1] != 216:
                raise ValueError(f"Each beat in batch must contain exactly 216 samples, got shape {arr.shape}")

        num_beats = arr.shape[0]

        if num_beats < 1:
            raise ValueError("Batch size must be at least 1 beat.")
        if num_beats > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {num_beats} exceeds maximum allowed limit of {MAX_BATCH_SIZE}.")

        # Check numeric integrity across entire batch
        if np.isnan(arr).any():
            raise ValueError("Batch signal input contains invalid NaN values.")
        if np.isinf(arr).any():
            raise ValueError("Batch signal input contains invalid Infinite values.")

        # Vectorized per-beat z-score normalization: (x - mean) / (std + 1e-8)
        means = np.mean(arr, axis=1, keepdims=True)
        stds = np.std(arr, axis=1, keepdims=True)
        norm_arr = (arr - means) / (stds + 1e-8)
        norm_f32 = norm_arr.astype(np.float32)

        # Reshape to [N, 1, 216]
        input_tensor = norm_f32.reshape(num_beats, 1, 216)

        try:
            raw_logits = self.session.run(None, {self.input_name: input_tensor})[0]  # shape (N, 5)
        except Exception as e:
            logger.error(f"ONNX Runtime batch execution failed: {e}")
            raise RuntimeError("Inference execution failed inside ONNX engine.") from e

        # Validate raw_logits shape and finite numeric state
        self._validate_raw_logits(raw_logits, num_beats)

        # Vectorized Temperature scaling: z / T
        scaled_logits = raw_logits / self.temperature

        # Vectorized Softmax
        max_scaled = np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_scaled)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        probs = exp_logits / sum_exp

        if not np.isfinite(probs).all():
            raise RuntimeError("Calibrated probabilities contain non-finite values.")

        predictions = []
        for i in range(num_beats):
            row_probs = probs[i]
            row_logits = raw_logits[i]
            pred_idx = int(np.argmax(row_probs))
            pred_label = AAMI_CLASS_MAPPING[pred_idx]
            confidence = float(row_probs[pred_idx])

            prob_dict = {AAMI_CLASS_MAPPING[j]: float(row_probs[j]) for j in range(5)}
            logits_dict = {AAMI_CLASS_MAPPING[j]: float(row_logits[j]) for j in range(5)}

            predictions.append({
                "beat_index": i,
                "class_index": pred_idx,
                "class_label": pred_label,
                "full_name": AAMI_CLASS_NAMES_FULL[pred_label],
                "confidence": confidence,
                "probabilities": prob_dict,
                "raw_logits": logits_dict,
            })

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        logger.info(
            f"Batch inference completed | num_beats={num_beats} | "
            f"latency_ms={latency_ms:.2f} | model={self.model_name}"
        )

        return {
            "status": "success",
            "model": self.model_name,
            "runtime": "onnx_int8",
            "num_beats": num_beats,
            "predictions": predictions,
        }
