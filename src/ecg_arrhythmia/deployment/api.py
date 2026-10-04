"""
FastAPI REST API Service for Inter-Patient ECG Arrhythmia Classification.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Union
import math
import time
import numpy as np

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ecg_arrhythmia.utils import get_logger
from ecg_arrhythmia.deployment.service import ONNXInferenceService
from ecg_arrhythmia.deployment.continuous import ContinuousInferencePipeline

logger = get_logger("deployment_api")

CLINICAL_DISCLAIMER = (
    "Research prototype for ECG arrhythmia classification according to AAMI EC57 standards. "
    "Not certified for clinical diagnostic use or primary patient monitoring."
)

# Global service & pipeline instances
_inference_service: Optional[ONNXInferenceService] = None
_continuous_pipeline: Optional[ContinuousInferencePipeline] = None


def get_inference_service() -> ONNXInferenceService:
    """Returns initialized singleton ONNXInferenceService instance."""
    global _inference_service
    if _inference_service is None or not _inference_service.is_ready():
        raise RuntimeError("Inference service is not initialized or not ready.")
    return _inference_service


def get_continuous_pipeline() -> ContinuousInferencePipeline:
    """Returns initialized singleton ContinuousInferencePipeline instance."""
    global _continuous_pipeline
    service = get_inference_service()
    if _continuous_pipeline is None or _continuous_pipeline.service != service:
        _continuous_pipeline = ContinuousInferencePipeline(service=service)
    return _continuous_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager initializing ONNX model session at startup."""
    global _inference_service
    logger.info("Initializing ONNX Inference Service at application startup...")
    try:
        _inference_service = ONNXInferenceService()
        logger.info("ONNX Inference Service initialized successfully on startup.")
    except Exception as e:
        logger.critical(f"Failed to initialize ONNX Inference Service on startup: {e}")
        # Allow app to start so health/readiness endpoints can report 503 unready status
        _inference_service = None
    
    yield
    
    logger.info("Shutting down application lifespan.")


app = FastAPI(
    title="Inter-Patient ECG Arrhythmia Classification API",
    description=(
        "Production-quality research API for 5-class heartbeat classification (N, S, V, F, Q) "
        "using a static INT8 quantized 1D Convolutional Neural Network (ONNX Runtime).\n\n"
        f"**Clinical Disclaimer**: {CLINICAL_DISCLAIMER}"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Observability middleware logging request metrics without logging raw ECG samples."""
    t_start = time.perf_counter()
    response = await call_next(request)
    t_end = time.perf_counter()
    duration_ms = (t_end - t_start) * 1000.0

    logger.info(
        f"HTTP Request | method={request.method} | path={request.url.path} | "
        f"status={response.status_code} | latency_ms={duration_ms:.2f}"
    )
    return response


# --- Pydantic Request / Response Schemas ---

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    model: str = Field(..., json_schema_extra={"example": "cnn_focal"})
    runtime: str = Field(..., json_schema_extra={"example": "onnxruntime"})
    quantization: str = Field(..., json_schema_extra={"example": "int8_static_qdq"})
    execution_provider: str = Field(..., json_schema_extra={"example": "CPUExecutionProvider"})
    input_shape: List[int] = Field(..., json_schema_extra={"example": [1, 1, 216]})
    num_classes: int = Field(..., json_schema_extra={"example": 5})
    temperature: float = Field(..., json_schema_extra={"example": 0.985423})
    classes: Dict[str, str] = Field(..., json_schema_extra={"example": {"0": "N", "1": "S", "2": "V", "3": "F", "4": "Q"}})
    disclaimer: str = Field(default=CLINICAL_DISCLAIMER)


class ReadinessResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ready"})
    model: str = Field(..., json_schema_extra={"example": "cnn_focal"})
    runtime: str = Field(..., json_schema_extra={"example": "onnxruntime"})
    is_ready: bool = Field(..., json_schema_extra={"example": True})
    disclaimer: str = Field(default=CLINICAL_DISCLAIMER)


class BeatPredictionRequest(BaseModel):
    signal: List[Union[float, int, str]] = Field(
        ...,
        description="Raw or amplitude-scaled 1D ECG beat waveform containing exactly 216 float samples (600 ms window at 360 Hz).",
        json_schema_extra={"example": [0.12, -0.05, 0.08] + [0.0] * 210 + [0.95, -0.42, 0.11]},
    )

    @field_validator("signal")
    @classmethod
    def validate_signal_length_and_values(cls, v: List[Union[float, int, str]]) -> List[float]:
        if len(v) != 216:
            raise ValueError(f"Signal must contain exactly 216 floating point values, got length {len(v)}.")

        parsed_signal: List[float] = []
        for idx, val in enumerate(v):
            try:
                f_val = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Signal element at sample index {idx} ('{val}') cannot be converted to a valid float.")

            if math.isnan(f_val):
                raise ValueError(f"Signal contains invalid NaN value at sample index {idx}.")
            if math.isinf(f_val):
                raise ValueError(f"Signal contains invalid Infinite value at sample index {idx}.")
            
            parsed_signal.append(f_val)
        
        return parsed_signal


class BatchBeatPredictionRequest(BaseModel):
    signals: List[List[Union[float, int, str]]] = Field(
        ...,
        description="List of pre-segmented 216-sample ECG beat waveforms (1 to 1000 beats).",
    )

    @field_validator("signals")
    @classmethod
    def validate_batch_signals(cls, v: List[List[Union[float, int, str]]]) -> List[List[float]]:
        if not v or len(v) == 0:
            raise ValueError("Batch signals list cannot be empty.")
        if len(v) > 1000:
            raise ValueError(f"Batch size {len(v)} exceeds maximum allowed limit of 1000 beats.")

        parsed_batch: List[List[float]] = []
        for beat_idx, beat_vals in enumerate(v):
            if len(beat_vals) != 216:
                raise ValueError(
                    f"Beat at batch index {beat_idx} must contain exactly 216 samples, got length {len(beat_vals)}."
                )
            parsed_beat: List[float] = []
            for sample_idx, val in enumerate(beat_vals):
                try:
                    f_val = float(val)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Beat index {beat_idx}, sample index {sample_idx} ('{val}') cannot be converted to float."
                    )
                if math.isnan(f_val):
                    raise ValueError(f"Beat index {beat_idx} contains invalid NaN value at sample index {sample_idx}.")
                if math.isinf(f_val):
                    raise ValueError(f"Beat index {beat_idx} contains invalid Infinite value at sample index {sample_idx}.")
                parsed_beat.append(f_val)
            parsed_batch.append(parsed_beat)

        return parsed_batch


class PredictionDetail(BaseModel):
    class_index: int = Field(..., json_schema_extra={"example": 2})
    class_label: str = Field(..., json_schema_extra={"example": "V"})
    full_name: str = Field(..., json_schema_extra={"example": "Ventricular Ectopic Beat (VEB)"})
    confidence: float = Field(..., json_schema_extra={"example": 0.9864})


class BeatPredictionResponse(BaseModel):
    prediction: PredictionDetail
    probabilities: Dict[str, float] = Field(..., json_schema_extra={"example": {"N": 0.01, "S": 0.01, "V": 0.97, "F": 0.00, "Q": 0.01}})
    raw_logits: Optional[Dict[str, float]] = Field(default=None)
    model: str = Field(..., json_schema_extra={"example": "cnn_focal"})
    runtime: str = Field(..., json_schema_extra={"example": "onnx_int8"})
    disclaimer: str = Field(default=CLINICAL_DISCLAIMER)


class SingleBatchItemPrediction(BaseModel):
    beat_index: int
    class_index: int
    class_label: str
    full_name: str
    confidence: float
    probabilities: Dict[str, float]
    raw_logits: Optional[Dict[str, float]] = None


class BatchBeatPredictionResponse(BaseModel):
    status: str = Field(default="success")
    model: str = Field(..., json_schema_extra={"example": "cnn_focal"})
    runtime: str = Field(..., json_schema_extra={"example": "onnx_int8"})
    num_beats: int = Field(..., json_schema_extra={"example": 2})
    predictions: List[SingleBatchItemPrediction]
    disclaimer: str = Field(default=CLINICAL_DISCLAIMER)


class ContinuousSignalPredictionRequest(BaseModel):
    signal: List[Union[float, int, str]] = Field(
        ...,
        description="Continuous 1D ECG signal amplitude values (minimum 216 samples).",
        json_schema_extra={"example": [0.15, 0.22, 0.18] + [0.0] * 500},
    )
    sampling_rate: float = Field(
        default=360.0,
        description="ECG sampling rate in Hz (must be 360.0 Hz).",
        json_schema_extra={"example": 360.0},
    )

    @field_validator("sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v: float) -> float:
        if abs(v - 360.0) > 1e-3:
            raise ValueError(f"Sampling rate {v} Hz is not supported. Continuous processing strictly requires 360 Hz.")
        return float(v)

    @field_validator("signal")
    @classmethod
    def validate_continuous_signal(cls, v: List[Union[float, int, str]]) -> List[float]:
        if len(v) < 216:
            raise ValueError(f"Signal length ({len(v)} samples) is too short. Minimum required length is 216 samples.")
        
        parsed_signal: List[float] = []
        for idx, val in enumerate(v):
            try:
                f_val = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Signal element at sample index {idx} ('{val}') cannot be converted to float.")
            if math.isnan(f_val):
                raise ValueError(f"Signal contains invalid NaN value at sample index {idx}.")
            if math.isinf(f_val):
                raise ValueError(f"Signal contains invalid Infinite value at sample index {idx}.")
            parsed_signal.append(f_val)
        
        return parsed_signal


class ContinuousBeatPredictionItem(BaseModel):
    beat_index: int
    r_peak_sample: int
    class_index: int
    class_label: str
    full_name: str
    confidence: float
    probabilities: Dict[str, float]
    raw_logits: Optional[Dict[str, float]] = None


class UnclassifiedPeakItem(BaseModel):
    beat_index: int
    r_peak_sample: int
    reason: str
    details: str


class ContinuousSignalPredictionResponse(BaseModel):
    status: str = Field(default="success")
    model: str = Field(..., json_schema_extra={"example": "cnn_focal"})
    runtime: str = Field(..., json_schema_extra={"example": "onnx_int8"})
    signal_length: int = Field(..., json_schema_extra={"example": 1000})
    sampling_rate: float = Field(..., json_schema_extra={"example": 360.0})
    num_detected_peaks: int = Field(..., json_schema_extra={"example": 3})
    num_processed_beats: int = Field(..., json_schema_extra={"example": 2})
    num_boundary_rejected: int = Field(..., json_schema_extra={"example": 1})
    predictions: List[ContinuousBeatPredictionItem]
    unclassified_peaks: List[UnclassifiedPeakItem] = Field(default_factory=list)
    disclaimer: str = Field(default=CLINICAL_DISCLAIMER)


# --- Exception Handlers ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for Pydantic request validation errors."""
    errors = []
    for err in exc.errors():
        msg = err.get("msg", "Invalid request input.")
        errors.append(msg)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "ValidationError",
            "message": " ; ".join(errors),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "InvalidInputError",
            "message": str(exc),
        },
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error(f"API Internal Error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "InferenceError",
            "message": "Internal inference execution failed.",
        },
    )


# --- Endpoints ---

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health and Model Metadata Check",
    tags=["System"],
)
async def health_check():
    """Returns operational health status, ONNX model graph information, and calibration metadata."""
    try:
        service = get_inference_service()
        metadata = service.get_health_metadata()
        if metadata.get("status") != "ok":
            raise HTTPException(status_code=503, detail="Inference service is not healthy.")
        return metadata
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Inference service is not ready.")


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Service Readiness Probe",
    tags=["System"],
)
async def ready_check():
    """Returns 200 OK when model and ONNX engine are ready to serve requests, or 503 if not ready."""
    try:
        service = get_inference_service()
        if not service.is_ready():
            raise HTTPException(status_code=503, detail="Inference service is not ready.")
        return {
            "status": "ready",
            "model": service.model_name,
            "runtime": "onnxruntime",
            "is_ready": True,
            "disclaimer": CLINICAL_DISCLAIMER,
        }
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Inference service is not ready.")


@app.post(
    "/predict/beat",
    response_model=BeatPredictionResponse,
    summary="Predict AAMI EC57 Class for a Single 216-Sample ECG Beat",
    tags=["Inference"],
)
async def predict_beat(request: BeatPredictionRequest):
    """
    Accepts a single pre-segmented 216-sample ECG beat waveform, applies per-beat z-score normalization,
    runs INT8 ONNX inference, applies temperature scaling, and returns calibrated class probabilities.
    """
    service = get_inference_service()
    try:
        result = service.predict_beat(request.signal)
        return result
    except ValueError as ve:
        raise ve
    except Exception as e:
        logger.error(f"Error predicting beat: {e}")
        raise RuntimeError("Inference processing failed.") from e


@app.post(
    "/predict/batch",
    response_model=BatchBeatPredictionResponse,
    summary="Predict AAMI EC57 Classes for a Batch of 216-Sample ECG Beats",
    tags=["Inference"],
)
async def predict_batch(request: BatchBeatPredictionRequest):
    """
    Accepts a batch of 1 to 1000 pre-segmented 216-sample ECG beat waveforms, applies vectorized
    per-beat z-score normalization, runs INT8 ONNX batch inference, applies temperature scaling,
    and returns calibrated class probabilities for each beat.
    """
    service = get_inference_service()
    try:
        result = service.predict_batch(request.signals)
        return result
    except ValueError as ve:
        raise ve
    except Exception as e:
        logger.error(f"Error predicting batch: {e}")
        raise RuntimeError("Inference processing failed.") from e


@app.post(
    "/predict/signal",
    response_model=ContinuousSignalPredictionResponse,
    summary="Predict AAMI EC57 Classes for an Unsegmented Continuous 1D ECG Signal",
    tags=["Inference"],
)
async def predict_signal(request: ContinuousSignalPredictionRequest):
    """
    Accepts an unsegmented continuous 1D ECG signal sampled at 360 Hz (minimum 216 samples),
    applies 0.5-40 Hz zero-phase Butterworth bandpass filtering, detects R-peaks via Pan-Tompkins algorithm,
    segments 216-sample beat windows (72 pre-R-peak, 144 post-R-peak), applies per-beat z-score normalization,
    runs INT8 ONNX batch inference, and returns calibrated predictions for all valid detected beats.
    """
    pipeline = get_continuous_pipeline()
    try:
        result = pipeline.process_signal(request.signal, sampling_rate=request.sampling_rate)
        return result
    except ValueError as ve:
        raise ve
    except Exception as e:
        logger.error(f"Error processing continuous signal: {e}")
        raise RuntimeError("Continuous signal processing failed.") from e
