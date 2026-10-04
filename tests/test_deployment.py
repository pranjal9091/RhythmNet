"""
Unit tests for deployment, ONNX export, and INT8 static quantization.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import onnx
import onnxruntime as ort

from ecg_arrhythmia.deep_learning.models import ECG1DCNN
from ecg_arrhythmia.deployment.onnx_export import (
    export_pytorch_to_onnx,
    verify_onnx_numerical_equivalence,
)
from ecg_arrhythmia.deployment.quantization import (
    TrainCalibrationDataReader,
    quantize_onnx_model_int8,
)
from ecg_arrhythmia.deployment.benchmark import (
    evaluate_deployment_model,
    benchmark_deployment_performance,
)


@pytest.fixture
def dummy_cnn():
    model = ECG1DCNN(in_channels=1, num_classes=5)
    model.eval()
    return model


@pytest.fixture
def dummy_train_loader():
    torch.manual_seed(42)
    x = torch.randn(100, 1, 216)
    y = torch.randint(0, 5, (100,))
    dataset = TensorDataset(x, y)
    
    def collate_fn(batch):
        waveforms = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return {"waveform": waveforms, "label": labels}

    return DataLoader(dataset, batch_size=16, collate_fn=collate_fn)


def test_export_pytorch_to_onnx(dummy_cnn, tmp_path):
    fp32_path = tmp_path / "model_fp32.onnx"
    out_path = export_pytorch_to_onnx(
        model=dummy_cnn,
        output_path=fp32_path,
        opset_version=17,
        device=torch.device("cpu"),
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Load and check ONNX graph
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    assert len(onnx_model.graph.input) == 1
    assert len(onnx_model.graph.output) == 1


def test_onnx_dynamic_batch_shape(dummy_cnn, tmp_path):
    fp32_path = tmp_path / "model_fp32.onnx"
    _ = export_pytorch_to_onnx(dummy_cnn, fp32_path, device=torch.device("cpu"))

    session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Test single-beat batch
    b1_input = np.random.randn(1, 1, 216).astype(np.float32)
    b1_out = session.run(None, {input_name: b1_input})[0]
    assert b1_out.shape == (1, 5)

    # Test multi-beat batch (16 beats)
    b16_input = np.random.randn(16, 1, 216).astype(np.float32)
    b16_out = session.run(None, {input_name: b16_input})[0]
    assert b16_out.shape == (16, 5)


def test_numerical_equivalence_pytorch_vs_onnx(dummy_cnn, dummy_train_loader, tmp_path):
    fp32_path = tmp_path / "model_fp32.onnx"
    _ = export_pytorch_to_onnx(dummy_cnn, fp32_path, device=torch.device("cpu"))

    equiv_metrics = verify_onnx_numerical_equivalence(
        pytorch_model=dummy_cnn,
        onnx_path=fp32_path,
        val_loader=dummy_train_loader,
        device=torch.device("cpu"),
    )

    assert equiv_metrics["max_abs_difference"] < 1e-4
    assert equiv_metrics["mean_abs_difference"] < 1e-5
    assert equiv_metrics["samples_verified"] == 100


def test_int8_static_quantization(dummy_cnn, dummy_train_loader, tmp_path):
    fp32_path = tmp_path / "model_fp32.onnx"
    int8_path = tmp_path / "model_int8.onnx"

    _ = export_pytorch_to_onnx(dummy_cnn, fp32_path, device=torch.device("cpu"))
    _ = quantize_onnx_model_int8(
        input_fp32_onnx=fp32_path,
        output_int8_onnx=int8_path,
        train_loader=dummy_train_loader,
        n_calibration_samples=50,
    )

    assert int8_path.exists()
    assert int8_path.stat().st_size > 0

    # Verify quantized graph can be loaded and executed by ONNX Runtime
    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_in = np.random.randn(4, 1, 216).astype(np.float32)
    int8_out = session.run(None, {input_name: dummy_in})[0]

    assert int8_out.shape == (4, 5)
    assert np.all(np.isfinite(int8_out))


def test_evaluate_deployment_model(dummy_cnn, dummy_train_loader, tmp_path):
    fp32_path = tmp_path / "model_fp32.onnx"
    _ = export_pytorch_to_onnx(dummy_cnn, fp32_path, device=torch.device("cpu"))

    res_pt = evaluate_deployment_model(dummy_cnn, dummy_train_loader, is_onnx=False)
    res_onnx = evaluate_deployment_model(fp32_path, dummy_train_loader, is_onnx=True)

    assert pytest.approx(res_pt["accuracy"], abs=1e-4) == res_onnx["accuracy"]
    assert pytest.approx(res_pt["macro_f1"], abs=1e-4) == res_onnx["macro_f1"]
    np.testing.assert_array_equal(res_pt["y_pred"], res_onnx["y_pred"])
