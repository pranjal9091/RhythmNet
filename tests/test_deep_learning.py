"""
Unit tests for Milestone 6 Deep Learning Pipeline & 1D CNN Baseline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ecg_arrhythmia.deep_learning import (
    ECGBeatDataset,
    ECG1DCNN,
    train_one_epoch,
    evaluate_epoch,
    predict_deep_model,
    get_model_param_counts,
)
from ecg_arrhythmia.models import (
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_confusion_matrices,
    compute_ece,
)
from ecg_arrhythmia.utils import get_project_root


def test_dataset_loading_and_shapes():
    """Verify ECGBeatDataset presents waveform shape [1, 216] and valid labels."""
    np.random.seed(42)
    beats = np.random.randn(50, 216).astype(np.float32)
    meta = pd.DataFrame({
        "beat_id": np.arange(50),
        "record_id": "100",
        "sample_index": np.arange(50) * 360,
        "aami_class": np.random.choice(["N", "S", "V", "F", "Q"], size=50),
    })

    dataset = ECGBeatDataset(beats=beats, metadata=meta)
    assert len(dataset) == 50

    sample = dataset[0]
    assert "waveform" in sample
    assert "label" in sample
    assert sample["waveform"].shape == (1, 216)
    assert isinstance(sample["label"].item(), int)
    assert 0 <= sample["label"].item() <= 4


def test_cnn_forward_pass_and_probabilities():
    """Test 1D CNN forward pass, output shape (batch, 5), and softmax probability sum."""
    model = ECG1DCNN(in_channels=1, num_classes=5)
    model.eval()

    x = torch.randn(8, 1, 216)
    logits = model(x)

    assert logits.shape == (8, 5)
    assert torch.isfinite(logits).all()

    probs = torch.softmax(logits, dim=1)
    assert probs.shape == (8, 5)
    np.testing.assert_allclose(probs.sum(dim=1).detach().numpy(), 1.0, rtol=1e-4)


def test_synthetic_training_step_and_gradients():
    """Verify a single training step runs, loss is finite, and gradients exist."""
    model = ECG1DCNN(in_channels=1, num_classes=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    beats = np.random.randn(20, 216).astype(np.float32)
    meta = pd.DataFrame({
        "beat_id": np.arange(20),
        "record_id": "100",
        "sample_index": np.arange(20),
        "aami_class": np.random.choice(["N", "S", "V", "F", "Q"], size=20),
    })

    dataset = ECGBeatDataset(beats=beats, metadata=meta)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    device = torch.device("cpu")
    loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device)

    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0

    # Check that model parameters received finite gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_checkpoint_save_and_reload(tmp_path):
    """Verify model checkpoint saving and reloading produces exact identical outputs in eval mode."""
    torch.manual_seed(42)
    model = ECG1DCNN(in_channels=1, num_classes=5)
    model.eval()

    x = torch.randn(5, 1, 216)
    orig_logits = model(x)

    ckpt_path = tmp_path / "cnn_test.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "best_val_macro_f1": 0.85,
    }
    torch.save(checkpoint, ckpt_path)

    reloaded_model = ECG1DCNN(in_channels=1, num_classes=5)
    reloaded_ckpt = torch.load(ckpt_path)
    reloaded_model.load_state_dict(reloaded_ckpt["model_state_dict"])
    reloaded_model.eval()

    reloaded_logits = reloaded_model(x)
    torch.testing.assert_close(orig_logits, reloaded_logits)


def test_zero_support_q_metrics():
    """Verify metrics calculation safely handles zero-support Validation Class Q."""
    y_true = np.array([0, 0, 1, 2, 3, 0, 1, 2])  # no class 4 (Q)
    y_pred = np.array([0, 0, 1, 2, 3, 0, 1, 2])
    y_prob = np.eye(5)[y_pred]

    overall = compute_overall_metrics(y_true, y_pred, y_prob, labels=[0, 1, 2, 3, 4])
    assert np.isfinite(overall["accuracy"])
    assert np.isfinite(overall["macro_f1"])

    per_class = compute_per_class_metrics(y_true, y_pred, y_prob)
    q_row = per_class[per_class["class"] == "Q"].iloc[0]
    assert q_row["support"] == 0
    assert q_row["f1"] == 0.0
    assert np.isnan(q_row["pr_auc"])


def test_disjoint_record_splits():
    """Verify processed dataset patient records remain strictly disjoint across partitions."""
    project_root = get_project_root()
    train_meta = pd.read_csv(project_root / "data" / "processed" / "train" / "metadata.csv")
    val_meta = pd.read_csv(project_root / "data" / "processed" / "val" / "metadata.csv")
    test_meta = pd.read_csv(project_root / "data" / "processed" / "test" / "metadata.csv")

    train_recs = set(train_meta["record_id"].unique())
    val_recs = set(val_meta["record_id"].unique())
    test_recs = set(test_meta["record_id"].unique())

    assert len(train_recs.intersection(val_recs)) == 0
    assert len(train_recs.intersection(test_recs)) == 0
    assert len(val_recs.intersection(test_recs)) == 0
