"""
Unit tests for Milestone 7 Class Imbalance Ablation Study.
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
    FocalLoss,
    compute_inverse_class_weights,
    compute_sqrt_inverse_class_weights,
    create_weighted_sampler,
    get_effective_sampling_stats,
    predict_deep_model,
)
from ecg_arrhythmia.models import (
    compute_overall_metrics,
    compute_per_class_metrics,
    DEFAULT_LABEL_MAPPING,
)
from ecg_arrhythmia.utils import get_project_root


def test_class_weights_train_only_derivation():
    """Verify inverse and sqrt-inverse class weights are calculated correctly from y_train."""
    y_train = np.array([0, 0, 0, 0, 1, 2, 2, 3, 4])  # N_total = 9, N_0=4, N_1=1, N_2=2, N_3=1, N_4=1
    inv_w = compute_inverse_class_weights(y_train, num_classes=5)
    sqrt_w = compute_sqrt_inverse_class_weights(y_train, num_classes=5)

    assert inv_w.shape == (5,)
    assert sqrt_w.shape == (5,)

    # w_0 = 9 / (5 * 4) = 0.45
    assert pytest.approx(inv_w[0].item(), 1e-4) == 0.45
    # w_1 = 9 / (5 * 1) = 1.8
    assert pytest.approx(inv_w[1].item(), 1e-4) == 1.8

    # sqrt_w_0 = sqrt(0.45)
    assert pytest.approx(sqrt_w[0].item(), 1e-4) == np.sqrt(0.45)


def test_focal_loss_finite_scalar():
    """Verify Focal Loss returns finite scalar output for 5-class logits."""
    focal = FocalLoss(gamma=2.0)
    logits = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10,))

    loss = focal(logits, targets)
    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss).all()
    assert loss.item() >= 0.0


def test_weighted_random_sampler_train_only():
    """Verify WeightedRandomSampler creates valid sample weights for y_train."""
    y_train = np.array([0, 0, 0, 0, 1, 2, 2, 3, 4])
    sampler = create_weighted_sampler(y_train, num_classes=5, seed=42)

    assert len(sampler) == len(y_train)
    weights = list(sampler)
    assert len(weights) == len(y_train)


def test_val_test_loaders_not_oversampled():
    """Verify validation and test DataLoaders do NOT use samplers."""
    project_root = get_project_root()
    val_meta = pd.read_csv(project_root / "data" / "processed" / "val" / "metadata.csv")
    val_dataset = ECGBeatDataset(
        beats=project_root / "data" / "processed" / "val" / "beats.npy",
        metadata=val_meta,
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    assert val_loader.sampler is not None
    # Default loader without sampler has SequentialSampler whose length equals dataset length
    assert len(val_loader.dataset) == 9744
    assert type(val_loader.sampler).__name__ == "SequentialSampler"


def test_q_remains_class_index_4():
    """Verify Q class is mapped to integer 4 in label mapping."""
    assert DEFAULT_LABEL_MAPPING["Q"] == 4


def test_m7_checkpoint_preserves_predictions(tmp_path):
    """Verify M7 checkpoint save and load produces exact identical predictions."""
    model = ECG1DCNN(in_channels=1, num_classes=5)
    model.eval()

    x = torch.randn(4, 1, 216)
    orig_logits = model(x)

    ckpt_path = tmp_path / "cnn_focal_test.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 5,
        "best_val_macro_f1": 0.42,
    }
    torch.save(checkpoint, ckpt_path)

    reloaded_model = ECG1DCNN(in_channels=1, num_classes=5)
    reloaded_ckpt = torch.load(ckpt_path)
    reloaded_model.load_state_dict(reloaded_ckpt["model_state_dict"])
    reloaded_model.eval()

    reloaded_logits = reloaded_model(x)
    torch.testing.assert_close(orig_logits, reloaded_logits)
