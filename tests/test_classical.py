"""
Unit tests for Milestone 5 Classical Machine Learning Baselines.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from ecg_arrhythmia.models import (
    build_classical_model,
    train_model,
    predict_model,
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_confusion_matrices,
    compute_ece,
    save_model,
    load_model,
    DEFAULT_LABEL_MAPPING,
    DEFAULT_CLASS_NAMES,
)
from ecg_arrhythmia.utils import get_project_root


def test_label_mapping_deterministic():
    """Verify label mapping contains all 5 AAMI classes and maps deterministically."""
    mapping = DEFAULT_LABEL_MAPPING
    assert len(mapping) == 5
    assert set(mapping.keys()) == {"N", "S", "V", "F", "Q"}
    assert mapping["N"] == 0
    assert mapping["S"] == 1
    assert mapping["V"] == 2
    assert mapping["F"] == 3
    assert mapping["Q"] == 4


def test_classical_model_training_and_shapes():
    """Test model build, train, prediction shape, and probability sum on synthetic data."""
    np.random.seed(42)
    N_samples = 100
    N_features = 83
    X = np.random.randn(N_samples, N_features)
    y = np.random.choice([0, 1, 2, 3, 4], size=N_samples)

    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        config = {"random_state": 42, "n_estimators": 10, "max_iter": 50}
        model = build_classical_model(model_name, config)
        trained_model = train_model(model, X, y)

        y_pred, y_prob = predict_model(trained_model, X)

        assert y_pred.shape == (N_samples,)
        assert y_prob.shape == (N_samples, 5)
        np.testing.assert_allclose(np.sum(y_prob, axis=1), 1.0, rtol=1e-3, atol=1e-3)


def test_confusion_matrix_shape():
    """Verify confusion matrix shape is strictly (5, 5)."""
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2, 0, 1, 1, 3])
    cm_raw, cm_norm = compute_confusion_matrices(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    assert cm_raw.shape == (5, 5)
    assert cm_norm.shape == (5, 5)


def test_metrics_handle_zero_support():
    """Verify overall and per-class metrics handle zero-support classes (Class Q) without crashing."""
    # Validation scenario: Class Q (label 4) has zero positive support
    y_true = np.array([0, 0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 0, 1, 2, 3, 0, 1, 2, 0])
    y_prob = np.eye(5)[y_pred]

    overall = compute_overall_metrics(y_true, y_pred, y_prob, labels=[0, 1, 2, 3, 4])
    assert isinstance(overall["accuracy"], float)
    assert isinstance(overall["macro_f1"], float)
    assert not np.isnan(overall["macro_f1"])

    per_class = compute_per_class_metrics(y_true, y_pred, y_prob)
    q_row = per_class[per_class["class"] == "Q"].iloc[0]
    assert q_row["support"] == 0
    assert q_row["precision"] == 0.0
    assert q_row["recall"] == 0.0
    assert q_row["f1"] == 0.0
    assert np.isnan(q_row["pr_auc"])


def test_ece_calculation_finite():
    """Verify Expected Calibration Error (ECE) is finite and within [0, 1]."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1, 2, 3, 4], size=50)
    y_prob = np.random.dirichlet(np.ones(5), size=50)

    ece = compute_ece(y_true, y_prob, n_bins=10)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_serialization_and_reload_identity(tmp_path):
    """Verify model saving and reloading produces exact identical predictions."""
    np.random.seed(42)
    X = np.random.randn(50, 83)
    y = np.random.choice([0, 1, 2, 3, 4], size=50)

    # Test Random Forest serialization via joblib
    rf = build_classical_model("random_forest", {"n_estimators": 5, "random_state": 42})
    rf = train_model(rf, X, y)
    rf_path = tmp_path / "rf.joblib"
    save_model(rf, rf_path)

    reloaded_rf = load_model(rf_path)
    pred_orig, prob_orig = predict_model(rf, X)
    pred_reload, prob_reload = predict_model(reloaded_rf, X)

    np.testing.assert_array_equal(pred_orig, pred_reload)
    np.testing.assert_allclose(prob_orig, prob_reload, rtol=1e-5)


def test_train_only_feature_preprocessor_artifact_exists():
    """Verify the feature preprocessor artifact exists and was fit on training set."""
    project_root = get_project_root()
    preprocessor_path = project_root / "artifacts" / "features" / "preprocessor.joblib"
    assert preprocessor_path.exists()

    preprocessor = joblib.load(preprocessor_path)
    assert hasattr(preprocessor, "transform")
    assert hasattr(preprocessor, "named_steps")


def test_no_train_test_record_overlap():
    """Verify patient records in train and test splits do not overlap."""
    project_root = get_project_root()
    train_meta = pd.read_csv(project_root / "data" / "features" / "train" / "metadata.csv")
    test_meta = pd.read_csv(project_root / "data" / "features" / "test" / "metadata.csv")

    train_records = set(train_meta["record_id"].unique())
    test_records = set(test_meta["record_id"].unique())

    overlap = train_records.intersection(test_records)
    assert len(overlap) == 0, f"Patient record leakage detected: {overlap}"
