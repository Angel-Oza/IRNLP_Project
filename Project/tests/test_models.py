"""
Unit and Integration Tests for Machine Learning Baseline Models.
"""

import os
import pickle
import pytest
import numpy as np
from models.baselines import get_baseline_models
from models.evaluator import ModelEvaluator
from models.trainer import BaselineTrainer


@pytest.fixture
def baseline_models():
    return get_baseline_models(random_state=42)


@pytest.fixture
def processed_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "processed")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def test_baseline_models_instantiation(baseline_models):
    assert "LogisticRegression" in baseline_models
    assert "RandomForest" in baseline_models
    assert "XGBoost" in baseline_models
    assert "MLP" in baseline_models


def test_model_training_and_shapes(baseline_models, processed_data):
    X_train, X_test, y_train, y_test = processed_data
    rf = baseline_models["RandomForest"]
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    assert preds.shape == y_test.shape
    assert len(np.unique(preds)) <= 22

    probs = rf.predict_proba(X_test)
    assert probs.shape == (len(X_test), 22)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X_test)), atol=1e-5)


def test_evaluator_metrics_bounds(processed_data):
    X_train, X_test, y_train, y_test = processed_data
    evaluator = ModelEvaluator(
        class_names=[f"class_{i}" for i in range(22)],
        feature_names=[f"f_{i}" for i in range(7)],
    )

    models = get_baseline_models(random_state=42)
    lr = models["LogisticRegression"]
    lr.fit(X_train, y_train)

    metrics = evaluator.evaluate_model("LogisticRegression", lr, X_test, y_test)
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert metrics["log_loss"] is not None and metrics["log_loss"] >= 0.0


def test_saved_checkpoints_inference():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_dir = os.path.join(base_dir, "models", "checkpoints")
    data_dir = os.path.join(base_dir, "data", "processed")

    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    for model_file in ["best_model.pkl", "randomforest.pkl", "xgboost.pkl", "logisticregression.pkl"]:
        path = os.path.join(ckpt_dir, model_file)
        assert os.path.exists(path), f"Checkpoint missing: {path}"
        with open(path, "rb") as f:
            model = pickle.load(f)

        preds = model.predict(X_test)
        assert len(preds) == len(y_test)
        acc = (preds == y_test).mean()
        assert acc >= 0.95, f"Accuracy of {model_file} dropped below 95%: {acc}"
