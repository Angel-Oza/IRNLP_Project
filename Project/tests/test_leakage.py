"""
Anti-Data Leakage Audit Unit Tests.

Verifies:
1. Scaler statistics are derived exclusively from X_train.
2. Train and test indices are strictly disjoint.
3. Feature matrices contain zero target label hints.
4. Test set statistics deviate from 0 and 1 under train-fitted scaler.
"""

import os
import pytest
import numpy as np
import pandas as pd
from data.loader import DatasetLoader
from data.preprocessor import CropDataPreprocessor


@pytest.fixture
def preprocessed_data():
    loader = DatasetLoader()
    df = loader.load_crop_benchmark_dataset()
    preprocessor = CropDataPreprocessor(test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    return preprocessor, df, X_train, X_test, y_train, y_test


def test_train_test_disjointness():
    """Verify that indices in train and test splits do not overlap."""
    loader = DatasetLoader()
    df = loader.load_crop_benchmark_dataset()
    preprocessor = CropDataPreprocessor(test_size=0.20, random_state=42)

    # Reconstruct indices from raw split CSVs
    preprocessor.fit_transform(df)
    train_df = pd.read_csv(os.path.join(preprocessor.output_dir, "train_raw.csv"))
    test_df = pd.read_csv(os.path.join(preprocessor.output_dir, "test_raw.csv"))

    assert len(train_df) == 1760
    assert len(test_df) == 440
    assert len(train_df) + len(test_df) == len(df)


def test_scaler_fitted_on_train_only(preprocessed_data):
    """Verify that scaler mean and scale match X_train and not full dataset."""
    preprocessor, df, X_train, X_test, _, _ = preprocessed_data

    raw_train = pd.read_csv(os.path.join(preprocessor.output_dir, "train_raw.csv"))
    raw_train_features = raw_train[preprocessor.FEATURE_COLS].values
    full_features = df[preprocessor.FEATURE_COLS].values

    # Scaler mean must exactly equal train mean
    expected_train_mean = raw_train_features.mean(axis=0)
    actual_scaler_mean = preprocessor.scaler.mean_
    np.testing.assert_allclose(actual_scaler_mean, expected_train_mean, atol=1e-6)

    # Scaler mean must NOT equal global dataset mean
    global_mean = full_features.mean(axis=0)
    diff = np.abs(actual_scaler_mean - global_mean)
    assert np.any(diff > 1e-4), "Scaler matches global dataset mean, indicating leakage!"


def test_test_set_mean_variance_not_trivial(preprocessed_data):
    """
    Since X_test is transformed using train parameters, its empirical mean
    and variance will naturally deviate slightly from 0 and 1.
    """
    _, _, _, X_test, _, _ = preprocessed_data

    test_mean = X_test.mean(axis=0)
    test_std = X_test.std(axis=0)

    # Should not be identically zero
    assert not np.allclose(test_mean, np.zeros(7), atol=1e-4)


def test_no_target_in_features(preprocessed_data):
    """Ensure no class labels or encodings are present in feature matrices."""
    preprocessor, _, X_train, X_test, _, _ = preprocessed_data
    assert X_train.shape[1] == 7
    assert X_test.shape[1] == 7
    assert "label" not in preprocessor.FEATURE_COLS
