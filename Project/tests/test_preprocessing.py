"""
Unit tests for data preprocessing module.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from data.loader import DatasetLoader
from data.preprocessor import CropDataPreprocessor


@pytest.fixture
def temp_output_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def preprocessor(temp_output_dir):
    return CropDataPreprocessor(test_size=0.20, random_state=42, output_dir=temp_output_dir)


@pytest.fixture
def sample_crop_df():
    loader = DatasetLoader()
    return loader.load_crop_benchmark_dataset()


def test_fit_transform_shapes_and_stratification(preprocessor, sample_crop_df):
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_crop_df)

    assert X_train.shape == (1760, 7)
    assert X_test.shape == (440, 7)
    assert y_train.shape == (1760,)
    assert y_test.shape == (440,)

    # Verify class balance in train (80 per class) and test (20 per class)
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    assert len(unique_train) == 22
    assert (counts_train == 80).all()

    unique_test, counts_test = np.unique(y_test, return_counts=True)
    assert len(unique_test) == 22
    assert (counts_test == 20).all()


def test_scaler_mean_variance(preprocessor, sample_crop_df):
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_crop_df)

    # Train features should have approximately mean 0 and std 1
    np.testing.assert_allclose(X_train.mean(axis=0), np.zeros(7), atol=1e-6)
    np.testing.assert_allclose(X_train.std(axis=0), np.ones(7), atol=1e-6)


def test_transform_single(preprocessor, sample_crop_df):
    preprocessor.fit_transform(sample_crop_df)

    sample_dict = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 200.0,
    }

    vec = preprocessor.transform_single(sample_dict)
    assert vec.shape == (1, 7)
    assert not np.isnan(vec).any()


def test_inverse_transform(preprocessor, sample_crop_df):
    X_train, _, _, _ = preprocessor.fit_transform(sample_crop_df)
    X_inv = preprocessor.inverse_transform_features(X_train)

    # Compare with raw train features
    raw_train = pd.read_csv(os.path.join(preprocessor.output_dir, "train_raw.csv"))
    raw_features = raw_train[preprocessor.FEATURE_COLS].values
    np.testing.assert_allclose(X_inv, raw_features, atol=1e-5)


def test_decode_target(preprocessor, sample_crop_df):
    preprocessor.fit_transform(sample_crop_df)
    decoded = preprocessor.decode_target(0)
    assert isinstance(decoded, str)
    assert decoded in preprocessor.class_names

    decoded_list = preprocessor.decode_target(np.array([0, 1, 2]))
    assert len(decoded_list) == 3


def test_load_saved_artifacts(preprocessor, sample_crop_df, temp_output_dir):
    preprocessor.fit_transform(sample_crop_df)

    new_prep = CropDataPreprocessor(output_dir=temp_output_dir)
    new_prep.load_saved_artifacts()

    assert new_prep.is_fitted
    assert len(new_prep.class_names) == 22
    assert new_prep.feature_names == preprocessor.FEATURE_COLS
