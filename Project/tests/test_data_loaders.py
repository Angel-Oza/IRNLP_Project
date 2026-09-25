"""
Unit tests for data loader module.
"""

import os
import pytest
import pandas as pd
from data.loader import DatasetLoader, get_data_loader


@pytest.fixture
def loader():
    return get_data_loader()


def test_soil_chemistry_loader(loader):
    df = loader.load_soil_chemistry_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 257
    for col in ["pH", "EC", "OC", "N", "P", "K", "Zn", "Fe", "Mn", "Cu"]:
        assert col in df.columns
        assert not df[col].isnull().any()
        assert (df[col] >= 0).all()


def test_sensor_telemetry_loader_single_year(loader):
    df = loader.load_sensor_telemetry_dataset(year=2016)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 30000
    assert "timestamp" in df.columns
    assert "SM_5cm" in df.columns
    assert "Temp_5cm" in df.columns
    assert "Precipitation" in df.columns


def test_sensor_telemetry_loader_multi_year(loader):
    df = loader.load_sensor_telemetry_dataset(years=[2016, 2017])
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 60000
    # Verify chronological sorting
    assert df["timestamp"].is_monotonic_increasing


def test_crop_benchmark_loader(loader):
    df = loader.load_crop_benchmark_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2200
    assert df["label"].nunique() == 22
    assert (df["label"].value_counts() == 100).all()
    for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        assert col in df.columns
        assert not df[col].isnull().any()


def test_invalid_file_handling(loader):
    with pytest.raises(FileNotFoundError):
        loader.load_soil_chemistry_dataset(file_path="non_existent_file.csv")
