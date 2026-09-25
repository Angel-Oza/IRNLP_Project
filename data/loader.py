"""
Data Loader Module for Explainable Multi-Source Soil Health Intelligence System.

Responsible for safely ingesting, validating, and schema-checking the three distinct
public datasets:
1. Dataset A: IVRI / ICAR KVK Bareilly Soil Laboratory Nutrient Test Dataset
2. Dataset B: Berambadi / Bechanahalli 10-Year In-Situ Environmental Telemetry (2016-2025)
3. Dataset C: Precision Agriculture 22-Crop Benchmark Dataset
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class DatasetLoader:
    """
    Unified Data Loader with strict schema verification and logical dataset separation.
    """

    # Expected column schemas
    CHEMISTRY_SCHEMA = {
        "Sample": str,
        "pH": float,
        "EC": float,
        "OC": float,
        "N": float,
        "P": float,
        "K": float,
        "S": float,
        "Zn": float,
        "B": float,
        "Fe": float,
        "Mn": float,
        "Cu": float,
        "Area in Acre": float,
        "Bigha": float,
    }

    TELEMETRY_SCHEMA = {
        "timestamp": str,
        "Temp_5cm": float,
        "SM_5cm": float,
        "EC_5cm": float,
        "Temp_50cm": float,
        "SM_50cm": float,
        "EC_50cm": float,
        "Precipitation": float,
    }

    CROP_BENCHMARK_SCHEMA = {
        "N": float,
        "P": float,
        "K": float,
        "temperature": float,
        "humidity": float,
        "ph": float,
        "rainfall": float,
        "label": str,
    }

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # Default to project root
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.dataset_dir = os.path.join(self.base_dir, "Dataset")

    def load_soil_chemistry_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Loads Dataset A: IVRI / ICAR KVK Bareilly Soil Chemical Nutrient Certificates.
        """
        if file_path is None:
            file_path = os.path.join(self.dataset_dir, "1", "AgriNet_Dataset-G7TD2K.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Soil chemistry dataset not found at: {file_path}")

        df = pd.read_csv(file_path)

        # Validate columns
        missing_cols = set(self.CHEMISTRY_SCHEMA.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Soil chemistry dataset is missing expected columns: {missing_cols}")

        # Clean types
        for col, dtype in self.CHEMISTRY_SCHEMA.items():
            if dtype == float:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def load_sensor_telemetry_dataset(
        self, year: Optional[int] = None, years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Loads Dataset B: Berambadi In-Situ Telemetry (15-min interval logs).
        If year is specified, loads single year CSV (e.g. 2016).
        If years is specified, loads and concatenates specified years chronologically.
        If both None, loads all available years (2016-2025).
        """
        telemetry_dir = os.path.join(self.dataset_dir, "2", "20782348")
        if not os.path.exists(telemetry_dir):
            raise FileNotFoundError(f"Telemetry directory not found at: {telemetry_dir}")

        if year is not None:
            target_files = [os.path.join(telemetry_dir, f"{year}_V1.csv")]
        elif years is not None:
            target_files = [os.path.join(telemetry_dir, f"{y}_V1.csv") for y in sorted(years)]
        else:
            target_files = sorted(glob.glob(os.path.join(telemetry_dir, "*_V1.csv")))

        if not target_files:
            raise FileNotFoundError(f"No telemetry CSV files found matching criteria in {telemetry_dir}")

        dfs = []
        for fp in target_files:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Telemetry file not found: {fp}")
            df_year = pd.read_csv(fp)
            df_year["timestamp"] = pd.to_datetime(df_year["timestamp"], errors="coerce")
            dfs.append(df_year)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
        return combined_df

    def load_crop_benchmark_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Loads Dataset C: Precision Agriculture 22-Crop Requirement Benchmark Dataset.
        """
        if file_path is None:
            file_path = os.path.join(self.dataset_dir, "3", "Crop_recommendation.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Crop benchmark dataset not found at: {file_path}")

        df = pd.read_csv(file_path)

        # Validate columns
        missing_cols = set(self.CROP_BENCHMARK_SCHEMA.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Crop benchmark dataset is missing expected columns: {missing_cols}")

        # Clean numerical types
        for col, dtype in self.CROP_BENCHMARK_SCHEMA.items():
            if dtype == float:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == str:
                df[col] = df[col].astype(str).str.strip().str.lower()

        return df


def get_data_loader(base_dir: Optional[str] = None) -> DatasetLoader:
    """Helper factory to retrieve DatasetLoader instance."""
    return DatasetLoader(base_dir=base_dir)


if __name__ == "__main__":
    loader = DatasetLoader()
    print("Testing Soil Chemistry Loader...")
    df_chem = loader.load_soil_chemistry_dataset()
    print(f"Loaded {len(df_chem)} chemistry records. Columns: {df_chem.columns.tolist()[:6]}...")

    print("\nTesting Telemetry Loader (2016)...")
    df_tel = loader.load_sensor_telemetry_dataset(year=2016)
    print(f"Loaded {len(df_tel)} telemetry records from 2016. Columns: {df_tel.columns.tolist()}...")

    print("\nTesting Crop Benchmark Loader...")
    df_crop = loader.load_crop_benchmark_dataset()
    print(f"Loaded {len(df_crop)} crop benchmark records across {df_crop['label'].nunique()} classes.")
