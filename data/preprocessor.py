"""
Preprocessing Pipeline Module for Explainable Multi-Source Soil Health Intelligence System.

Guarantees:
- Strict train-only fitting for scalers and encoders (Zero Data Leakage).
- Deterministic stratified train/test split.
- Export of clean artifacts, metadata, and mapping dictionaries.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data.loader import DatasetLoader


class CropDataPreprocessor:
    """
    Robust Preprocessor for Dataset C (Agronomic Benchmark).
    Guarantees strict train/test isolation and reproducible scaling.
    """

    FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    TARGET_COL = "label"

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        output_dir: Optional[str] = None,
    ):
        self.test_size = test_size
        self.random_state = random_state
        if output_dir is None:
            self.output_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "processed")
            )
        else:
            self.output_dir = os.path.abspath(output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names = list(self.FEATURE_COLS)
        self.class_names: List[str] = []
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes stratified train/test split, fits scaler strictly on train set,
        and returns scaled arrays: (X_train, X_test, y_train, y_test).
        """
        X = df[self.FEATURE_COLS].copy()
        y = df[self.TARGET_COL].copy().str.strip().str.lower()

        # Stratified train/test split
        X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Fit LabelEncoder strictly on training labels
        y_train = self.label_encoder.fit_transform(y_train_series)
        y_test = self.label_encoder.transform(y_test_series)

        self.class_names = list(self.label_encoder.classes_)
        self.label_to_id = {cls_name: int(idx) for idx, cls_name in enumerate(self.class_names)}
        self.id_to_label = {int(idx): cls_name for idx, cls_name in enumerate(self.class_names)}

        # Fit StandardScaler strictly on X_train
        X_train = self.scaler.fit_transform(X_train_df)
        # Transform X_test using train-fitted parameters
        X_test = self.scaler.transform(X_test_df)

        self.is_fitted = True

        # Save processed splits and metadata to disk
        self._save_artifacts(
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_df,
            X_test_df,
            y_train_series,
            y_test_series,
        )

        return X_train, X_test, y_train, y_test

    def transform_single(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Transforms a single input dictionary of features during inference.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        df_single = pd.DataFrame([[feature_dict[col] for col in self.FEATURE_COLS]], columns=self.FEATURE_COLS)
        return self.scaler.transform(df_single)

    def inverse_transform_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse scales feature matrix back to physical units."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.scaler.inverse_transform(X_scaled)

    def decode_target(self, y_encoded: Union[int, np.ndarray]) -> Union[str, List[str]]:
        """Converts encoded integer class IDs back to string crop names."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        if isinstance(y_encoded, (int, np.integer)):
            return self.id_to_label[int(y_encoded)]
        return [self.id_to_label[int(idx)] for idx in y_encoded]

    def _save_artifacts(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        X_train_df: pd.DataFrame,
        X_test_df: pd.DataFrame,
        y_train_series: pd.Series,
        y_test_series: pd.Series,
    ) -> None:
        """Saves all processed matrices, models, and metadata to disk."""
        np.save(os.path.join(self.output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.output_dir, "y_test.npy"), y_test)

        # Save raw unscaled CSVs with labels
        train_raw = X_train_df.copy()
        train_raw["label"] = y_train_series
        train_raw.to_csv(os.path.join(self.output_dir, "train_raw.csv"), index=False)

        test_raw = X_test_df.copy()
        test_raw["label"] = y_test_series
        test_raw.to_csv(os.path.join(self.output_dir, "test_raw.csv"), index=False)

        # Save scaler and label encoder via pickle
        with open(os.path.join(self.output_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(self.output_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

        # Save metadata and mapping dicts
        metadata = {
            "num_train_samples": int(len(X_train)),
            "num_test_samples": int(len(X_test)),
            "test_size": float(self.test_size),
            "random_state": int(self.random_state),
            "num_features": len(self.FEATURE_COLS),
            "feature_names": self.FEATURE_COLS,
            "scaler_mean": [float(m) for m in self.scaler.mean_],
            "scaler_scale": [float(s) for s in self.scaler.scale_],
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "label_to_id": self.label_to_id,
            "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
        }

        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def load_saved_artifacts(self) -> None:
        """Loads fitted scaler and metadata from output_dir."""
        scaler_path = os.path.join(self.output_dir, "scaler.pkl")
        le_path = os.path.join(self.output_dir, "label_encoder.pkl")
        meta_path = os.path.join(self.output_dir, "metadata.json")

        if not (os.path.exists(scaler_path) and os.path.exists(le_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"Missing saved preprocessor artifacts in {self.output_dir}")

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(le_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.class_names = meta["class_names"]
        self.label_to_id = meta["label_to_id"]
        self.id_to_label = {int(k): v for k, v in meta["id_to_label"].items()}
        self.feature_names = meta["feature_names"]
        self.is_fitted = True


def run_preprocessing_pipeline(base_dir: Optional[str] = None) -> CropDataPreprocessor:
    """Executes end-to-end dataset loading and preprocessing."""
    loader = DatasetLoader(base_dir=base_dir)
    df_crop = loader.load_crop_benchmark_dataset()

    preprocessor = CropDataPreprocessor(output_dir=os.path.join(loader.base_dir, "data", "processed"))
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df_crop)

    print(f"Preprocessing successfully executed.")
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    print(f"Num Classes: {len(preprocessor.class_names)}")
    return preprocessor


if __name__ == "__main__":
    run_preprocessing_pipeline()
