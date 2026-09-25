"""
Multi-Modal Dataset and Pre-Embedding Cacher.

Pairs scaled numerical sensor vectors with retrieved agronomic text embeddings (z_text)
and sensor stream reliability scores (r).
"""

import os
import json
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from rag.retriever import get_soil_retriever


class SoilMultiModalDataset(Dataset):
    """
    PyTorch Dataset yielding (x_sensor, z_text, reliability, label).
    """
    def __init__(
        self,
        X_sensor: np.ndarray,
        Z_text: np.ndarray,
        y: np.ndarray,
        reliability: Optional[np.ndarray] = None,
    ):
        self.X_sensor = torch.tensor(X_sensor, dtype=torch.float32)
        self.Z_text = torch.tensor(Z_text, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        if reliability is None:
            self.reliability = torch.ones((len(X_sensor), 1), dtype=torch.float32)
        elif isinstance(reliability, np.ndarray):
            if reliability.ndim == 1:
                reliability = reliability[:, np.newaxis]
            self.reliability = torch.tensor(reliability, dtype=torch.float32)
        else:
            self.reliability = reliability

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_sensor[idx], self.Z_text[idx], self.reliability[idx], self.y[idx]


def prepare_multimodal_embeddings(
    data_dir: Optional[str] = None, force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates and caches 384-dimensional dense text embeddings for all training and test instances.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = data_dir or os.path.join(base_dir, "data", "processed")

    z_train_path = os.path.join(data_dir, "Z_train_text.npy")
    z_test_path = os.path.join(data_dir, "Z_test_text.npy")

    if os.path.exists(z_train_path) and os.path.exists(z_test_path) and not force_recompute:
        print("Loading cached multi-modal text embeddings...")
        Z_train = np.load(z_train_path)
        Z_test = np.load(z_test_path)
        return Z_train, Z_test

    print("Computing multi-modal text embeddings from FAISS knowledge base...")
    retriever = get_soil_retriever()

    train_raw = pd.read_csv(os.path.join(data_dir, "train_raw.csv"))
    test_raw = pd.read_csv(os.path.join(data_dir, "test_raw.csv"))

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    # Compute train embeddings
    print(f"Embedding {len(train_raw)} training soil states...")
    train_queries = [
        retriever.build_query_from_soil_state(row[feature_cols].to_dict())
        for _, row in train_raw.iterrows()
    ]
    Z_train = retriever.model.encode(train_queries, show_progress_bar=False, normalize_embeddings=True)
    Z_train = np.array(Z_train, dtype=np.float32)

    # Compute test embeddings
    print(f"Embedding {len(test_raw)} testing soil states...")
    test_queries = [
        retriever.build_query_from_soil_state(row[feature_cols].to_dict())
        for _, row in test_raw.iterrows()
    ]
    Z_test = retriever.model.encode(test_queries, show_progress_bar=False, normalize_embeddings=True)
    Z_test = np.array(Z_test, dtype=np.float32)

    np.save(z_train_path, Z_train)
    np.save(z_test_path, Z_test)
    print(f"Saved text embeddings: Z_train={Z_train.shape}, Z_test={Z_test.shape}")
    return Z_train, Z_test


def get_multimodal_dataloaders(
    batch_size: int = 32, data_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """Factory creating train and test DataLoaders."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = data_dir or os.path.join(base_dir, "data", "processed")

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    Z_train, Z_test = prepare_multimodal_embeddings(data_dir=data_dir)

    train_ds = SoilMultiModalDataset(X_train, Z_train, y_train)
    test_ds = SoilMultiModalDataset(X_test, Z_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
