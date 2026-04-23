from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TensorDatasetMTL(Dataset):
    def __init__(self, X: np.ndarray, y_wind: np.ndarray, y_risk: np.ndarray, y_warn: np.ndarray, indices: np.ndarray):
        self.X = X[indices]
        self.y_wind = y_wind[indices]
        self.y_risk = y_risk[indices]
        self.y_warn = y_warn[indices]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y_wind[idx], dtype=torch.float32),
            torch.tensor(self.y_risk[idx], dtype=torch.long),
            torch.tensor(self.y_warn[idx], dtype=torch.float32),
            torch.tensor(self.indices[idx], dtype=torch.long),
        )


@dataclass
class DatasetBundle:
    X: np.ndarray
    X_static: np.ndarray
    y_wind: np.ndarray
    y_risk: np.ndarray
    y_warn: np.ndarray
    adj: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def load_dataset_npz(path: str | Path) -> DatasetBundle:
    d = np.load(path, allow_pickle=True)
    return DatasetBundle(
        X=d["X"],
        X_static=d["X_static"],
        y_wind=d["y_wind"],
        y_risk=d["y_risk"],
        y_warn=d["y_warn"],
        adj=d["adj"],
        train_idx=d["train_idx"],
        val_idx=d["val_idx"],
        test_idx=d["test_idx"],
    )


def make_dataloaders(bundle: DatasetBundle, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDatasetMTL(bundle.X, bundle.y_wind, bundle.y_risk, bundle.y_warn, bundle.train_idx)
    val_ds = TensorDatasetMTL(bundle.X, bundle.y_wind, bundle.y_risk, bundle.y_warn, bundle.val_idx)
    test_ds = TensorDatasetMTL(bundle.X, bundle.y_wind, bundle.y_risk, bundle.y_warn, bundle.test_idx)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
