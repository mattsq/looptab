"""Thin PyTorch Dataset wrappers around the numpy generators."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .generators import make_iterated, make_linear, make_parity


@dataclass
class TabularDataset(Dataset):
    X: np.ndarray
    y: np.ndarray

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx]) if self.y.ndim > 1 else torch.tensor(self.y[idx])
        return x, y


def make_splits(
    task: str,
    task_cfg: dict,
    task_seed: int,
    train_sample_seed: int,
    test_sample_seed: int,
    n_train: int,
    n_test: int,
) -> tuple["TabularDataset", "TabularDataset"]:
    """Return (train_ds, test_ds) sharing task_seed but using different sample_seeds."""
    def _build(sample_seed, n):
        if task == "linear":
            X, y = make_linear(n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg)
            return TabularDataset(X, y)
        elif task == "parity":
            X, y, _ = make_parity(n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg)
            return TabularDataset(X, y)
        elif task == "iterated":
            X, y = make_iterated(n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg)
            return TabularDataset(X, y)
        else:
            raise ValueError(f"Unknown task: {task}")

    return _build(train_sample_seed, n_train), _build(test_sample_seed, n_test)


def make_loaders(train_ds, test_ds, batch_size: int, num_workers: int = 0):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
