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


@dataclass
class TrajectoryDataset(Dataset):
    """X = s0; traj = the full CA trajectory [s1..s_T_max] of shape (n, T_max, w) (M3b).

    Yields ``(x, traj_row)`` so the curriculum trainer can, per batch, pick a depth T and
    supervise loop step i against ``traj_row[i-1]`` (step-aligned DS) — or just the final
    frame (final-state DS). The trajectory's last frame equals the canonical s_T target.
    """

    X: np.ndarray
    traj: np.ndarray  # (n, T_max, w), int64

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.traj[idx])


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


def make_trajectory_dataset(
    task_cfg: dict,
    task_seed: int,
    sample_seed: int,
    n: int,
    T_max: int,
) -> "TrajectoryDataset":
    """Build a trajectory training set for the iterated-CA task at length ``T_max`` (M3b).

    Only the iterated task has a trajectory; ``task_cfg`` may carry the fixed-T reference
    (``T``) which is ignored here in favour of ``T_max`` for trajectory generation.
    """
    cfg = {k: v for k, v in task_cfg.items() if k != "T"}
    X, _, traj = make_iterated(
        n=n, T=T_max, task_seed=task_seed, sample_seed=sample_seed, return_trajectory=True, **cfg
    )
    return TrajectoryDataset(X, traj)


def make_loaders(train_ds, test_ds, batch_size: int, num_workers: int = 0):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
