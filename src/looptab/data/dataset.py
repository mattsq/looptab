"""Thin PyTorch Dataset wrappers around the numpy generators."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .generators import (
    make_converge,
    make_iterated,
    make_linear,
    make_multi_parity,
    make_parity,
)


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

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Whole dataset as (features, targets) tensors for the in-memory loader.

        Equivalent, row-for-row and dtype-for-dtype, to stacking ``__getitem__`` over every
        index (``from_numpy`` is a zero-copy view of the same float32/int64 buffers), so the
        fast loader yields batches bit-identical to the default-collate path it replaces.
        """
        return torch.from_numpy(self.X), torch.from_numpy(self.y)


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

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Whole dataset as (features, trajectory) tensors for the in-memory loader."""
        return torch.from_numpy(self.X), torch.from_numpy(self.traj)


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
        elif task == "multi_parity":
            X, y, _ = make_multi_parity(
                n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg
            )
            return TabularDataset(X, y)
        elif task == "iterated":
            X, y = make_iterated(n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg)
            return TabularDataset(X, y)
        elif task == "converge":
            X, y = make_converge(n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg)
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
    task: str = "iterated",
) -> "TrajectoryDataset":
    """Build a trajectory training set at length ``T_max`` (M3b/M8).

    The CA-family tasks carry a trajectory: ``iterated`` (target s_T, traj last frame == s_T)
    and ``converge`` (target s_inf the fixed point — traj last frame is s_{T_max}, which for
    slow-converging rows is *not* yet s_inf; that gap is intentional, M8). ``task_cfg`` may carry
    the fixed-T reference (``T``) which is ignored here in favour of ``T_max``.
    """
    cfg = {k: v for k, v in task_cfg.items() if k != "T"}
    gen = make_converge if task == "converge" else make_iterated
    X, _, traj = gen(
        n=n, T=T_max, task_seed=task_seed, sample_seed=sample_seed, return_trajectory=True, **cfg
    )
    return TrajectoryDataset(X, traj)


class InMemoryLoader:
    """Fast batch iterator over an in-memory dataset (drop-in for ``DataLoader``).

    The synthetic suite fits entirely in RAM, so for the tiny models here the dominant
    training-loop cost was ``DataLoader``'s per-sample ``__getitem__`` + default-collate
    path, not the matmuls. This batches by slicing pre-stacked tensors instead, which
    removes that per-row Python overhead.

    **Determinism is preserved bit-for-bit** (CLAUDE.md §5.3). Creating a ``DataLoader``
    iterator consumes the global RNG in a fixed per-epoch protocol that this reproduces
    exactly, so both the global-RNG state *and* the batch composition match the loader it
    replaces (verified against ``DataLoader`` over multiple epochs):
      1. one int64 ``_base_seed`` draw (the worker seed ``_BaseDataLoaderIter`` always takes),
         discarded here as there are no workers;
      2. with ``shuffle=True``, ``RandomSampler``'s own int64 seed draw -> fresh ``Generator``
         -> ``randperm`` (``shuffle=False`` mirrors the draw-free ``SequentialSampler``).
    Batches are contiguous chunks of that permutation (``drop_last=False``), identical to
    ``BatchSampler``. Training trajectories are therefore unchanged.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = int(X.shape[0])

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # (1) The worker base-seed draw DataLoader takes on every iterator creation, even
        # with num_workers=0. Discarded, but consumed to keep the global RNG state aligned.
        _ = torch.empty((), dtype=torch.int64).random_()
        if self.shuffle:
            # (2) RandomSampler: one global-RNG int64 seed -> fresh generator -> randperm.
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            gen = torch.Generator()
            gen.manual_seed(seed)
            perm = torch.randperm(self.n, generator=gen)
        else:
            perm = torch.arange(self.n)
        for start in range(0, self.n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield self.X[idx], self.y[idx]


def make_loaders(train_ds, test_ds, batch_size: int, num_workers: int = 0):
    # `num_workers` is accepted for call-site compatibility but unused: the data is already
    # resident in memory, so worker processes would only add IPC/serialization overhead.
    train_X, train_y = train_ds.tensors()
    test_X, test_y = test_ds.tensors()
    train_loader = InMemoryLoader(train_X, train_y, batch_size, shuffle=True)
    test_loader = InMemoryLoader(test_X, test_y, batch_size, shuffle=False)
    return train_loader, test_loader
