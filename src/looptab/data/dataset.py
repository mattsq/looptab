"""Thin PyTorch Dataset wrappers around the numpy generators."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .generators import (
    make_converge,
    make_disruption,
    make_hopfield,
    make_iterated,
    make_linear,
    make_mixed_converge,
    make_multi_parity,
    make_nested_converge,
    make_parity,
    make_sudoku,
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
    seed: int | None = None,
) -> tuple["TabularDataset", "TabularDataset"]:
    """Return (train_ds, test_ds) sharing task_seed but using different sample_seeds.

    ``seed`` is the runner's raw per-run seed (un-offset). It is only used by the real-data
    `multilabel` task as the **K-fold index** (so seed s selects test fold s of a fixed partition);
    synthetic generators ignore it (they key everything off ``task_seed`` / the sample seeds).
    """

    # Real multi-label datasets (M20) have a FINITE pool, so train/test must be a single disjoint
    # partition — not two independent draws like the synthetic generators. In K-fold mode (the
    # review-fix default) the per-run `seed` selects the disjoint test fold; in legacy random-split
    # mode the partition is keyed on `task_seed`. (Imported lazily to avoid a circular import:
    # real.py imports TabularDataset from this module.)
    if task == "multilabel":
        from .real import make_multilabel_splits

        return make_multilabel_splits(
            task_cfg=task_cfg, split_seed=task_seed, n_train=n_train, n_test=n_test, fold=seed
        )

    # M26/M34: multivariate-time-series forecasting (regression). Expanding-window backtest — the
    # per-run `seed` selects the disjoint chronological test block (like multilabel's CV fold). The
    # `dataset` param picks the vendored series (defaults to the task name, so e.g. task=etth2 with
    # no explicit `dataset` override loads etth2 — see _FORECAST_SHA256 in data/real.py).
    if task in ("etth1", "weather", "etth2", "ettm1", "ettm2", "electricity", "traffic"):
        from .real import make_forecast_splits

        params = {**task_cfg, "dataset": task_cfg.get("dataset", task)}
        return make_forecast_splits(
            task_cfg=params, split_seed=task_seed, n_train=n_train, n_test=n_test, fold=seed
        )

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
        elif task == "hopfield":
            X, y = make_hopfield(n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg)
            return TabularDataset(X, y)
        elif task == "mixed_converge":
            X, y = make_mixed_converge(
                n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg
            )
            return TabularDataset(X, y)
        elif task == "nested_converge":
            X, y = make_nested_converge(
                n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg
            )
            return TabularDataset(X, y)
        elif task == "disruption":
            X, y = make_disruption(n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg)
            return TabularDataset(X, y)
        elif task == "sudoku":
            X, y = make_sudoku(n=n, task_seed=task_seed, sample_seed=sample_seed, **task_cfg)
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
    slow-converging rows is *not* yet s_inf; that gap is intentional, M8). ``hopfield`` (M13)
    carries the threshold-net iterate chain, ``mixed_converge`` (M15) the per-position mixed-CA
    chain, and ``nested_converge`` (M17) the two-timescale round chain (one frame per outer round)
    — all with the same fixed-point contract. ``task_cfg`` may carry the fixed-T reference
    (``T``) which is ignored here in favour of ``T_max``.
    """
    cfg = {k: v for k, v in task_cfg.items() if k != "T"}
    gen = {
        "converge": make_converge,
        "hopfield": make_hopfield,
        "mixed_converge": make_mixed_converge,
        "nested_converge": make_nested_converge,
        "disruption": make_disruption,
    }.get(task, make_iterated)
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

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        device: str | None = None,
    ):
        # ``device`` (GPU-resident batching): the synthetic suite is ~10MB, so on CUDA the whole
        # dataset is parked on the device ONCE and batches become device-side gathers, removing a
        # host->device copy per batch. Honest sizing: the copies are only ~0.9% of runtime, so
        # this is ~1.02x on the fp32 path the repo runs today (it measured ~1.10x under AMP,
        # where compute is cheaper and the stalls matter more) — small, but free and bit-identical,
        # and it stops the copies becoming the floor if the compute ever gets faster.
        # Every train/eval routine already calls ``X.to(device)``, which is a no-op for tensors
        # already resident, so nothing downstream changes. ``None``/"cpu" leaves the tensors
        # exactly where they were (the pre-existing path, untouched).
        self.device = device
        if device is not None and torch.device(device).type != "cpu":
            X = X.to(device)
            y = y.to(device)
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
            # Drawn on the CPU generator REGARDLESS of `device`, so the permutation — and hence
            # every batch's composition — is bit-identical whether or not the data is resident.
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            gen = torch.Generator()
            gen.manual_seed(seed)
            perm = torch.randperm(self.n, generator=gen)
        else:
            perm = torch.arange(self.n)
        if self.X.device.type != "cpu":
            perm = perm.to(self.X.device)  # index on-device; same indices, same order
        for start in range(0, self.n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield self.X[idx], self.y[idx]


def make_loaders(
    train_ds,
    test_ds,
    batch_size: int,
    num_workers: int = 0,
    device: str | None = None,
    device_resident_max_bytes: int = 256 * 2**20,
):
    # `num_workers` is accepted for call-site compatibility but unused: the data is already
    # resident in memory, so worker processes would only add IPC/serialization overhead.
    # `device`: park SMALL datasets on the accelerator once instead of copying every batch (see
    # InMemoryLoader).  M34 added forecasting tables large enough to invalidate the old "the suite
    # is ~10MB" assumption: electricity's train+test tensors are ~1.0GiB and traffic's ~2.7GiB.
    # Keeping those resident steals the activation headroom the 8GB GPU needs and can push WDDM
    # into shared-memory oversubscription (multi-second steps) or a hard OOM.  Above the explicit
    # threshold, retain the tensors on CPU and let the existing `.to(device)` batch staging path
    # copy them.  Batch order/values stay bit-identical; only storage location changes.
    train_X, train_y = train_ds.tensors()
    test_X, test_y = test_ds.tensors()
    total_bytes = sum(
        t.numel() * t.element_size() for t in (train_X, train_y, test_X, test_y)
    )
    resident_device = device
    if (
        device is not None
        and torch.device(device).type != "cpu"
        and total_bytes > device_resident_max_bytes
    ):
        resident_device = None
    train_loader = InMemoryLoader(
        train_X, train_y, batch_size, shuffle=True, device=resident_device
    )
    test_loader = InMemoryLoader(
        test_X, test_y, batch_size, shuffle=False, device=resident_device
    )
    train_loader.dataset_nbytes = total_bytes
    test_loader.dataset_nbytes = total_bytes
    train_loader.requested_device = device
    test_loader.requested_device = device
    return train_loader, test_loader
