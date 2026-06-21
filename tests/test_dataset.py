"""Regression guard for the custom `InMemoryLoader` (§5.3).

`InMemoryLoader` replaced `torch.utils.data.DataLoader` for the RAM-resident synthetic suite.
Its entire justification is that it is **bit-identical** to `DataLoader` — same batch composition
*and* same global-RNG consumption per epoch — so swapping it in left every committed result
unchanged. That equivalence rests on reproducing two torch internals (the `_BaseDataLoaderIter`
worker `_base_seed` draw, then `RandomSampler`'s seed draw → fresh `Generator` → `randperm`). If a
future torch version changes either, training trajectories would silently diverge. These tests pin
the equivalence so that divergence fails loudly instead.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from looptab.data.dataset import InMemoryLoader, TabularDataset, TrajectoryDataset


def _record(loader, epochs=3):
    """Iterate `loader` for several epochs, capturing every batch and the global-RNG state
    after each epoch (via a probe draw) — the two things that must match `DataLoader`."""
    out = []
    for _ in range(epochs):
        batches = [(X.clone(), y.clone()) for X, y in loader]
        probe = torch.randn(3)  # advances/captures the global RNG exactly as post-epoch code would
        out.append((batches, probe))
    return out


def _assert_matches(ds, batch_size, shuffle, seed=0):
    Xt, yt = ds.tensors()

    torch.manual_seed(seed)
    dl_rec = _record(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False))

    torch.manual_seed(seed)
    il_rec = _record(InMemoryLoader(Xt, yt, batch_size, shuffle=shuffle))

    assert len(dl_rec) == len(il_rec)
    for (dl_batches, dl_probe), (il_batches, il_probe) in zip(dl_rec, il_rec):
        assert len(dl_batches) == len(il_batches)
        for (dx, dy), (ix, iy) in zip(dl_batches, il_batches):
            assert dx.dtype == ix.dtype and dy.dtype == iy.dtype
            assert torch.equal(dx, ix)
            assert torch.equal(dy, iy)
        # Post-epoch global RNG state must match too: the per-epoch RNG draw count is what
        # keeps training bit-identical, not just the permutation that gets used.
        assert torch.equal(dl_probe, il_probe)


def _tabular(n, d, multi_output, w=5):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (
        rng.integers(0, 2, size=(n, w)).astype(np.int64)
        if multi_output
        else rng.integers(0, 2, size=n).astype(np.int64)
    )
    return TabularDataset(X, y)


def test_inmemory_matches_dataloader_single_output_shuffled():
    _assert_matches(_tabular(40, 6, multi_output=False), batch_size=16, shuffle=True)


def test_inmemory_matches_dataloader_single_output_sequential():
    _assert_matches(_tabular(40, 6, multi_output=False), batch_size=16, shuffle=False)


def test_inmemory_matches_dataloader_multi_output_shuffled():
    _assert_matches(_tabular(40, 6, multi_output=True, w=5), batch_size=16, shuffle=True)


def test_inmemory_matches_dataloader_non_divisible_batch():
    # 37 rows / batch 16 -> a short final batch (drop_last=False); the riskiest chunking case.
    _assert_matches(_tabular(37, 4, multi_output=False), batch_size=16, shuffle=True)


def test_inmemory_matches_dataloader_single_batch():
    # batch_size >= n: one full batch per epoch.
    _assert_matches(_tabular(20, 4, multi_output=True, w=3), batch_size=64, shuffle=True)


def test_inmemory_matches_dataloader_trajectory():
    # TrajectoryDataset yields a 3-D target (n, T, w); the curriculum path depends on it.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 8)).astype(np.float32)
    traj = rng.integers(0, 2, size=(30, 5, 8)).astype(np.int64)
    _assert_matches(TrajectoryDataset(X, traj), batch_size=8, shuffle=True)
