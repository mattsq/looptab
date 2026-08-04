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
import pytest
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


# --- GPU-resident batching ---------------------------------------------------------------
# `device` parks the dataset on the accelerator so batches are device-side gathers instead of a
# host->device copy per batch. Its justification is the same as the loader's own: **bit-identical
# batches**, so committed results are untouched. The permutation is drawn on the CPU generator
# regardless of `device`, so both the batch composition and the global-RNG consumption must match
# the resident-free path exactly. These pin that.


def _record_values(loader, epochs=3):
    """Like `_record` but moves batches to CPU, so a device-resident loader can be compared
    value-for-value against the CPU one."""
    out = []
    for _ in range(epochs):
        batches = [(X.cpu().clone(), y.cpu().clone()) for X, y in loader]
        probe = torch.randn(3)
        out.append((batches, probe))
    return out


def _assert_device_equivalent(ds, batch_size, shuffle, device, seed=0):
    Xt, yt = ds.tensors()

    torch.manual_seed(seed)
    ref = _record_values(InMemoryLoader(Xt, yt, batch_size, shuffle=shuffle))

    torch.manual_seed(seed)
    got = _record_values(InMemoryLoader(Xt, yt, batch_size, shuffle=shuffle, device=device))

    assert len(ref) == len(got)
    for (rb, rprobe), (gb, gprobe) in zip(ref, got):
        assert len(rb) == len(gb)
        for (rx, ry), (gx, gy) in zip(rb, gb):
            assert rx.dtype == gx.dtype and ry.dtype == gy.dtype
            assert torch.equal(rx, gx)
            assert torch.equal(ry, gy)
        # The device path must not perturb the global RNG stream either.
        assert torch.equal(rprobe, gprobe)


def test_device_cpu_is_a_noop():
    # device="cpu" must be indistinguishable from device=None (the pre-existing path).
    _assert_device_equivalent(_tabular(37, 6, multi_output=True, w=5), 16, True, device="cpu")


def test_device_cpu_does_not_copy():
    # "cpu" should leave the very same tensor objects in place, not clone them.
    ds = _tabular(20, 4, multi_output=False)
    Xt, yt = ds.tensors()
    loader = InMemoryLoader(Xt, yt, 8, shuffle=False, device="cpu")
    assert loader.X is Xt and loader.y is yt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_cuda_batches_are_bit_identical():
    _assert_device_equivalent(_tabular(37, 6, multi_output=True, w=5), 16, True, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_cuda_sequential_and_trajectory():
    _assert_device_equivalent(_tabular(40, 6, multi_output=False), 16, False, device="cuda")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 8)).astype(np.float32)
    traj = rng.integers(0, 2, size=(30, 5, 8)).astype(np.int64)
    _assert_device_equivalent(TrajectoryDataset(X, traj), 8, True, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_cuda_tensors_are_resident():
    ds = _tabular(20, 4, multi_output=False)
    Xt, yt = ds.tensors()
    loader = InMemoryLoader(Xt, yt, 8, shuffle=False, device="cuda")
    assert loader.X.device.type == "cuda" and loader.y.device.type == "cuda"
    for X, y in loader:
        assert X.device.type == "cuda" and y.device.type == "cuda"
        break
