"""Tests for the M20 real multi-label loader (§9.4 real-tabular bridge).

Covers the §5 invariants for a new data source: golden content-hash guard, determinism (same seed
=> identical split bytes), train/test disjointness (no leakage), TRAIN-only standardization, and the
(float32, int64 (n,L)) contract. Vendored caches must exist (built by the fetch_multilabel script).
"""

import hashlib

import numpy as np
import pytest

from looptab.data.real import (
    _CACHE_DIR,
    _GOLDEN_SHA256,
    load_multilabel_pool,
    make_multilabel_splits,
)
from looptab.eval.metrics import subset_accuracy_baseline

DATASETS = ["emotions", "yeast", "scene"]
# Canonical Mulan shapes — a regression guard on the vendored content (counts, not just hashes).
EXPECTED = {
    "emotions": {"n": 593, "d": 72, "L": 6},
    "yeast": {"n": 2417, "d": 103, "L": 14},
    "scene": {"n": 2407, "d": 294, "L": 6},
}

_have_cache = all((_CACHE_DIR / f"{name}.npz").exists() for name in DATASETS)
pytestmark = pytest.mark.skipif(
    not _have_cache,
    reason="vendored datasets/*.npz absent; build via scratchpad/fetch_multilabel.py",
)


@pytest.mark.parametrize("name", DATASETS)
def test_pool_shape_and_content_hash(name):
    X, Y = load_multilabel_pool(name)
    e = EXPECTED[name]
    assert X.shape == (e["n"], e["d"])
    assert Y.shape == (e["n"], e["L"])
    assert X.dtype == np.float32 and Y.dtype == np.int64
    assert set(np.unique(Y).tolist()) <= {0, 1}
    assert not np.isnan(X).any()
    # The loader's guard must equal the recorded golden hash (and recompute it the same way).
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(Y).tobytes())
    assert h.hexdigest() == _GOLDEN_SHA256[name]


@pytest.mark.parametrize("name", DATASETS)
def test_split_determinism(name):
    cfg = {"dataset": name, "train_frac": 0.7, "standardize": True}
    tr1, te1 = make_multilabel_splits(cfg, split_seed=5)
    tr2, te2 = make_multilabel_splits(cfg, split_seed=5)
    assert np.array_equal(tr1.X, tr2.X) and np.array_equal(tr1.y, tr2.y)
    assert np.array_equal(te1.X, te2.X) and np.array_equal(te1.y, te2.y)


def _row_multiset(Xs, Ys):
    # Count (features, labels) PAIRS — a multiset, robust to datasets with duplicate rows (scene has
    # a few), unlike a plain set which would collapse them and give false disjointness failures.
    from collections import Counter

    return Counter((x.tobytes(), y.tobytes()) for x, y in zip(Xs, Ys))


@pytest.mark.parametrize("name", DATASETS)
def test_split_disjoint_and_covers_pool(name):
    # Raw (unstandardized) features so rows are byte-identical to the pool.
    cfg = {"dataset": name, "train_frac": 0.7, "standardize": False}
    tr, te = make_multilabel_splits(cfg, split_seed=3)
    X, Y = load_multilabel_pool(name)
    # train ⊎ test == pool as a MULTISET ⇒ an exact partition (no row lost, none duplicated across
    # splits) even when the pool itself contains duplicate rows.
    assert _row_multiset(tr.X, tr.y) + _row_multiset(te.X, te.y) == _row_multiset(X, Y)
    assert len(tr.X) + len(te.X) == X.shape[0]


@pytest.mark.parametrize("name", DATASETS)
def test_split_varies_with_seed(name):
    cfg = {"dataset": name, "train_frac": 0.7, "standardize": False}
    tr_a, _ = make_multilabel_splits(cfg, split_seed=1)
    tr_b, _ = make_multilabel_splits(cfg, split_seed=2)
    # Different seed => different partition (compare the label blocks, which are shape-aligned).
    n = min(len(tr_a.y), len(tr_b.y))
    assert not np.array_equal(tr_a.y[:n], tr_b.y[:n])


@pytest.mark.parametrize("name", DATASETS)
def test_standardization_uses_train_stats_only(name):
    cfg = {"dataset": name, "train_frac": 0.7, "standardize": True}
    tr, te = make_multilabel_splits(cfg, split_seed=7)
    # Train features are z-scored (mean ~0, std ~1 per column, modulo constant columns).
    assert np.allclose(tr.X.mean(axis=0), 0.0, atol=1e-4)
    train_std = tr.X.std(axis=0)
    assert np.all((np.abs(train_std - 1.0) < 1e-3) | (train_std < 1e-6))
    # Test is normalized with TRAIN stats, so its per-column mean is NOT forced to zero (leakage
    # would show as a near-zero test mean). At least some columns must drift.
    assert not np.allclose(te.X.mean(axis=0), 0.0, atol=1e-2)


@pytest.mark.parametrize("name", DATASETS)
def test_split_caps_are_respected(name):
    cfg = {"dataset": name, "train_frac": 0.7, "standardize": False}
    tr, te = make_multilabel_splits(cfg, split_seed=0, n_train=50, n_test=30)
    assert len(tr.X) == 50 and len(te.X) == 30
    assert len(tr.y) == 50 and len(te.y) == 30


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_multilabel_pool("not_a_dataset")


def test_subset_accuracy_baseline_matches_most_common_row():
    # A tiny hand-built loader-like iterable: most common row [1,0] appears 3/5 times -> 0.6.
    import torch

    y = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1], [1, 1]])
    loader = [(torch.zeros(5, 2), y)]
    assert subset_accuracy_baseline(loader) == pytest.approx(0.6)


# --- M20 review fix: K-fold CV (disjoint independent test folds) ---------------------------------
@pytest.mark.parametrize("name", DATASETS)
def test_kfold_indices_partition_pool(name):
    # The disjoint-cover guarantee is a property of the INDEX partition, so test it on the indices
    # (robust to datasets with duplicate feature rows — scene has a few — which a row-byte set would
    # wrongly collapse). This is the actual no-leakage guarantee.
    from looptab.data.real import _kfold_indices

    X, _ = load_multilabel_pool(name)
    n = X.shape[0]
    test_folds = []
    for fold in range(10):
        train_idx, test_idx = _kfold_indices(n, 10, fold, cv_seed=0)
        assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())  # no leakage within a fold
        assert len(train_idx) + len(test_idx) == n
        test_folds.append(set(test_idx.tolist()))
    # The 10 test folds are pairwise disjoint and together cover every index exactly once.
    assert sum(len(s) for s in test_folds) == n
    assert set().union(*test_folds) == set(range(n))


@pytest.mark.parametrize("name", DATASETS)
def test_kfold_split_no_leakage(name):
    # Dataset-level no-leakage at one fold, via the duplicate-robust row MULTISET.
    cfg = {"dataset": name, "n_folds": 10, "cv_seed": 0, "standardize": False}
    X, Y = load_multilabel_pool(name)
    tr, te = make_multilabel_splits(cfg, split_seed=99, fold=4)  # split_seed ignored in CV
    assert _row_multiset(tr.X, tr.y) + _row_multiset(te.X, te.y) == _row_multiset(X, Y)


@pytest.mark.parametrize("name", DATASETS)
def test_kfold_determinism_and_seed_independence(name):
    cfg = {"dataset": name, "n_folds": 10, "cv_seed": 0, "standardize": True}
    a1 = make_multilabel_splits(cfg, split_seed=0, fold=3)
    a2 = make_multilabel_splits(cfg, split_seed=777, fold=3)  # split_seed must NOT matter in CV
    assert np.array_equal(a1[1].X, a2[1].X) and np.array_equal(a1[1].y, a2[1].y)
    # Different cv_seed => different partition.
    b = make_multilabel_splits({**cfg, "cv_seed": 1}, split_seed=0, fold=3)
    assert not np.array_equal(a1[1].y, b[1].y[: len(a1[1].y)])


# --- M25: optional feature-padding for the cell-mixing arm (trm_mixer needs d % L == 0) ----------
def test_pad_to_label_multiple_off_is_identical():
    # Flag absent/False ⇒ byte-identical to the un-padded loader (M20 runs stay bit-identical).
    base = {"dataset": "yeast", "n_folds": 10, "cv_seed": 0, "standardize": True}
    tr0, te0 = make_multilabel_splits(base, split_seed=0, fold=2)
    off = {**base, "pad_to_label_multiple": False}
    tr1, te1 = make_multilabel_splits(off, split_seed=0, fold=2)
    assert np.array_equal(tr0.X, tr1.X) and np.array_equal(te0.X, te1.X)
    assert tr0.X.shape[1] == EXPECTED["yeast"]["d"]  # unchanged (103, not divisible by 14)


def test_pad_to_label_multiple_pads_yeast_to_divisible():
    cfg = {"dataset": "yeast", "n_folds": 10, "cv_seed": 0, "standardize": False,
           "pad_to_label_multiple": True}
    tr, te = make_multilabel_splits(cfg, split_seed=0, fold=1)
    L = EXPECTED["yeast"]["L"]  # 14
    d_padded = tr.X.shape[1]
    assert d_padded == 112 and d_padded % L == 0  # 103 -> 112 (next multiple of 14)
    assert te.X.shape[1] == d_padded
    # 9 pad columns total (112 - 103), all constant zero (no information, no leakage).
    n_zero_cols = int(np.sum(np.all(tr.X == 0.0, axis=0)))
    assert n_zero_cols == d_padded - EXPECTED["yeast"]["d"]  # exactly 9 zero columns
    # No real feature is lost: the non-zero columns reproduce the un-padded features as a MULTISET
    # (order within cells is preserved but padding is interleaved, so compare column content).
    tr_ref, _ = make_multilabel_splits({k: v for k, v in cfg.items()
                                        if k != "pad_to_label_multiple"}, split_seed=0, fold=1)
    kept = tr.X[:, ~np.all(tr.X == 0.0, axis=0)]
    assert np.array_equal(kept, tr_ref.X)  # the padding only INSERTS zeros, never reorders reals


def test_pad_distributed_no_dead_mixer_cell():
    # M25-review fix: pad columns are DISTRIBUTED one-per-cell, so NO mixer cell (reshape to
    # (L, cell_dim)) is left with all-zero input — the yeast label-13 dead-cell confound.
    cfg = {"dataset": "yeast", "n_folds": 10, "cv_seed": 0, "standardize": True,
           "pad_to_label_multiple": True}
    tr, _ = make_multilabel_splits(cfg, split_seed=0, fold=0)
    L, d = EXPECTED["yeast"]["L"], tr.X.shape[1]
    cells = tr.X.reshape(tr.X.shape[0], L, d // L)
    dead = [c for c in range(L) if np.all(cells[:, c, :] == 0.0)]
    assert dead == [], f"cells with zero input: {dead}"
    # Every cell keeps at least cell_dim - 1 real (non-constant) features.
    for c in range(L):
        zero_slots = int(np.sum(np.all(cells[:, c, :] == 0.0, axis=0)))
        assert zero_slots <= 1


def test_pad_to_label_multiple_noop_when_already_divisible():
    # scene 294 / 6 = 49 exactly ⇒ padding is a no-op (no columns added).
    cfg = {"dataset": "scene", "n_folds": 10, "cv_seed": 0, "standardize": False,
           "pad_to_label_multiple": True}
    tr, _ = make_multilabel_splits(cfg, split_seed=0, fold=0)
    assert tr.X.shape[1] == EXPECTED["scene"]["d"]


def test_pad_to_label_multiple_determinism():
    cfg = {"dataset": "yeast", "n_folds": 10, "cv_seed": 0, "standardize": True,
           "pad_to_label_multiple": True}
    a = make_multilabel_splits(cfg, split_seed=0, fold=3)
    b = make_multilabel_splits(cfg, split_seed=0, fold=3)
    assert np.array_equal(a[0].X, b[0].X) and np.array_equal(a[1].X, b[1].X)


# --- M26: multivariate time-series forecasting (regression bridge) --------------------------------
_FORECAST_SHAPES = {"etth1": (17420, 7), "weather": (52696, 21)}
_have_forecast = {ds: (_CACHE_DIR / f"{ds}.npz").exists() for ds in _FORECAST_SHAPES}
_forecast_params = [
    pytest.param(ds, marks=pytest.mark.skipif(not have, reason=f"datasets/{ds}.npz absent"))
    for ds, have in _have_forecast.items()
]


@pytest.mark.parametrize("dataset", _forecast_params)
def test_forecast_series_shape_and_hash(dataset):
    import hashlib

    from looptab.data.real import _FORECAST_SHA256, load_forecast_series

    series = load_forecast_series(dataset)
    assert series.shape == _FORECAST_SHAPES[dataset] and series.dtype == np.float32
    assert hashlib.sha256(series.tobytes()).hexdigest() == _FORECAST_SHA256[dataset]


@pytest.mark.parametrize("dataset", _forecast_params)
def test_forecast_windows_shapes_and_divisibility(dataset):
    from looptab.data.real import make_forecast_splits

    M = _FORECAST_SHAPES[dataset][1]
    L, H = 96, 24
    cfg = {"dataset": dataset, "lookback": L, "horizon": H, "n_folds": 10, "test_frac": 0.3}
    tr, te = make_forecast_splits(cfg, split_seed=0, fold=0)
    assert tr.X.shape[1] == M * L and tr.X.shape[1] % M == 0  # divisible ⇒ mixer-compatible
    assert tr.y.shape[1:] == (M, H) and te.y.shape[1:] == (M, H)  # M variable-cells × horizon
    assert tr.X.dtype == np.float32 and tr.y.dtype == np.float32


@pytest.mark.parametrize("dataset", _forecast_params)
def test_forecast_determinism_and_seed_maps_to_fold(dataset):
    from looptab.data.real import make_forecast_splits

    cfg = {"dataset": dataset, "lookback": 96, "horizon": 24, "n_folds": 10, "test_frac": 0.3}
    a = make_forecast_splits(cfg, split_seed=0, fold=3)
    b = make_forecast_splits(cfg, split_seed=0, fold=3)
    assert np.array_equal(a[1].X, b[1].X) and np.array_equal(a[1].y, b[1].y)
    c = make_forecast_splits(cfg, split_seed=0, fold=4)  # different fold ⇒ disjoint block
    assert not np.array_equal(a[1].y[:3], c[1].y[:3])


@pytest.mark.parametrize("dataset", _forecast_params)
def test_forecast_no_lookahead_leakage(dataset):
    # Every TRAIN window must end strictly before the test block's first input (expanding-window
    # backtest with a purge gap). Raw (unstandardized) windows compared to the source series.
    from looptab.data.real import _forecast_windows, load_forecast_series, make_forecast_splits

    M = _FORECAST_SHAPES[dataset][1]
    L, H = 96, 24
    cfg = {"dataset": dataset, "lookback": L, "horizon": H, "n_folds": 10, "test_frac": 0.3,
           "standardize": False}
    tr, te = make_forecast_splits(cfg, split_seed=0, fold=5)
    Xall, _ = _forecast_windows(load_forecast_series(dataset), L, H)  # (N, M, L)
    first_test = te.X[0].reshape(M, L)
    test_origin = int(np.where((Xall == first_test).all(axis=(1, 2)))[0][0])
    last_train = tr.X[-1].reshape(M, L)
    last_train_start = int(np.where((Xall == last_train).all(axis=(1, 2)))[0][0])
    assert last_train_start + L + H - 1 < test_origin  # train window ends before the test origin


@pytest.mark.parametrize("dataset", _forecast_params)
def test_forecast_standardization_no_leakage(dataset):
    # M26-review m-1: the z-score stats must come ONLY from series strictly before the test block.
    from looptab.data.real import _forecast_windows, load_forecast_series, make_forecast_splits

    M = _FORECAST_SHAPES[dataset][1]
    L, H = 96, 24
    cfg = {"dataset": dataset, "lookback": L, "horizon": H, "n_folds": 10, "test_frac": 0.3,
           "standardize": True}
    series = load_forecast_series(dataset)
    te = make_forecast_splits(cfg, split_seed=0, fold=5)[1]
    Xall, _ = _forecast_windows(series, L, H)
    raw_te = make_forecast_splits({**cfg, "standardize": False}, split_seed=0, fold=5)[1]
    raw = raw_te.X[0].reshape(M, L)
    origin = int(np.where((Xall == raw).all(axis=(1, 2)))[0][0])
    mu, sd = series[:origin].mean(axis=0), series[:origin].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    expected = ((raw - mu[:, None]) / sd[:, None]).astype(np.float32)
    assert np.allclose(te.X[0].reshape(M, L), expected, atol=1e-5)


def test_multilabel_f1_matches_hand_computation():
    from looptab.eval.metrics import multilabel_f1

    preds = np.array([[1, 0], [1, 1], [0, 0], [1, 1]])
    targ = np.array([[1, 0], [0, 1], [0, 1], [1, 1]])
    # label0: tp2 fp1 fn0 -> 0.8 ; label1: tp2 fp0 fn1 -> 0.8 ; macro=0.8
    # micro: TP4 FP1 FN1 -> 8/10 = 0.8
    out = multilabel_f1(preds, targ)
    assert out["macro_f1"] == pytest.approx(0.8)
    assert out["micro_f1"] == pytest.approx(0.8)
    # zero-division (a label with no positives anywhere) contributes 0, not NaN.
    z = multilabel_f1(np.zeros((3, 2), int), np.zeros((3, 2), int))
    assert z["micro_f1"] == 0.0 and z["macro_f1"] == 0.0
