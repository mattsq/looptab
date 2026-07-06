"""Real multi-label tabular datasets (M20 — the §9.4 real-tabular bridge).

The synthetic suite (§3) is network-free by construction; this module is the deliberate,
separately-scoped step out to real data (§9.4). The datasets are **vendored** as numpy `.npz`
caches under ``datasets/`` (built once by ``scratchpad/fetch_multilabel.py``), so the *task path*
stays network-free and deterministic — loaded with numpy only, with a content-hash guard (§5).

Why multi-label classification is the right port of the §9.2 joint-state finding: outputs are
binary-per-label, so the existing multi-output head and metrics apply unchanged; **exact-match =
subset accuracy** is exactly leg-1's whole-row coherence metric; and ``trm_decoupled`` (independent
per-label latents) is literally *binary-relevance*, while the joint latent is a learned label
coupling. So Δ(trm − trm_decoupled) on EM directly tests whether joint-state coherence transfers
off-synthetic.

Seed discipline (§3) adapted to real data: the dataset is the fixed *function*; per-seed variance
comes from the random disjoint train/test *partition* (the "rows") plus model init. The split is a
pure function of ``split_seed`` (the runner passes ``task_seed = base + seed``), so a run is fully
determined by (config, seed) as §5 requires.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .dataset import TabularDataset

# Content sha256 = sha256(X.tobytes() + Y.tobytes()) of each vendored cache, recomputed on load and
# checked so a corrupted/swapped file fails loudly (§5 determinism). Printed by the fetch script.
_GOLDEN_SHA256 = {
    "emotions": "931fcac96e1cc899115e4f5b48c74b369bd74886b4fcc3a5227a91ef0824a076",
    "yeast": "d0fd35d7060b5fd72afe4caa4a2099b44ceba124ae74504250b8ead6fa60cfaa",
    "scene": "00af0d7df400a1f41d42fcab4760a7496e16cfbbf2a486bf1f8208aef82b88ec",
}

# datasets/ sits at the repo root, two levels up from this file (src/looptab/data/real.py).
_CACHE_DIR = Path(__file__).resolve().parents[3] / "datasets"

# M26 — multivariate-time-series forecasting caches. Content sha256 = sha256(series.tobytes());
# see scratchpad/fetch_forecast.py. Each `{dataset}.npz` holds a (T,M) float32 chronological series.
_FORECAST_SHA256 = {
    "etth1": "8fec12c3e12d38424e0b03cdff21909ef67b92609ba3d76b64f3937ac481e93a",   # (17420, 7)
    "weather": "8e6d8069ecea5a05ede434b930c9fada115f998d9a67455e86995bfabe5b3e30",  # (52696, 21)
}


def _content_sha256(X: np.ndarray, Y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(Y).tobytes())
    return h.hexdigest()


def load_multilabel_pool(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the full vendored ``(X, Y)`` pool for ``dataset`` and verify its content hash.

    ``X`` is ``(n, d)`` float32 features; ``Y`` is ``(n, L)`` int64 binary labels.
    """
    if dataset not in _GOLDEN_SHA256:
        raise ValueError(
            f"Unknown multilabel dataset '{dataset}'. Available: {sorted(_GOLDEN_SHA256)}"
        )
    path = _CACHE_DIR / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Vendored cache {path} missing. Rebuild it with "
            "`uv run --with scikit-learn --with scipy --with pandas "
            "python scratchpad/fetch_multilabel.py`."
        )
    with np.load(path, allow_pickle=True) as npz:
        X = np.ascontiguousarray(npz["X"], dtype=np.float32)
        Y = np.ascontiguousarray(npz["Y"], dtype=np.int64)
    sha = _content_sha256(X, Y)
    if sha != _GOLDEN_SHA256[dataset]:
        raise ValueError(
            f"{dataset} content hash mismatch (got {sha}, expected {_GOLDEN_SHA256[dataset]}). "
            "The vendored cache changed — refusing to run on non-canonical data (§5)."
        )
    return X, Y


def _pad_features_to_label_multiple(X: np.ndarray, n_labels: int) -> np.ndarray:
    """Pad ``X`` to ``d_pad`` columns (a multiple of ``n_labels``) for the cell-mixing arm (M25).

    ``TRMMixer`` reshapes the flat input into ``(n_cells=n_labels, cell_dim=d_pad/n_labels)`` cells,
    so ``d`` must be a multiple of ``n_labels``. The pad columns are constant zero (they survive
    z-scoring as zeros, carry no information), and every arm in the run sees the same padded input.

    The pad columns are **DISTRIBUTED one-per-cell** (as each cell's last slot for the first ``pad``
    cells), NOT appended contiguously (M25-review fix). Contiguous appending would fill one or more
    whole cells with zeros — e.g. yeast 103→112 puts all 9 pad columns into cell 13 (the mixer's
    prediction for label 13, yeast's RAREST label at base-rate 0.014), giving that output cell NO
    direct input while the feedforward control still sees every feature. That asymmetric, per-cell
    handicap manufactured a macro-F1 deficit (macro weights rare labels equally). Distributing the
    pad leaves every cell with ``cell_dim - 1`` or ``cell_dim`` real features (no dead cell), so the
    only remaining "unfairness" is the (unavoidable) arbitrary feature→cell grouping, shared fairly.
    ``pad == 0`` (``d`` already divisible) ⇒ a no-op that returns ``X`` unchanged.
    """
    n, d = X.shape
    cell_dim = -(-d // n_labels)  # ceil: smallest cell width making d a multiple of n_labels
    total = n_labels * cell_dim
    pad = total - d
    if pad == 0:
        return X
    if pad > n_labels:
        # More than one pad column per cell would be needed; the M25 datasets never hit this
        # (yeast pad=9 < 14 labels), and a per-cell distribution rule for pad>n_labels is
        # unspecified — fail loudly rather than silently starve cells.
        raise ValueError(
            f"pad_to_label_multiple: pad ({pad}) > n_labels ({n_labels}); the one-pad-per-cell "
            "distribution is undefined. This dataset/label geometry is unsupported."
        )
    out = np.zeros((n, total), dtype=X.dtype)
    src = 0
    for c in range(n_labels):
        n_real = cell_dim - (1 if c < pad else 0)  # first `pad` cells give a slot to a zero pad
        out[:, c * cell_dim : c * cell_dim + n_real] = X[:, src : src + n_real]
        src += n_real
    return out


def _kfold_indices(n: int, n_folds: int, fold: int, cv_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices for one fold of a deterministic K-fold partition.

    The permutation is keyed on a FIXED ``cv_seed`` (NOT the per-run seed), so all folds share one
    partition and the K test folds are **disjoint and cover the whole pool exactly once** — the
    independence fix for the M20 review (the old random 70/30 splits overlapped ~30%). ``fold`` (the
    runner passes the per-run seed) selects which fold is test; with K seeds = K folds, each row is
    a test row exactly once.
    """
    perm = np.random.default_rng(cv_seed).permutation(n)
    folds = np.array_split(perm, n_folds)
    f = fold % n_folds
    test_idx = folds[f]
    train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
    return train_idx, test_idx


def make_multilabel_splits(
    task_cfg: dict,
    split_seed: int,
    n_train: int | None = None,
    n_test: int | None = None,
    fold: int | None = None,
) -> tuple[TabularDataset, TabularDataset]:
    """Deterministic disjoint train/test split of a vendored multi-label dataset.

    Two modes (both give disjoint train/test; features always z-scored on TRAIN stats only):
      - **K-fold CV (preferred, M20-review fix):** set ``n_folds`` in ``task_cfg``. The K test folds
        are disjoint and partition the pool, so per-fold metrics are evaluated on *independent* test
        sets — restoring the paired sign test's validity (the random-split mode's overlapping test
        sets, ~0.30, made it anti-conservative). ``fold`` (the runner passes the per-run ``seed``)
        picks the test fold; the partition is keyed on a fixed ``cv_seed`` (shared across runs).
      - **Random split (legacy):** no ``n_folds`` → a single ``train_frac`` random split keyed on
        ``split_seed``. Disjoint per run, but successive runs' test sets OVERLAP, so the runner
        skips the sign test in this mode.

    ``task_cfg`` keys (from the experiment config's ``task.params``):
      - ``dataset``     — "emotions" | "yeast" (required).
      - ``n_folds``     — K for K-fold CV (e.g. 10). Absent ⇒ random-split mode.
      - ``cv_seed``     — seed for the (fixed) K-fold permutation (default 0).
      - ``train_frac``  — random-split mode only: fraction used for train (default 0.7).
      - ``standardize`` — z-score features on TRAIN statistics only (default True). Essential:
                          unlike the binary synthetic tasks these features are continuous and on
                          wildly different scales. Fit-on-train-only keeps test leakage out.
      - ``pad_to_label_multiple`` — pad ``X`` with zero columns so the feature count is a multiple
                          of the label count (default False). Needed only by the cell-mixing arm
                          ``trm_mixer`` (M25), which requires ``in_features % out_features == 0``
                          (e.g. yeast 103→112 for 14 labels). Pad columns are DISTRIBUTED
                          one-per-cell (see ``_pad_features_to_label_multiple``), NOT appended
                          contiguously, so no mixer cell is left all-zero. They are constant zero
                          (surviving z-scoring as zeros) and seen by every arm, keeping the
                          comparison fair. Off ⇒ byte-identical to the un-padded loader.

    ``n_train`` / ``n_test`` act as optional **caps** on the split sizes (``None`` = use the full
    partition). The result is a pure function of (config, fold/seed), per §5.
    """
    dataset = task_cfg["dataset"]
    standardize = bool(task_cfg.get("standardize", True))
    n_folds = task_cfg.get("n_folds")

    X, Y = load_multilabel_pool(dataset)

    if bool(task_cfg.get("pad_to_label_multiple", False)):
        X = _pad_features_to_label_multiple(X, Y.shape[1])

    n = X.shape[0]

    if n_folds:
        cv_seed = int(task_cfg.get("cv_seed", 0))
        which = split_seed if fold is None else fold
        train_idx, test_idx = _kfold_indices(n, int(n_folds), int(which), cv_seed)
    else:
        train_frac = float(task_cfg.get("train_frac", 0.7))
        perm = np.random.default_rng(split_seed).permutation(n)
        n_tr = int(round(train_frac * n))
        # Guard the degenerate ends so both splits are non-empty regardless of train_frac.
        n_tr = max(1, min(n - 1, n_tr))
        train_idx, test_idx = perm[:n_tr], perm[n_tr:]

    if n_train is not None:
        train_idx = train_idx[:n_train]
    if n_test is not None:
        test_idx = test_idx[:n_test]

    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xte, Yte = X[test_idx], Y[test_idx]

    if standardize:
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)  # leave constant columns untouched (avoid div-by-zero)
        Xtr = ((Xtr - mu) / sd).astype(np.float32)
        Xte = ((Xte - mu) / sd).astype(np.float32)  # test uses TRAIN stats — no leakage

    return TabularDataset(Xtr, Ytr), TabularDataset(Xte, Yte)


# --- M26: multivariate time-series forecasting (multi-target REGRESSION bridge) ---

def load_forecast_series(dataset: str = "etth1") -> np.ndarray:
    """Load a vendored forecasting series ``(T, M)`` float32 and verify its content hash (§5)."""
    if dataset not in _FORECAST_SHA256:
        raise ValueError(f"Unknown forecast dataset '{dataset}'. Have: {sorted(_FORECAST_SHA256)}")
    path = _CACHE_DIR / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Vendored cache {path} missing. Build it once with "
            "`uv run python scratchpad/fetch_forecast.py`."
        )
    with np.load(path, allow_pickle=True) as npz:
        series = np.ascontiguousarray(npz["series"], dtype=np.float32)
    sha = hashlib.sha256(series.tobytes()).hexdigest()
    if sha != _FORECAST_SHA256[dataset]:
        raise ValueError(
            f"{dataset} content hash mismatch (got {sha}, expected {_FORECAST_SHA256[dataset]}). "
            "The vendored cache changed — refusing to run on non-canonical data (§5)."
        )
    return series


def _forecast_windows(series: np.ndarray, lookback: int, horizon: int):
    """Slide (lookback → horizon) windows over the series.

    Returns ``X (N, M, lookback)`` (each cell = one variable's history — the mixer's variable-cell
    layout) and ``y (N, M, horizon)`` (each cell = that variable's future). Window ``n`` starts at
    series index ``n``: input ``series[n:n+L]``, target ``series[n+L:n+L+H]``; ``N = T-L-H+1``.
    """
    T, M = series.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError(f"lookback+horizon ({lookback}+{horizon}) exceeds series length {T}.")
    # (N, L, M) input / (N, H, M) target via strided views, then move the variable axis to be the
    # CELL axis so X is (N, M, L) and y is (N, M, H) — variable i is cell i on both sides (the
    # shared input/output topology the mixer needs; M25's multi-label reshape LACKED this).
    idx = np.arange(N)[:, None]
    Xin = series[idx + np.arange(lookback)[None, :]]       # (N, L, M)
    Ytg = series[idx + lookback + np.arange(horizon)[None, :]]  # (N, H, M)
    X = np.transpose(Xin, (0, 2, 1))                        # (N, M, L)
    y = np.transpose(Ytg, (0, 2, 1))                        # (N, M, H)
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


def make_forecast_splits(
    task_cfg: dict,
    split_seed: int,
    n_train: int | None = None,
    n_test: int | None = None,
    fold: int | None = None,
) -> tuple[TabularDataset, TabularDataset]:
    """Expanding-window (rolling-origin) backtest split for M-variate forecasting (M26).

    Time series forbid the random K-fold M20 uses (it leaks future→past). Instead we reserve the
    LAST ``test_frac`` of windows as a test region, cut it into ``n_folds`` disjoint CONTIGUOUS
    blocks, and for fold ``k`` (the runner passes the per-run seed): test = block ``k``; train =
    every window strictly BEFORE that block, minus a ``lookback+horizon-1`` PURGE gap so no train
    window's target overlaps a test window's input. So each seed trains only on its block's PAST
    (an honest forecasting protocol) and the ``n_folds`` test blocks are disjoint ⇒ the paired sign
    test is valid across seeds (like M20's disjoint CV folds, but time-ordered — NB train sets are
    nested/overlapping, so treat the binomial p as *indicative*, cf. Dietterich 1998).

    Features/targets are z-scored PER VARIABLE on TRAIN statistics only (the raw series values
    strictly before the test block), broadcast over the lookback/horizon axes — the normalized-MSE
    convention of the MTS benchmark. ``task_cfg`` keys: ``dataset`` ("etth1"|"weather", default
    etth1), ``lookback`` (default 96), ``horizon`` (default 24), ``n_folds`` (default 10),
    ``test_frac`` (default 0.3), ``standardize`` (True). Returns X ``(N, M*lookback)`` float32
    (flattened variable-cell layout), y ``(N, M, horizon)``.
    """
    dataset = task_cfg.get("dataset", "etth1")
    lookback = int(task_cfg.get("lookback", 96))
    horizon = int(task_cfg.get("horizon", 24))
    n_folds = int(task_cfg.get("n_folds", 10))
    test_frac = float(task_cfg.get("test_frac", 0.3))
    standardize = bool(task_cfg.get("standardize", True))

    series = load_forecast_series(dataset)
    X, y = _forecast_windows(series, lookback, horizon)   # (N, M, L), (N, M, H)
    N, M, _ = X.shape

    which = split_seed if fold is None else fold
    k = int(which) % n_folds
    test_region_start = int(round((1.0 - test_frac) * N))
    block = max(1, (N - test_region_start) // n_folds)
    tb0 = test_region_start + k * block                # test block start (also its series index)
    tb1 = N if k == n_folds - 1 else tb0 + block       # last fold absorbs the remainder
    purge = lookback + horizon - 1
    train_end = max(0, tb0 - purge)                    # train windows [0, train_end)

    train_idx = np.arange(0, train_end)
    test_idx = np.arange(tb0, tb1)
    if n_train is not None:
        train_idx = train_idx[-n_train:]               # most-recent n_train windows (recency)
    if n_test is not None:
        test_idx = test_idx[:n_test]

    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]

    if standardize:
        # Per-variable stats from the raw series STRICTLY BEFORE the test block (no leakage).
        train_series = series[:tb0] if tb0 > 0 else series[: lookback + horizon]
        mu = train_series.mean(axis=0).astype(np.float32)      # (M,)
        sd = train_series.std(axis=0).astype(np.float32)
        sd = np.where(sd < 1e-8, 1.0, sd)
        mu_c, sd_c = mu[None, :, None], sd[None, :, None]      # broadcast over (N, M, L/H)
        Xtr = ((Xtr - mu_c) / sd_c).astype(np.float32)
        Xte = ((Xte - mu_c) / sd_c).astype(np.float32)
        ytr = ((ytr - mu_c) / sd_c).astype(np.float32)
        yte = ((yte - mu_c) / sd_c).astype(np.float32)

    # Flatten the input to (N, M*lookback) for the model API (cell i = variable i's lookback block);
    # keep the target (N, M, horizon) so out_features=M cells each forecast `horizon` steps.
    Xtr = Xtr.reshape(len(Xtr), M * lookback)
    Xte = Xte.reshape(len(Xte), M * lookback)
    return TabularDataset(Xtr, ytr), TabularDataset(Xte, yte)
