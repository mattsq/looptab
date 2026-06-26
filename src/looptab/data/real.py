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

    ``n_train`` / ``n_test`` act as optional **caps** on the split sizes (``None`` = use the full
    partition). The result is a pure function of (config, fold/seed), per §5.
    """
    dataset = task_cfg["dataset"]
    standardize = bool(task_cfg.get("standardize", True))
    n_folds = task_cfg.get("n_folds")

    X, Y = load_multilabel_pool(dataset)
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
