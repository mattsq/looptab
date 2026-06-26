"""Throwaway one-time fetcher for the M20 real-tabular bridge (NOT a runtime dependency).

Downloads the canonical Mulan multi-label datasets `emotions` and `yeast` from OpenML and
vendors them as numpy `.npz` caches under `datasets/`, so the *task path* stays network-free and
deterministic (CLAUDE.md §5; loaded with numpy only in `src/looptab/data/real.py`).

Run once, out of band:
    uv run --with scikit-learn --with scipy --with pandas python scratchpad/fetch_multilabel.py

It prints a CONTENT sha256 = sha256(X.tobytes() + Y.tobytes()) per dataset; paste those into the
`_GOLDEN_SHA256` table in `src/looptab/data/real.py` (the loader recomputes + verifies it). We
hash array *content*, not the `.npz` bytes (the zip container is not byte-stable — timestamps).

Sources (canonical Mulan versions, dense numeric features, binary labels):
    emotions: OpenML data_id 40589 — 593 rows, 72 features, 6 labels
    yeast:    OpenML data_id 40597 — 2417 rows, 103 features, 14 labels
"""

import hashlib
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

# OpenML data ids for the canonical Mulan multi-label versions (see module docstring).
#   scene: OpenML 40595 — 2407 rows, 294 features, 6 labels (near-mutually-exclusive image scenes).
DATASETS = {"emotions": 40589, "yeast": 40597, "scene": 40595}

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets"


def content_sha256(X: np.ndarray, Y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(Y).tobytes())
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, did in DATASETS.items():
        d = fetch_openml(data_id=did, as_frame=True, parser="auto")
        # fetch_openml splits features (d.data) from the multi-label targets (d.target) using the
        # dataset's declared target_names, so no manual column slicing is needed.
        X = np.ascontiguousarray(d.data.to_numpy(dtype=np.float32))
        # Targets come back as pandas 'category' columns with string categories that differ by
        # dataset ('0'/'1' for some, 'TRUE'/'FALSE' for others). Map robustly to {0,1}.
        truthy = {"1", "true", "yes", "t", "y"}

        def to01(col):
            s = col.astype(str).str.strip().str.lower()
            uniq = set(s.unique())
            assert len(uniq) <= 2, f"{name}: label col {col.name} has values {uniq}"
            return s.isin(truthy).astype(int)

        Y = np.ascontiguousarray(d.target.apply(to01).to_numpy(dtype=np.int64))
        assert not np.isnan(X).any(), f"{name}: NaNs in features"
        assert set(np.unique(Y).tolist()) <= {0, 1}, f"{name}: non-binary labels"
        feat = np.array(list(d.data.columns), dtype=object)
        lab = np.array(list(d.target.columns), dtype=object)
        sha = content_sha256(X, Y)
        np.savez(
            OUT_DIR / f"{name}.npz",
            X=X,
            Y=Y,
            feature_names=feat,
            label_names=lab,
            content_sha256=np.array(sha),
        )
        card = float(Y.sum(1).mean())
        print(
            f"{name:9s} id={did} X{X.shape} Y{Y.shape} "
            f"labelcard={card:.3f} allzero_rows={int((Y.sum(1) == 0).sum())}"
        )
        print(f"  content_sha256 = {sha}")


if __name__ == "__main__":
    main()
