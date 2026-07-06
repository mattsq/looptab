"""Throwaway one-time fetcher for the M26 multivariate-time-series bridge (NOT a runtime dep).

Downloads the canonical long-term MTS forecasting benchmarks and vendors each as a numpy `.npz`
under `datasets/`, so the *task path* stays network-free and deterministic (CLAUDE.md §5; loaded
with numpy only in `src/looptab/data/real.py`).

Run once, out of band:
    uv run python scratchpad/fetch_forecast.py

Sources (raw CSVs; the date column is dropped, the remaining columns are the variates in order):
    etth1   : ETDataset (Zhou 2021, Informer) — 17420 hourly rows, 7 vars (6 loads + oil-temp OT).
    weather : Autoformer benchmark (Wu 2021)  — 52696 10-min rows, 21 meteorological vars.

Prints a CONTENT sha256 = sha256(series.tobytes()) per dataset; paste into `_FORECAST_SHA256` in
`src/looptab/data/real.py` (the loader recomputes + verifies it). We hash array *content*, not the
`.npz` bytes (the zip container is not byte-stable — timestamps).
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

SOURCES = {
    "etth1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "weather": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"
    "weather/weather.csv",
}
OUT_DIR = Path(__file__).resolve().parent.parent / "datasets"


def content_sha256(series: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(series).tobytes()).hexdigest()


def fetch_one(name: str, url: str) -> None:
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (trusted canonical URLs)
        raw = resp.read().decode("utf-8")
    lines = raw.strip().splitlines()
    cols = lines[0].split(",")[1:]  # drop the leading date column; the rest are variates in order
    rows = [[float(p) for p in ln.split(",")[1:]] for ln in lines[1:]]
    series = np.asarray(rows, dtype=np.float32)  # (T, M), chronological
    assert series.shape[1] == len(cols), (series.shape, len(cols))
    path = OUT_DIR / f"{name}.npz"
    np.savez(path, series=series, columns=np.asarray(cols))
    print(f"wrote {path}  shape={series.shape}  content_sha256={content_sha256(series)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        fetch_one(name, url)


if __name__ == "__main__":
    main()
