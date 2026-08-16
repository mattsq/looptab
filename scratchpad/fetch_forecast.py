"""Throwaway one-time fetcher for the M26 multivariate-time-series bridge (NOT a runtime dep).

Downloads the canonical long-term MTS forecasting benchmarks and vendors each as a numpy `.npz`
under `datasets/`, so the *task path* stays network-free and deterministic (CLAUDE.md §5; loaded
with numpy only in `src/looptab/data/real.py`).

Run once, out of band:
    uv run python scratchpad/fetch_forecast.py

Sources (raw CSVs; the date column is dropped, the remaining columns are the variates in order):
    etth1       : ETDataset (Zhou 2021, Informer) — 17420 hourly rows, 7 vars (6 loads + OT).
    etth2       : ETDataset, second transformer — 17420 hourly rows, 7 vars, same schema as etth1.
    ettm1       : ETDataset, 15-min sampling     — 69680 rows, 7 vars, same schema as etth1.
    ettm2       : ETDataset, 15-min sampling     — 69680 rows, 7 vars, same schema as etth1.
    weather     : Autoformer benchmark (Wu 2021)  — 52696 10-min rows, 21 meteorological vars.
    electricity : Autoformer benchmark (Wu 2021)  — 26304 hourly rows, 321 client load vars.
    traffic     : Autoformer benchmark (Wu 2021)  — 17544 hourly rows, 862 sensor occupancy vars.

electricity/traffic give the M34 forecasting-breadth milestone a wide channel-count spread
(7 -> 21 -> 321 -> 862) to turn M32's two-point channel-independence-share observation into a
proper dose-response curve (CLAUDE.md §11.2 #14).

Prints a CONTENT sha256 = sha256(series.tobytes()) per dataset; paste into `_FORECAST_SHA256` in
`src/looptab/data/real.py` (the loader recomputes + verifies it). We hash array *content*, not the
`.npz` bytes (the zip container is not byte-stable — timestamps).
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

_ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
_TSL_BASE = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"

SOURCES = {
    "etth1": _ETT_BASE + "ETTh1.csv",
    "etth2": _ETT_BASE + "ETTh2.csv",
    "ettm1": _ETT_BASE + "ETTm1.csv",
    "ettm2": _ETT_BASE + "ETTm2.csv",
    "weather": _TSL_BASE + "weather/weather.csv",
    "electricity": _TSL_BASE + "electricity/electricity.csv",
    "traffic": _TSL_BASE + "traffic/traffic.csv",
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
