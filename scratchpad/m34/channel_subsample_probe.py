"""M34-followup: CHANNEL-SUBSAMPLING probe -- the causal channel-count test flagged as required
across all three review rounds of the M34 milestone (results/log/m34.md).

Every prior "channel-count" comparison in this milestone (etth2 M=7 vs weather M=21 vs
electricity M=321 vs traffic M=862) used FOUR DIFFERENT DATASETS -- channel count was always
perfectly confounded with dataset identity, domain, correlation structure, and sampling
frequency. This script breaks that confound the only way that actually works: draw a random
subset of M channels from the SAME dataset (electricity, 321 real channels) and ask whether a
same-dataset subsample behaves like the FULL dataset (M=321, mixing helpful) or like a real
same-M dataset (etth2, M=7, mixing null/weakly-harmful).

If the subsample tracks channel count (behaves like etth2), that's real evidence for a
channel-count mechanism. If it tracks dataset identity (behaves like full electricity regardless
of M), that's evidence the "reversal" is about electricity/traffic specifically (correlation
structure, domain), not channel count per se.

Scope, given a 2-3hr budget: ONE new point (M=7, subsampled from electricity), the two-arm core
comparison (trm_mixer vs trm_mixer_nomix -- the mixing-sign question), held AMP recipe (15
epochs, no microbatch needed at this width), 5 seeds (matches the exploratory scope already used
for electricity/traffic). Reuses etth2's already budget-matched M=7 widths (hidden=224/latent=96
mixer, hidden=156 nomix) since geometry only depends on M/lookback/horizon, not content.

Writes a proper run-record JSON (git_sha, subsample seed+columns, config, per-seed metrics) --
the process gap flagged in the third review pass.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Windows console falls back to cp1252 under redirection, which can't encode Δ/±/− -- force UTF-8
# (same fix as run.py::main()).
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

from looptab.data.dataset import TabularDataset, make_loaders
from looptab.data.real import _forecast_windows_at, load_forecast_series
from looptab.eval.metrics import evaluate_regression
from looptab.registry import get_model
from looptab.run import _git_sha
from looptab.train.loop import train

DATASET = "electricity"
M_SUB = 64
CHANNEL_SUBSAMPLE_SEED = 123
LOOKBACK, HORIZON = 96, 24
N_FOLDS, TEST_FRAC = 10, 0.3
N_TRAIN, N_TEST = 6000, 1000
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 15
DEVICE = "cuda"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# M=7 widths reuse etth2's already budget-matched values (geometry-only, dataset-content-
# independent). M=64 widths freshly probed inline (both arms converge to hidden=latent=500,
# ratio ~1.01 vs trm_flat(hidden=64), see conversation record).
_WIDTHS_BY_M = {
    7: {
        "trm_mixer": dict(hidden_dim=224, latent_dim=96, token_hidden=8),
        "trm_mixer_nomix": dict(hidden_dim=156, latent_dim=156, token_hidden=8),
    },
    64: {
        "trm_mixer": dict(hidden_dim=500, latent_dim=500, token_hidden=8),
        "trm_mixer_nomix": dict(hidden_dim=500, latent_dim=500, token_hidden=8),
    },
}
ARM_WIDTHS = _WIDTHS_BY_M[M_SUB]


def make_subsampled_splits(series_sub: np.ndarray, fold: int):
    """Mirrors make_forecast_splits's expanding-window backtest logic exactly
    (src/looptab/data/real.py), but starting from an already-subsampled (T, M_sub) series
    instead of calling load_forecast_series(dataset) internally."""
    T, M = series_sub.shape
    N = T - LOOKBACK - HORIZON + 1
    k = fold % N_FOLDS
    test_region_start = int(round((1.0 - TEST_FRAC) * N))
    block = max(1, (N - test_region_start) // N_FOLDS)
    tb0 = test_region_start + k * block
    tb1 = N if k == N_FOLDS - 1 else tb0 + block
    purge = LOOKBACK + HORIZON - 1
    train_end = max(0, tb0 - purge)

    train_idx = np.arange(0, train_end)[-N_TRAIN:]
    test_idx = np.arange(tb0, tb1)[:N_TEST]

    Xtr, ytr = _forecast_windows_at(series_sub, train_idx, LOOKBACK, HORIZON)
    Xte, yte = _forecast_windows_at(series_sub, test_idx, LOOKBACK, HORIZON)

    train_series = series_sub[:tb0] if tb0 > 0 else series_sub[: LOOKBACK + HORIZON]
    mu = train_series.mean(axis=0).astype(np.float32)
    sd = train_series.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd)
    mu_c, sd_c = mu[None, :, None], sd[None, :, None]
    Xtr = ((Xtr - mu_c) / sd_c).astype(np.float32)
    Xte = ((Xte - mu_c) / sd_c).astype(np.float32)
    ytr = ((ytr - mu_c) / sd_c).astype(np.float32)
    yte = ((yte - mu_c) / sd_c).astype(np.float32)

    Xtr = Xtr.reshape(len(Xtr), M * LOOKBACK)
    Xte = Xte.reshape(len(Xte), M * LOOKBACK)
    return TabularDataset(Xtr, ytr), TabularDataset(Xte, yte)


full_series = load_forecast_series(DATASET)
T_full, M_full = full_series.shape
subsample_cols = np.sort(
    np.random.default_rng(CHANNEL_SUBSAMPLE_SEED).choice(M_full, size=M_SUB, replace=False)
)
series_sub = np.ascontiguousarray(full_series[:, subsample_cols])
print(f"{DATASET}: subsampled {M_SUB}/{M_full} channels (seed={CHANNEL_SUBSAMPLE_SEED}): "
      f"{subsample_cols.tolist()}")

record = {
    "git_sha": _git_sha(),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "dataset": DATASET, "m_full": int(M_full), "m_sub": M_SUB,
    "channel_subsample_seed": CHANNEL_SUBSAMPLE_SEED,
    "subsample_columns": subsample_cols.tolist(),
    "lookback": LOOKBACK, "horizon": HORIZON, "n_folds": N_FOLDS, "test_frac": TEST_FRAC,
    "n_train": N_TRAIN, "n_test": N_TEST, "seeds": SEEDS,
    "train_cfg": {"epochs": EPOCHS, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128,
                  "device": DEVICE, "amp": True, "loss_type": "mse"},
    "arm_widths": ARM_WIDTHS,
    "per_seed": [],
}

deltas = []
for seed in SEEDS:
    t0 = time.time()
    train_ds, test_ds = make_subsampled_splits(series_sub, fold=seed)
    train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=128, device=DEVICE)
    X0, _ = train_ds[0]
    in_features = int(X0.shape[0])
    num_classes = int(train_ds.y.shape[-1])
    out_features = int(train_ds.y.shape[-2])

    seed_metrics = {}
    for arm_name, widths in ARM_WIDTHS.items():
        torch.manual_seed(seed)
        m = get_model(
            arm_name, in_features=in_features, num_classes=num_classes, out_features=out_features,
            n_steps=8, n_latent=1, use_rmsnorm=True, deep_supervision=True, **widths,
        )
        train(m, train_loader, epochs=EPOCHS, lr=1e-3, weight_decay=1e-4,
              device=DEVICE, amp=True, loss_type="mse")
        test_metrics = evaluate_regression(m, test_loader, DEVICE)
        seed_metrics[arm_name] = test_metrics["mse"]

    delta = seed_metrics["trm_mixer"] - seed_metrics["trm_mixer_nomix"]
    deltas.append(delta)
    dt = time.time() - t0
    record["per_seed"].append({
        "seed": seed, "mixer_mse": seed_metrics["trm_mixer"],
        "nomix_mse": seed_metrics["trm_mixer_nomix"], "delta_mse": delta, "wall_s": dt,
    })
    print(f"seed={seed} mixer_mse={seed_metrics['trm_mixer']:.4f} "
          f"nomix_mse={seed_metrics['trm_mixer_nomix']:.4f} delta={delta:+.4f} ({dt:.1f}s)")

n = len(deltas)
mean = sum(deltas) / n
std = (sum((d - mean) ** 2 for d in deltas) / (n - 1)) ** 0.5
pos = sum(1 for d in deltas if d > 0)
neg = sum(1 for d in deltas if d < 0)
record["delta_mse_mean"] = mean
record["delta_mse_std"] = std
record["sign_pos"] = pos
record["sign_neg"] = neg

print(f"\n{DATASET} M={M_SUB} subsample: Δ(mixer-nomix) MSE = {mean:+.4f} ± {std:.4f}  sign {pos}/{neg}")
print("Reference points: real etth2 (M=7) amp = +0.0038 (6/4, null); real weather (M=21) amp = "
      "-0.0029 (5/5, null); electricity-full (M=321) amp+microbatch = -0.0271 (4/0, helpful); "
      "traffic-full (M=862) amp+microbatch = -0.0450 (4/0, helpful).")

RESULTS_DIR.mkdir(exist_ok=True)
out_path = RESULTS_DIR / f"m34_channel_subsample_{DATASET}_m{M_SUB}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(record, f, indent=2)
print(f"\nRun record: {out_path}")
