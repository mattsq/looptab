"""M34-followup: multi-init-at-FIXED-fold probe for traffic's `ff_matched` (post-review,
CLAUDE.md #19 THIRD correction pass).

The FF-convergence probe found `ff_matched` on `traffic` has high seed-to-seed variance in train
MSE (0.31-0.77 at 45 epochs) that `nomix`/`trm_flat` don't share, but `run_point`'s `seed`
simultaneously sets `torch.manual_seed(seed)` (model init) AND the chronological test fold
(`fold = seed % n_folds`), so the mechanism (init-sensitivity vs fold-difficulty vs an
architecture x time-period interaction) could not be isolated there.

v1 of this script (results now superseded) fixed the fold but did NOT isolate initialization: it
called `torch.manual_seed(init_seed)` once and then let `train()` run, but `InMemoryLoader.__iter__`
(`data/dataset.py`) draws its per-epoch shuffle permutation from the GLOBAL torch RNG on every
epoch -- so varying `init_seed` varied BOTH the model's initial weights AND the entire minibatch
order for all 45 epochs. External review caught this. **Fix: reset the global RNG to a FIXED,
init-independent value immediately after model construction, before `train()` is called**, so
every run replays an IDENTICAL shuffle sequence and only the model's initial weights differ.

v1 also (a) only tested `ff_matched`, so "the sensitivity is specific to ff_matched" rested on
comparing this arm's fixed-fold multi-init spread to OTHER arms' different-fold single-init
spread -- not a matched comparison; this version adds a `trm_flat` control under the identical
fixed-fold multi-init protocol. (b) only printed to stdout, violating the repo's run-record
invariant (CLAUDE.md §5.7); this version writes a JSON record with git_sha/config/per-init
metrics to `results/`, like every other run in this repo.

Still cheap: both arms are narrow (hidden=64) and train in under a minute per init.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from looptab.data.dataset import make_loaders
from looptab.data.real import make_forecast_splits
from looptab.eval.metrics import evaluate_regression
from looptab.registry import get_model
from looptab.run import _git_sha
from looptab.train.loop import train

FOLD = 1  # fixed -- the empirically hardest fold seen so far for ff_matched on traffic
INIT_SEEDS = [100, 101, 102, 103, 104]  # arbitrary, deliberately far from any fold/seed value
LOADER_RESET_SEED = 999  # fixed global-RNG state for ALL loader shuffling, independent of init
EPOCHS = 45
DEVICE = "cuda"
ARMS = ["ff_matched", "trm"]  # "trm" (labelled trm_flat in configs) = control: ff_matched-specific?
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

task_cfg = dict(dataset="traffic", lookback=96, horizon=24, n_folds=10, test_frac=0.3)
train_ds, test_ds = make_forecast_splits(
    task_cfg, split_seed=0, n_train=6000, n_test=1000, fold=FOLD
)
train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=128, device=DEVICE)

X0, y0 = train_ds[0]
in_features = int(X0.shape[0])
num_classes = int(train_ds.y.shape[-1])   # horizon
out_features = int(train_ds.y.shape[-2])  # M variables (cells)

print(f"fold={FOLD} in_features={in_features} num_classes={num_classes} out_features={out_features}")

record = {
    "git_sha": _git_sha(),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "task_cfg": task_cfg,
    "fold": FOLD,
    "n_train": 6000,
    "n_test": 1000,
    "init_seeds": INIT_SEEDS,
    "loader_reset_seed": LOADER_RESET_SEED,
    "train_cfg": {
        "epochs": EPOCHS, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128,
        "device": DEVICE, "amp": True, "loss_type": "mse",
        "hidden_dim": 64, "latent_dim": 64, "n_steps": 8,
    },
    "arms": {},
}

for arm_name in ARMS:
    print(f"\n=== {arm_name} ===")
    print(f"{'init_seed':>10} {'train_mse':>10} {'test_mse':>10} {'wall_s':>8}")
    results = []
    for init_seed in INIT_SEEDS:
        t0 = time.time()
        torch.manual_seed(init_seed)
        extra_kwargs = (
            {"use_rmsnorm": True, "n_latent": 1, "deep_supervision": True}
            if arm_name == "trm" else {}
        )
        m = get_model(
            arm_name,
            in_features=in_features,
            num_classes=num_classes,
            out_features=out_features,
            hidden_dim=64,
            latent_dim=64,
            n_steps=8,
            **extra_kwargs,
        )
        # Isolate initialization from minibatch order: reset the global RNG to a FIXED value
        # (independent of init_seed) right before training, so every run's per-epoch shuffle
        # sequence is IDENTICAL and only the model's starting weights differ.
        torch.manual_seed(LOADER_RESET_SEED)
        train(
            m, train_loader, epochs=EPOCHS, lr=1e-3, weight_decay=1e-4,
            device=DEVICE, amp=True, loss_type="mse",
        )
        train_metrics = evaluate_regression(m, train_loader, DEVICE)
        test_metrics = evaluate_regression(m, test_loader, DEVICE)
        dt = time.time() - t0
        results.append({
            "init_seed": init_seed,
            "train_mse": train_metrics["mse"],
            "test_mse": test_metrics["mse"],
        })
        print(f"{init_seed:>10} {train_metrics['mse']:>10.4f} {test_metrics['mse']:>10.4f} {dt:>8.1f}")

    train_mses = [r["train_mse"] for r in results]
    test_mses = [r["test_mse"] for r in results]
    n = len(train_mses)
    train_mean = sum(train_mses) / n
    train_std = (sum((v - train_mean) ** 2 for v in train_mses) / (n - 1)) ** 0.5
    test_mean = sum(test_mses) / n
    test_std = (sum((v - test_mean) ** 2 for v in test_mses) / (n - 1)) ** 0.5
    print(f"\n{arm_name}: mean train_mse={train_mean:.4f} std={train_std:.4f} "
          f"range=[{min(train_mses):.4f}, {max(train_mses):.4f}]")
    print(f"{arm_name}: mean test_mse={test_mean:.4f} std={test_std:.4f} "
          f"range=[{min(test_mses):.4f}, {max(test_mses):.4f}]")

    record["arms"][arm_name] = {
        "per_init": results,
        "train_mse_mean": train_mean, "train_mse_std": train_std,
        "train_mse_range": [min(train_mses), max(train_mses)],
        "test_mse_mean": test_mean, "test_mse_std": test_std,
        "test_mse_range": [min(test_mses), max(test_mses)],
    }

RESULTS_DIR.mkdir(exist_ok=True)
out_path = RESULTS_DIR / f"m34_traffic_ff_multiinit_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(record, f, indent=2)
print(f"\nRun record: {out_path}")
