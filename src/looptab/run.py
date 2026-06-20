"""Single entry point: python -m looptab.run --config <yaml> [--seed <int>]

Trains every `arm` (recurrent variants + matched control) on the task, optionally
swept over one task parameter, across all seeds, and reports Δ between named arms
with variance bands. No lone-recurrent number is ever emitted (CLAUDE.md prime
directive); the deep-supervision ablation is its own arm so the loop and deep
supervision stay unconfounded (§4/§8).
"""

import argparse
import copy
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import ExperimentConfig, ModelConfig
from .data.dataset import make_loaders, make_splits
from .eval.metrics import accuracy, delta_report, exact_match
from .registry import get_model
from .train.loop import train


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _build_model(arm: ModelConfig, in_features: int, num_classes: int):
    kwargs = dict(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dim=arm.hidden_dim,
        latent_dim=arm.latent_dim,
        n_steps=arm.n_steps,
    )
    if arm.name == "trm":
        kwargs["deep_supervision"] = arm.deep_supervision
    return get_model(arm.name, **kwargs)


def run_point(cfg: ExperimentConfig, task_params: dict, seed: int) -> dict:
    """Train every arm for one (sweep-value, seed) point. Returns {label: metrics}."""
    task_cfg = cfg.task

    # I1: vary the *function* across outer seeds (new task_seed per seed) so the
    # variance band reflects function-level variation, not just init+rows. Train and
    # test still share this task_seed within the seed, per §3.
    this_task_seed = task_cfg.task_seed + seed
    train_ds, test_ds = make_splits(
        task=task_cfg.name,
        task_cfg=task_params,
        task_seed=this_task_seed,
        train_sample_seed=task_cfg.train_sample_seed + seed * 100,
        test_sample_seed=task_cfg.test_sample_seed + seed * 100,
        n_train=task_cfg.n_train,
        n_test=task_cfg.n_test,
    )
    train_loader, test_loader = make_loaders(train_ds, test_ds, cfg.train.batch_size)

    X_sample, _ = train_ds[0]
    in_features = int(X_sample.shape[0])
    num_classes = 2  # binary per-bit; multi-output (Task B) head is M1

    device = cfg.train.device
    results = {}
    for arm in cfg.arms:
        # C3: reseed immediately before each arm so model init and the dataloader
        # shuffle stream are identical across arms and independent of arm order.
        torch.manual_seed(seed)
        m = _build_model(arm, in_features, num_classes)
        train(
            m,
            train_loader,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            deep_supervision_weight=arm.deep_supervision_weight,
            device=device,
        )
        results[arm.resolved_label()] = {
            "accuracy": accuracy(m, test_loader, device),
            "exact_match": exact_match(m, test_loader, device),
            "n_params": m.count_params(),
        }
    return results


def _aggregate(per_seed: list[dict], labels: list[str]) -> dict:
    """Mean/std (and per-seed) of each metric for each arm across seeds."""
    out = {}
    for lbl in labels:
        accs = [s[lbl]["accuracy"] for s in per_seed]
        ems = [s[lbl]["exact_match"] for s in per_seed]
        out[lbl] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "exact_match_mean": float(np.mean(ems)),
            "exact_match_std": float(np.std(ems)),
            "accuracy_per_seed": accs,
            "n_params": per_seed[0][lbl]["n_params"],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="override: run a single seed")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw)

    seeds = [args.seed] if args.seed is not None else cfg.seeds
    labels = [a.resolved_label() for a in cfg.arms]

    # Sweep axis: one point if no sweep is declared.
    if cfg.sweep is not None:
        sweep_param = cfg.sweep.param
        sweep_values = cfg.sweep.values
    else:
        sweep_param = None
        sweep_values = [None]

    points = []  # one entry per sweep value
    for sv in sweep_values:
        task_params = copy.deepcopy(cfg.task.params)
        if sweep_param is not None:
            task_params[sweep_param] = sv
        tag = f"{sweep_param}={sv}" if sweep_param else "single"
        print(f"\n=== {tag} ===")

        per_seed = []
        for seed in seeds:
            r = run_point(cfg, task_params, seed)
            r_rec = {"seed": seed, **r}
            per_seed.append(r_rec)
            summary = "  ".join(f"{lbl}={r[lbl]['accuracy']:.3f}" for lbl in labels)
            print(f"  [seed={seed}] {summary}")

        agg = _aggregate(per_seed, labels)
        for lbl in labels:
            a = agg[lbl]
            print(f"  {lbl:>16}: acc {a['accuracy_mean']:.4f} ± {a['accuracy_std']:.4f}")

        deltas = {}
        for a, b in cfg.resolved_deltas():
            rep = delta_report(
                [s[a]["accuracy"] for s in per_seed],
                [s[b]["accuracy"] for s in per_seed],
                label="accuracy",
            )
            deltas[f"{a}-{b}"] = rep
            print(f"  Δ({a} − {b}) = {rep['delta_mean']:+.4f} ± {rep['delta_std']:.4f}")

        points.append({
            "sweep_param": sweep_param,
            "sweep_value": sv,
            "agg": agg,
            "deltas": deltas,
            "seeds": per_seed,
        })

    # --- write run record + the curve artifacts (CSV is the k-vs-accuracy curve) ---
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    tag = Path(args.config).stem

    record = {
        "config": cfg.model_dump(),
        "git_sha": _git_sha(),
        "timestamp": stamp,
        "seeds": seeds,
        "points": points,
    }
    json_path = out_dir / f"{tag}_{stamp}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    csv_path = out_dir / f"{tag}_{stamp}_curve.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep_param", "sweep_value", "arm", "metric", "mean", "std", "n_params"])
        for p in points:
            for lbl in labels:
                a = p["agg"][lbl]
                w.writerow([p["sweep_param"], p["sweep_value"], lbl, "accuracy",
                            a["accuracy_mean"], a["accuracy_std"], a["n_params"]])
                w.writerow([p["sweep_param"], p["sweep_value"], lbl, "exact_match",
                            a["exact_match_mean"], a["exact_match_std"], a["n_params"]])

    print(f"\nResults : {json_path}")
    print(f"Curve   : {csv_path}")
    if cfg.sweep is not None:
        _maybe_plot(points, labels, sweep_param, out_dir / f"{tag}_{stamp}_curve.png")


def _maybe_plot(points, labels, sweep_param, out_path):
    """Render the curve with variance bands if matplotlib is available; else skip."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("Plot    : skipped (matplotlib not installed; CSV holds the curve)")
        return
    xs = [p["sweep_value"] for p in points]
    fig, ax = plt.subplots()
    for lbl in labels:
        means = np.array([p["agg"][lbl]["accuracy_mean"] for p in points])
        stds = np.array([p["agg"][lbl]["accuracy_std"] for p in points])
        ax.plot(xs, means, marker="o", label=lbl)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.2)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("test accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Plot    : {out_path}")


if __name__ == "__main__":
    main()
