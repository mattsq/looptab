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
from typing import Optional

import numpy as np
import torch
import yaml

from .config import ExperimentConfig, ModelConfig
from .data.dataset import make_loaders, make_splits
from .eval.metrics import accuracy, delta_report, exact_match, majority_baseline
from .registry import get_model
from .train.loop import train


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _build_model(
    arm: ModelConfig,
    in_features: int,
    num_classes: int,
    out_features: Optional[int] = None,
):
    kwargs = dict(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dim=arm.hidden_dim,
        latent_dim=arm.latent_dim,
        n_steps=arm.n_steps,
        out_features=out_features,
    )
    if arm.name == "trm":
        kwargs["deep_supervision"] = arm.deep_supervision
    return get_model(arm.name, **kwargs)


def run_point(cfg: ExperimentConfig, task_params: dict, seed: int) -> tuple[dict, dict, float]:
    """Train every arm for one (sweep-value, seed) point. Returns (results, models)."""
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

    # Exact-match is only a distinct metric for multi-output targets (§3). For
    # single-output tasks it equals accuracy, so we don't report it (avoids a
    # redundant CSV row that reads like an independent signal).
    multi_output = train_ds.y.ndim > 1
    out_features = int(train_ds.y.shape[-1]) if multi_output else None

    device = cfg.train.device
    results = {}
    models = {}
    for arm in cfg.arms:
        # C3: reseed immediately before each arm so model init and the dataloader
        # shuffle stream are identical across arms and independent of arm order.
        torch.manual_seed(seed)
        m = _build_model(arm, in_features, num_classes, out_features)
        train(
            m,
            train_loader,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            deep_supervision_weight=arm.deep_supervision_weight,
            device=device,
        )
        metrics = {
            "accuracy": accuracy(m, test_loader, device),
            "n_params": m.count_params(),
        }
        if multi_output:
            metrics["exact_match"] = exact_match(m, test_loader, device)
        results[arm.resolved_label()] = metrics
        models[arm.resolved_label()] = m

    return results, models, majority_baseline(test_loader)


def _std(xs: list[float]) -> float:
    """Sample std (ddof=1); falls back to 0 for a single seed."""
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def _aggregate(per_seed: list[dict], labels: list[str]) -> dict:
    """Mean/std (and per-seed) of each metric for each arm across seeds."""
    out = {}
    for lbl in labels:
        accs = [s[lbl]["accuracy"] for s in per_seed]
        stats = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": _std(accs),
            "accuracy_per_seed": accs,
            "n_params": per_seed[0][lbl]["n_params"],
        }
        if "exact_match" in per_seed[0][lbl]:
            ems = [s[lbl]["exact_match"] for s in per_seed]
            stats["exact_match_mean"] = float(np.mean(ems))
            stats["exact_match_std"] = _std(ems)
        out[lbl] = stats

    if "baseline" in per_seed[0]:
        baselines = [s["baseline"] for s in per_seed]
        out["baseline"] = {
            "mean": float(np.mean(baselines)),
            "std": _std(baselines),
            "per_seed": baselines,
        }
    return out


def run_extrapolation_point(
    cfg: ExperimentConfig,
    task_params: dict,
    seed: int,
    models: dict,
    T_test: int,
    R_test_values: list[int],
    device: str,
) -> tuple[dict, float]:
    """Evaluate trained models on a task with CA step length T_test, varying R_test."""
    task_cfg = cfg.task
    this_task_seed = task_cfg.task_seed + seed

    # Generate test split with T_test
    test_params = copy.deepcopy(task_params)
    test_params["T"] = T_test

    _, test_ds = make_splits(
        task=task_cfg.name,
        task_cfg=test_params,
        task_seed=this_task_seed,
        train_sample_seed=task_cfg.train_sample_seed + seed * 100,
        test_sample_seed=task_cfg.test_sample_seed + seed * 100,
        n_train=task_cfg.n_train,
        n_test=task_cfg.n_test,
    )
    _, test_loader = make_loaders(test_ds, test_ds, cfg.train.batch_size)
    multi_output = test_ds.y.ndim > 1

    point_results = {}
    for arm in cfg.arms:
        lbl = arm.resolved_label()
        m = models[lbl]

        # recurrent arm (TRM) can be unrolled to different steps R'
        if arm.name == "trm":
            for R in R_test_values:
                metrics = {
                    "accuracy": accuracy(m, test_loader, device, n_steps=R),
                }
                if multi_output:
                    metrics["exact_match"] = exact_match(m, test_loader, device, n_steps=R)
                point_results[(lbl, R)] = metrics
        else:
            # Control arm (feedforward) is not recurrent, so R' doesn't apply to it.
            # We evaluate it once, and copy results for all R' to make plotting/CSV easier.
            metrics = {
                "accuracy": accuracy(m, test_loader, device),
            }
            if multi_output:
                metrics["exact_match"] = exact_match(m, test_loader, device)
            for R in R_test_values:
                point_results[(lbl, R)] = metrics

    return point_results, majority_baseline(test_loader)


def _aggregate_extrapolation(
    all_seed_extrap: list[tuple[dict, float]],
    labels: list[str],
    T_values: list[int],
    R_values: list[int],
) -> dict:
    agg = {}
    for T in T_values:
        agg[T] = {}
        baselines = [seed_data[T][1] for seed_data in all_seed_extrap]
        agg[T]["baseline_mean"] = float(np.mean(baselines))
        agg[T]["baseline_std"] = _std(baselines)
        for R in R_values:
            agg[T][R] = {}
            for lbl in labels:
                accs = [seed_data[T][0][(lbl, R)]["accuracy"] for seed_data in all_seed_extrap]
                stats = {
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": _std(accs),
                }
                if "exact_match" in all_seed_extrap[0][T][0][(lbl, R)]:
                    ems = [
                        seed_data[T][0][(lbl, R)]["exact_match"] for seed_data in all_seed_extrap
                    ]
                    stats["exact_match_mean"] = float(np.mean(ems))
                    stats["exact_match_std"] = _std(ems)
                agg[T][R][lbl] = stats
    return agg


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
    extrap_results = None

    for sv in sweep_values:
        task_params = copy.deepcopy(cfg.task.params)
        if sweep_param is not None:
            task_params[sweep_param] = sv
        tag = f"{sweep_param}={sv}" if sweep_param else "single"
        print(f"\n=== {tag} ===")

        per_seed = []
        all_seed_extrap = []
        for seed in seeds:
            r, models, baseline = run_point(cfg, task_params, seed)
            r_rec = {"seed": seed, "baseline": baseline, **r}
            per_seed.append(r_rec)
            summary = "  ".join(f"{lbl}={r[lbl]['accuracy']:.3f}" for lbl in labels)
            print(f"  [seed={seed}] {summary} (baseline={baseline:.3f})")

            # Run extrapolation for this seed if configured
            if cfg.extrapolation is not None:
                seed_extrap = {}
                for T_test in cfg.extrapolation.T_values:
                    seed_extrap[T_test] = run_extrapolation_point(
                        cfg,
                        task_params,
                        seed,
                        models,
                        T_test,
                        cfg.extrapolation.R_values,
                        cfg.train.device,
                    )
                all_seed_extrap.append(seed_extrap)

        agg = _aggregate(per_seed, labels)
        print(
            f"  {'majority_baseline':>16}: "
            f"acc {agg['baseline']['mean']:.4f} ± {agg['baseline']['std']:.4f}"
        )
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

        # Aggregate extrapolation results across seeds
        if cfg.extrapolation is not None:
            extrap_results = _aggregate_extrapolation(
                all_seed_extrap,
                labels,
                cfg.extrapolation.T_values,
                cfg.extrapolation.R_values,
            )
            print("\n=== Depth Extrapolation Summary ===")
            for T in cfg.extrapolation.T_values:
                base_mean = extrap_results[T]["baseline_mean"]
                base_std = extrap_results[T]["baseline_std"]
                print(f"  Task T={T} (baseline: {base_mean:.3f}±{base_std:.3f}):")
                for R in cfg.extrapolation.R_values:
                    arm_summaries = []
                    for lbl in labels:
                        acc_mean = extrap_results[T][R][lbl]["accuracy_mean"]
                        acc_std = extrap_results[T][R][lbl]["accuracy_std"]
                        summary_str = f"{lbl}(R'={R})={acc_mean:.3f}±{acc_std:.3f}"
                        if "exact_match_mean" in extrap_results[T][R][lbl]:
                            em_mean = extrap_results[T][R][lbl]["exact_match_mean"]
                            em_std = extrap_results[T][R][lbl]["exact_match_std"]
                            summary_str += f" [EM:{em_mean:.3f}±{em_std:.3f}]"
                        arm_summaries.append(summary_str)
                    print(f"    Unroll R'={R:2d}:  " + "  ".join(arm_summaries))

        points.append(
            {
                "sweep_param": sweep_param,
                "sweep_value": sv,
                "agg": agg,
                "deltas": deltas,
                "seeds": per_seed,
            }
        )

    # --- write run record + the curve artifacts ---
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
    if extrap_results is not None:
        record["extrapolation"] = extrap_results

    json_path = out_dir / f"{tag}_{stamp}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    csv_path = out_dir / f"{tag}_{stamp}_curve.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep_param", "sweep_value", "arm", "metric", "mean", "std", "n_params"])
        for p in points:
            # Save baseline entry to the curve CSV
            w.writerow(
                [
                    p["sweep_param"],
                    p["sweep_value"],
                    "baseline",
                    "accuracy",
                    p["agg"]["baseline"]["mean"],
                    p["agg"]["baseline"]["std"],
                    0,
                ]
            )
            for lbl in labels:
                a = p["agg"][lbl]
                w.writerow(
                    [
                        p["sweep_param"],
                        p["sweep_value"],
                        lbl,
                        "accuracy",
                        a["accuracy_mean"],
                        a["accuracy_std"],
                        a["n_params"],
                    ]
                )
                if "exact_match_mean" in a:
                    w.writerow(
                        [
                            p["sweep_param"],
                            p["sweep_value"],
                            lbl,
                            "exact_match",
                            a["exact_match_mean"],
                            a["exact_match_std"],
                            a["n_params"],
                        ]
                    )

    print(f"\nResults : {json_path}")
    print(f"Curve   : {csv_path}")
    if cfg.sweep is not None:
        _maybe_plot(points, labels, sweep_param, out_dir / f"{tag}_{stamp}_curve.png")

    if extrap_results is not None:
        extrap_csv_path = out_dir / f"{tag}_{stamp}_extrapolation.csv"
        with open(extrap_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["T_test", "R_test", "arm", "metric", "mean", "std"])
            for T in cfg.extrapolation.T_values:
                # Write baseline row for this T_test
                w.writerow(
                    [
                        T,
                        0,
                        "baseline",
                        "accuracy",
                        extrap_results[T]["baseline_mean"],
                        extrap_results[T]["baseline_std"],
                    ]
                )
                for R in cfg.extrapolation.R_values:
                    for lbl in labels:
                        stats = extrap_results[T][R][lbl]
                        w.writerow(
                            [
                                T,
                                R,
                                lbl,
                                "accuracy",
                                stats["accuracy_mean"],
                                stats["accuracy_std"],
                            ]
                        )
                        if "exact_match_mean" in stats:
                            w.writerow(
                                [
                                    T,
                                    R,
                                    lbl,
                                    "exact_match",
                                    stats["exact_match_mean"],
                                    stats["exact_match_std"],
                                ]
                            )
        print(f"Extrap  : {extrap_csv_path}")

        _maybe_plot_extrapolation(
            extrap_results,
            labels,
            cfg.extrapolation.R_values,
            cfg.extrapolation.T_values,
            out_dir / f"{tag}_{stamp}_extrapolation.png",
        )


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
    # Draw the majority-class baseline so a degenerate sweep point is obvious at a
    # glance (parity with the extrapolation plot, which already shows it).
    if "baseline" in points[0]["agg"]:
        base = np.array([p["agg"]["baseline"]["mean"] for p in points])
        ax.plot(xs, base, color="gray", linestyle="--", label="majority baseline")
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("test accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot    : {out_path}")


def _maybe_plot_extrapolation(extrap_results, labels, R_values, T_values, out_path):
    """Render extrapolation plots if matplotlib is available; else skip."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("Extrap Plot: skipped (matplotlib not installed; CSV holds the curve)")
        return

    fig, axes = plt.subplots(len(T_values), 1, figsize=(8, 4 * len(T_values)), sharex=False)
    if len(T_values) == 1:
        axes = [axes]

    for idx, T in enumerate(T_values):
        ax = axes[idx]
        for lbl in labels:
            means = []
            stds = []
            for R in R_values:
                stats = extrap_results[T][R][lbl]
                means.append(stats["accuracy_mean"])
                stds.append(stats["accuracy_std"])
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(R_values, means, marker="o", label=lbl)
            ax.fill_between(R_values, means - stds, means + stds, alpha=0.2)

        # Plot majority baseline as a horizontal line
        ax.axhline(
            extrap_results[T]["baseline_mean"],
            color="gray",
            linestyle="--",
            label="majority baseline",
        )
        ax.set_title(f"Evaluation on T={T} steps of CA")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Test Unroll Steps (R')")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Extrap Plot: {out_path}")


if __name__ == "__main__":
    main()
