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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from .config import ExperimentConfig, ModelConfig
from .data.dataset import make_loaders, make_splits, make_trajectory_dataset
from .eval.introspection import run_introspection
from .eval.metrics import (
    accuracy,
    delta_report,
    evaluate,
    evaluate_act,
    evaluate_regression,
    majority_baseline,
    persistence_baseline_mse,
    subset_accuracy_baseline,
)
from .registry import get_model
from .train.loop import (
    train,
    train_act,
    train_curriculum,
    train_deep_supervision,
    train_progressive,
)


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
    n_steps: Optional[int] = None,
):
    # `n_steps` overrides the arm's static depth when the experiment couples depth to a
    # swept task param (M3a `couple_n_steps_to_param`); otherwise the per-arm value stands.
    kwargs = dict(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dim=arm.hidden_dim,
        latent_dim=arm.latent_dim,
        n_steps=arm.n_steps if n_steps is None else n_steps,
        out_features=out_features,
    )
    # The TRM loop and both untied-stack controls (§4b) emit per-step readouts, so deep
    # supervision can be ablated on the same axis for each.
    if arm.name in (
        "trm", "trm_decoupled", "trm_mixer", "untied_stack", "untied_matched",
        "untied_mixer", "untied_mixer_matched",
    ):
        kwargs["deep_supervision"] = arm.deep_supervision
    # M18 ingredients 3 & 4 live in the recurrent core (TRM); pass them only to arms that
    # accept them so controls keep their byte-identical construction. Off by default.
    if arm.name == "trm":
        kwargs["use_rmsnorm"] = arm.use_rmsnorm
        kwargs["n_latent"] = arm.n_latent
        kwargs["use_act"] = arm.use_act  # M23: build the halt head when ACT is enabled (TRM only)
    if arm.name == "trm_mixer":  # M23 re-test: cell-mixing loop shares the rmsnorm/n_latent knobs
        kwargs["use_rmsnorm"] = arm.use_rmsnorm
        kwargs["n_latent"] = arm.n_latent
        kwargs["token_hidden"] = arm.token_hidden
    if arm.name in ("untied_mixer", "untied_mixer_matched"):  # M24e: mixing-matched §4b controls
        kwargs["use_rmsnorm"] = arm.use_rmsnorm
        kwargs["token_hidden"] = arm.token_hidden
    return get_model(arm.name, **kwargs)


def _baselines(loader, *, want_exact_match: bool) -> dict[str, float]:
    out = {"accuracy": majority_baseline(loader)}
    if want_exact_match:
        out["exact_match"] = subset_accuracy_baseline(loader)
    return out


def _regression_baselines(loader, *, lookback: int, n_vars: int) -> dict[str, float]:
    """Persistence (naive-forecast) baseline for M26: predict the last observed value for every
    horizon step. `accuracy`=−mse mirrors the classification baseline so the runner's gap plumbing
    stays uniform; the meaningful figures are mse/mae/r2."""
    b = persistence_baseline_mse(loader, lookback=lookback, n_vars=n_vars)
    return {"accuracy": -b["mse"], "mse": b["mse"], "mae": b["mae"], "r2": b["r2"]}


def run_point(cfg: ExperimentConfig, task_params: dict, seed: int) -> tuple[dict, dict, dict]:
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
        seed=seed,
    )
    train_loader, test_loader = make_loaders(train_ds, test_ds, cfg.train.batch_size)

    X_sample, _ = train_ds[0]
    in_features = int(X_sample.shape[0])

    # M26 forecasting REGRESSION: targets are (N, M, H) float — M variable-cells each predicting an
    # H-step horizon. The readout width per cell = the horizon (num_classes=H, no softmax), cells =
    # M variables. Detected by objective so every classification task stays byte-identical.
    regression = getattr(task_cfg, "objective", "classification") == "regression"
    if regression:
        num_classes = int(train_ds.y.shape[-1])       # horizon H = readout width per cell
        out_features = int(train_ds.y.shape[-2])      # M variables = cells
        multi_output = True
    else:
        # The binary suite (M0–M22) is 2-class per cell; Sudoku (M23) is `size`-class. Infer the
        # class count from the training targets (max label + 1), clamped to ≥2. Every binary task
        # has both classes present in every column, so this is exactly 2 there → model construction
        # and all committed results stay bit-identical; multi-class tasks (sudoku) get their `size`.
        num_classes = max(2, int(train_ds.y.max()) + 1)
        # Exact-match is only a distinct metric for multi-output targets (§3). For single-output
        # tasks it equals accuracy, so we don't report it (avoids a redundant CSV row that reads
        # like an independent signal).
        multi_output = train_ds.y.ndim > 1
        out_features = int(train_ds.y.shape[-1]) if multi_output else None

    # M3a: optionally couple every arm's unroll depth to a swept task param (e.g. T), so
    # one config sweeps depth with the loop's n_steps tracking the CA step count.
    coupled_steps = None
    if cfg.couple_n_steps_to_param is not None:
        coupled_steps = int(task_params[cfg.couple_n_steps_to_param])

    # M3b: a curriculum trains across a range of CA depths instead of one fixed T. Build the
    # trajectory training set once at T_max; every arm is unrolled to a per-batch depth and
    # (for the loop) supervised step-aligned against the intermediate states. Models are built
    # at depth T_max — the deepest unroll they need and the reference eval depth.
    curriculum = cfg.curriculum
    traj_loader = None
    if curriculum is not None:
        coupled_steps = curriculum.T_max
        traj_ds = make_trajectory_dataset(
            task=task_cfg.name,
            task_cfg=task_params,
            task_seed=this_task_seed,
            sample_seed=task_cfg.train_sample_seed + seed * 100,
            n=task_cfg.n_train,
            T_max=curriculum.T_max,
        )
        traj_loader, _ = make_loaders(traj_ds, traj_ds, cfg.train.batch_size)

    device = cfg.train.device
    results = {}
    models = {}
    for arm in cfg.arms:
        # C3: reseed immediately before each arm so model init and the dataloader
        # shuffle stream are identical across arms and independent of arm order.
        torch.manual_seed(seed)
        m = _build_model(arm, in_features, num_classes, out_features, n_steps=coupled_steps)
        if regression and (curriculum is not None or arm.use_act or arm.n_sup > 1):
            # M26 regression uses the standard MSE train path only; the curriculum/ACT/N_sup
            # routines are CA-trajectory / classification mechanisms (they build CE losses and
            # exact-match halt targets). Fail loudly rather than silently train on the wrong loss.
            raise ValueError(
                f"arm '{arm.resolved_label()}': regression (objective=regression) supports only "
                "standard train path — curriculum / use_act / n_sup>1 are classification routines."
            )
        if curriculum is not None and arm.n_sup > 1:
            # The N_sup detached-carry routine is a standard-train mechanism; combining it with
            # the trajectory curriculum would conflate two different supervision schemes. Fail
            # loudly rather than silently picking one (CLAUDE.md §5.6, one knob per ablation).
            raise ValueError(
                f"arm '{arm.resolved_label()}' sets n_sup>1, which is incompatible with a "
                "curriculum run (train_deep_supervision is for the standard-train path)."
            )
        if curriculum is not None and arm.ds_mode in ("progressive_final", "progressive_step"):
            # M7: Deep Thinking progressive loss (TRM loop arms only; controls take the
            # standard curriculum path below via their "final" ds_mode).
            train_progressive(
                m,
                traj_loader,
                T_min=curriculum.T_min,
                T_max=curriculum.T_max,
                ds_mode=arm.ds_mode,
                alpha=arm.progressive_alpha,
                epochs=cfg.train.epochs,
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                device=device,
                seed=seed,
            )
        elif curriculum is not None:
            train_curriculum(
                m,
                traj_loader,
                T_min=curriculum.T_min,
                T_max=curriculum.T_max,
                ds_mode=arm.ds_mode,
                deep_supervision_weight=arm.deep_supervision_weight,
                epochs=cfg.train.epochs,
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                device=device,
                seed=seed,
            )
        elif arm.use_act:
            # M23: ACT adaptive-computation deep supervision (the §4 unbuilt TRM ingredient).
            # n_sup is the max segment count; the halt head enables adaptive test-time compute.
            if arm.name != "trm":
                raise ValueError(
                    f"arm '{arm.resolved_label()}' sets use_act=True, but only 'trm' has a "
                    "halt head."
                )
            train_act(
                m,
                train_loader,
                max_segments=arm.n_sup,
                n_steps=coupled_steps,
                epochs=cfg.train.epochs,
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                deep_supervision_weight=arm.deep_supervision_weight,
                halt_weight=arm.halt_weight,
                ema_decay=arm.ema_decay,
                device=device,
            )
        elif arm.n_sup > 1:
            # M18 ingredient 1: canonical TRM deep supervision (N_sup detached-carry passes).
            # Coupled depth (if any) flows in as n_steps so each pass unrolls to the task T.
            if arm.name not in ("trm", "trm_decoupled"):
                # train_deep_supervision needs init_state/return_state (TRM, TRMDecoupled only);
                # the feedforward/untied controls would crash on those kwargs. Fail loudly with a
                # clear message rather than an opaque TypeError (review fix S3).
                raise ValueError(
                    f"arm '{arm.resolved_label()}' (name '{arm.name}') sets n_sup>1, but only "
                    "'trm'/'trm_decoupled' support the detached-carry routine "
                    "(init_state/return_state)."
                )
            train_deep_supervision(
                m,
                train_loader,
                n_sup=arm.n_sup,
                carry=arm.n_sup_carry,
                n_steps=coupled_steps,
                epochs=cfg.train.epochs,
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                deep_supervision_weight=arm.deep_supervision_weight,
                ema_decay=arm.ema_decay,
                device=device,
            )
        else:
            train(
                m,
                train_loader,
                epochs=cfg.train.epochs,
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                deep_supervision_weight=arm.deep_supervision_weight,
                ema_decay=arm.ema_decay,
                loss_type="mse" if regression else "ce",
                device=device,
            )
        # M26 forecasting: MSE/MAE/R² from the raw regression readout (no argmax). `accuracy`
        # mirrors −mse so the generic curve/baseline plumbing stays meaningful; the reported
        # headline metrics are mse/mae/r2. Train "accuracy" here is −train-MSE (same diagnostic
        # role: a big train/test MSE gap = overfit).
        if regression:
            want_f1 = False
            test_metrics = evaluate_regression(m, test_loader, device)
            train_acc = evaluate_regression(m, train_loader, device)["accuracy"]
            metrics = {
                "accuracy": test_metrics["accuracy"],
                "train_accuracy": train_acc,
                "n_params": m.count_params(),
                "mse": test_metrics["mse"],
                "mae": test_metrics["mae"],
                "r2": test_metrics["r2"],
            }
            results[arm.resolved_label()] = metrics
            models[arm.resolved_label()] = m
            continue
        # One forward pass over the test set yields both accuracy and (for multi-output
        # Task B) exact-match; train accuracy is a separate pass. F1 is multilabel-only
        # (M20) so synthetic-task evals stay byte-identical.
        if arm.use_act:
            # M23: ACT arms are evaluated ADAPTIVELY (each example halts when the head says it is
            # solved). Train acc uses the same adaptive prediction for an honest like-for-like.
            want_f1 = False
            test_metrics = evaluate_act(
                m, test_loader, max_segments=arm.n_sup, device=device,
                want_exact_match=multi_output, n_steps=coupled_steps,
            )
            train_acc = evaluate_act(
                m, train_loader, max_segments=arm.n_sup, device=device,
                want_exact_match=False, n_steps=coupled_steps,
            )["accuracy"]
        else:
            want_f1 = task_cfg.name == "multilabel"
            test_metrics = evaluate(
                m, test_loader, device, want_exact_match=multi_output, want_f1=want_f1
            )
            train_acc = accuracy(m, train_loader, device)
        metrics = {
            "accuracy": test_metrics["accuracy"],
            # Train accuracy is the M3a optimization-vs-capacity diagnostic: a loop that
            # fails at high T with *low train acc too* is an optimization failure (Phase 2's
            # step-aligned DS may help), not a capacity verdict against the loop.
            "train_accuracy": train_acc,
            "n_params": m.count_params(),
        }
        if arm.use_act and "avg_segments" in test_metrics:
            metrics["avg_segments"] = test_metrics["avg_segments"]  # adaptive-compute diagnostic
        if multi_output:
            metrics["exact_match"] = test_metrics["exact_match"]
            # M9 coherence diagnostic (whole-row coherence vs raw token-acc); see eval.metrics.
            metrics["coherence_excess"] = test_metrics["coherence_excess"]
            metrics["mean_wrong_per_row"] = test_metrics["mean_wrong_per_row"]
        if want_f1:
            # M20: micro/macro-F1 — the honest co-headline to EM on imbalanced multi-label.
            metrics["micro_f1"] = test_metrics["micro_f1"]
            metrics["macro_f1"] = test_metrics["macro_f1"]
        results[arm.resolved_label()] = metrics
        models[arm.resolved_label()] = m

    # M21: optional measurement-only introspection pass. Runs AFTER training on each trained
    # model over a single fixed batch (the first test batch), so it cannot perturb any metric
    # above; when `diagnostics` is unset the whole block is skipped → byte-identical to before.
    diagnostics = {}
    if cfg.diagnostics is not None and cfg.diagnostics.enabled:
        Xb, yb = next(iter(test_loader))
        Xb = Xb.to(device)
        for arm in cfg.arms:
            lbl = arm.resolved_label()
            diagnostics[lbl] = run_introspection(
                models[lbl],
                (Xb, yb),
                overunroll_factor=cfg.diagnostics.overunroll_factor,
                n_random_inits=cfg.diagnostics.n_random_inits,
                power_iter_steps=cfg.diagnostics.power_iter_steps,
                jac_n_examples=cfg.diagnostics.jac_n_examples,
                seed=seed,
            )

    if regression:
        baselines = _regression_baselines(
            test_loader, lookback=int(task_params.get("lookback", 96)), n_vars=out_features
        )
    else:
        baselines = _baselines(test_loader, want_exact_match=multi_output)
    return results, models, baselines, diagnostics


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
        if "train_accuracy" in per_seed[0][lbl]:
            tr = [s[lbl]["train_accuracy"] for s in per_seed]
            stats["train_accuracy_mean"] = float(np.mean(tr))
            stats["train_accuracy_std"] = _std(tr)
        if "exact_match" in per_seed[0][lbl]:
            ems = [s[lbl]["exact_match"] for s in per_seed]
            stats["exact_match_mean"] = float(np.mean(ems))
            stats["exact_match_std"] = _std(ems)
        if "coherence_excess" in per_seed[0][lbl]:
            ces = [s[lbl]["coherence_excess"] for s in per_seed]
            stats["coherence_excess_mean"] = float(np.mean(ces))
            stats["coherence_excess_std"] = _std(ces)
            mwr = [s[lbl]["mean_wrong_per_row"] for s in per_seed]
            stats["mean_wrong_per_row_mean"] = float(np.mean(mwr))
            stats["mean_wrong_per_row_std"] = _std(mwr)
        for f1k in ("micro_f1", "macro_f1"):  # M20 multilabel F1 (present only for that task)
            if f1k in per_seed[0][lbl]:
                vals = [s[lbl][f1k] for s in per_seed]
                stats[f"{f1k}_mean"] = float(np.mean(vals))
                stats[f"{f1k}_std"] = _std(vals)
        for rk in ("mse", "mae", "r2"):  # M26 forecasting-regression metrics
            if rk in per_seed[0][lbl]:
                vals = [s[lbl][rk] for s in per_seed]
                stats[f"{rk}_mean"] = float(np.mean(vals))
                stats[f"{rk}_std"] = _std(vals)
        if "avg_segments" in per_seed[0][lbl]:  # M23 ACT adaptive-compute diagnostic
            segs = [s[lbl]["avg_segments"] for s in per_seed]
            stats["avg_segments_mean"] = float(np.mean(segs))
            stats["avg_segments_std"] = _std(segs)
        out[lbl] = stats

    if "baseline" in per_seed[0]:
        baselines = [s["baseline"] for s in per_seed]
        out["baseline"] = {
            "mean": float(np.mean(baselines)),
            "std": _std(baselines),
            "per_seed": baselines,
        }
    if "baselines" in per_seed[0]:
        out["baselines"] = {}
        for metric in per_seed[0]["baselines"]:
            vals = [s["baselines"][metric] for s in per_seed]
            out["baselines"][metric] = {
                "mean": float(np.mean(vals)),
                "std": _std(vals),
                "per_seed": vals,
            }
    return out


def _aggregate_diagnostics(per_seed: list[dict], labels: list[str]) -> dict:
    """Mean/std (+ per-seed) of every *scalar* introspection metric, per arm across seeds (M21).

    List-valued trajectories (e.g. the residual curve) are kept per-seed in the JSON record but
    not reduced here — the CSV/summary report the scalar descriptors. Returns ``{}`` when the
    seeds carry no diagnostics (the off path), so callers can treat it as 'no diagnostics ran'.
    """
    if not per_seed or "diagnostics" not in per_seed[0]:
        return {}
    out: dict = {}
    for lbl in labels:
        metrics = {}
        # Union of scalar keys across seeds (recurrent arms carry Some keys controls don't).
        keys = [
            k
            for k, v in per_seed[0]["diagnostics"].get(lbl, {}).items()
            if isinstance(v, (int, float))
        ]
        for k in keys:
            vals = [
                s["diagnostics"][lbl][k]
                for s in per_seed
                if lbl in s.get("diagnostics", {}) and k in s["diagnostics"][lbl]
            ]
            if vals:
                metrics[k] = {"mean": float(np.mean(vals)), "std": _std(vals), "per_seed": vals}
        out[lbl] = metrics
    return out


def run_extrapolation_point(
    cfg: ExperimentConfig,
    task_params: dict,
    seed: int,
    models: dict,
    T_test: int,
    R_test_values: list[int],
    device: str,
) -> tuple[dict, dict]:
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
        seed=seed,
    )
    _, test_loader = make_loaders(test_ds, test_ds, cfg.train.batch_size)
    multi_output = test_ds.y.ndim > 1

    point_results = {}
    for arm in cfg.arms:
        lbl = arm.resolved_label()
        m = models[lbl]

        # recurrent arms (TRM and the decoupled variant) can be unrolled to different steps R'
        if arm.name in ("trm", "trm_decoupled"):
            for R in R_test_values:
                point_results[(lbl, R)] = evaluate(
                    m, test_loader, device, want_exact_match=multi_output, n_steps=R
                )
        else:
            # Control arm (feedforward) is not recurrent, so R' doesn't apply to it.
            # We evaluate it once, and copy results for all R' to make plotting/CSV easier.
            metrics = evaluate(m, test_loader, device, want_exact_match=multi_output)
            for R in R_test_values:
                point_results[(lbl, R)] = metrics

    return point_results, _baselines(test_loader, want_exact_match=multi_output)


def _compute_seed(
    cfg: ExperimentConfig, task_params: dict, seed: int
) -> tuple[dict, Optional[dict]]:
    """All per-seed work for one axis point: the arm sweep plus, if configured, the depth
    extrapolation for that seed. Returns ``(r_rec, seed_extrap)`` — both fully serializable
    (no models), so this is the unit dispatched to worker processes. It is a pure function of
    ``(cfg, task_params, seed)``: ``run_point`` self-reseeds every arm, so parallel execution
    is bit-identical to serial (CLAUDE.md §5.3).
    """
    r, models, baselines, diagnostics = run_point(cfg, task_params, seed)
    r_rec = {"seed": seed, "baseline": baselines["accuracy"], "baselines": baselines, **r}
    if diagnostics:
        r_rec["diagnostics"] = diagnostics
    seed_extrap = None
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
    return r_rec, seed_extrap


def _init_worker(num_threads: Optional[int]) -> None:
    """Pool initializer: pin each worker to ``num_threads`` so workers × intra-op threads
    don't oversubscribe the CPU (the tiny models are fastest at 1 thread; see TrainConfig)."""
    if num_threads is not None:
        torch.set_num_threads(num_threads)


def _compute_seeds(
    cfg: ExperimentConfig, task_params: dict, seeds: list[int]
) -> list[tuple[dict, Optional[dict]]]:
    """Run ``_compute_seed`` for every seed, serially or across a process pool, preserving
    seed order. Parallelism is opt-in (``parallel_workers``) and bit-identical to serial."""
    workers = min(cfg.parallel_workers, len(seeds))
    if workers <= 1 or len(seeds) <= 1:
        return [_compute_seed(cfg, task_params, s) for s in seeds]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(cfg.train.num_threads,),
    ) as ex:
        # map preserves input order, so per-seed records stay aligned with `seeds`.
        return list(ex.map(_compute_seed, [cfg] * len(seeds), [task_params] * len(seeds), seeds))


def _aggregate_extrapolation(
    all_seed_extrap: list[tuple[dict, dict]],
    labels: list[str],
    T_values: list[int],
    R_values: list[int],
    delta_pairs: Optional[list[list[str]]] = None,
) -> dict:
    agg = {}
    for T in T_values:
        agg[T] = {}
        baseline_maps = [seed_data[T][1] for seed_data in all_seed_extrap]
        baselines = [b["accuracy"] for b in baseline_maps]
        agg[T]["baseline_mean"] = float(np.mean(baselines))
        agg[T]["baseline_std"] = _std(baselines)
        agg[T]["baselines"] = {
            metric: {
                "mean": float(np.mean([b[metric] for b in baseline_maps])),
                "std": _std([b[metric] for b in baseline_maps]),
            }
            for metric in baseline_maps[0]
        }
        for R in R_values:
            agg[T][R] = {}
            per_seed_acc = {}  # lbl -> per-seed accuracies, kept so cells can be paired-tested
            for lbl in labels:
                accs = [seed_data[T][0][(lbl, R)]["accuracy"] for seed_data in all_seed_extrap]
                per_seed_acc[lbl] = accs
                stats = {
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": _std(accs),
                    "accuracy_per_seed": accs,
                }
                if "exact_match" in all_seed_extrap[0][T][0][(lbl, R)]:
                    ems = [
                        seed_data[T][0][(lbl, R)]["exact_match"] for seed_data in all_seed_extrap
                    ]
                    stats["exact_match_mean"] = float(np.mean(ems))
                    stats["exact_match_std"] = _std(ems)
                agg[T][R][lbl] = stats
            # Paired Δ + sign test per (T, R') cell, so the extrapolation diagonal (R'=T) —
            # which carries the milestone headline (e.g. the M3b short-horizon stepDS win) —
            # gets the SAME paired significance every other Δ in the run gets, not eyeballed
            # bands. (Review fix M2: per-seed extrapolation data is retained for this.)
            if delta_pairs:
                cell_deltas = {}
                for a, b in delta_pairs:
                    if a in per_seed_acc and b in per_seed_acc:
                        cell_deltas[f"{a}-{b}"] = delta_report(
                            per_seed_acc[a], per_seed_acc[b], label="accuracy"
                        )
                agg[T][R]["_deltas"] = cell_deltas
    return agg


def budget_audit(points: list[dict], labels: list[str], cfg: ExperimentConfig) -> dict:
    """Verify every param-matched arm stays within `budget_tol` of the reference, per cell.

    The M3a confound guard (LOG.md / CLAUDE.md §11): a depth sweep only attributes its Δ to
    *depth* if the parameter budget is held fixed across every arm AND every T. The reference
    arm (the loop) defines the budget; arms listed in `budget_ceiling` (e.g. the ~n_steps×
    `untied_stack`) are deliberately non-matched and exempt. Returns per-cell rows plus a
    `breaches` list — arms that broke tolerance without being a declared ceiling. A breach on
    the loop/ff_matched (the depth-attribution arms) is a blocker; a breach on `untied_matched`
    is the expected width-quantization finding (narrow blocks at high T), surfaced not hidden.
    """
    ref = cfg.budget_reference
    tol = cfg.budget_tol
    ceiling = set(cfg.budget_ceiling)
    rows = []
    breaches = []
    for p in points:
        ref_params = p["agg"][ref]["n_params"]
        for lbl in labels:
            n = p["agg"][lbl]["n_params"]
            ratio = n / ref_params if ref_params else float("nan")
            if lbl == ref:
                role = "reference"
            elif lbl in ceiling:
                role = "ceiling"
            else:
                role = "matched"
            within = role == "ceiling" or abs(ratio - 1.0) <= tol
            rows.append(
                {
                    "config": p["label"],
                    "arm": lbl,
                    "n_params": n,
                    "ref_params": ref_params,
                    "ratio": ratio,
                    "role": role,
                    "within_tol": within,
                }
            )
            if role == "matched" and not within:
                breaches.append((p["label"], lbl, ratio))
    return {"rows": rows, "breaches": breaches, "tol": tol, "reference": ref}


def cv_sign_test_status(task_name: str, task_params: dict, seeds: list[int]) -> tuple[bool, str]:
    """Whether the paired sign test is valid for ONE axis point, plus a human-readable reason.

    Synthetic tasks always qualify: each seed draws a fresh function + rows, so the per-seed Δs are
    independent. Real ``multilabel`` qualifies ONLY under K-fold CV — i.e. ``n_folds`` is set AND
    the selected seeds map to **distinct** folds (``fold = seed % n_folds``, the mapping `run_point`
    → `make_multilabel_splits(fold=seed)` uses), so every per-seed Δ is on a DISJOINT, independent
    test fold. It is suppressed when:
      - there is no ``n_folds`` (legacy random-split mode — successive test sets overlap ~0.30), or
      - two selected seeds share a fold (``seed % n_folds`` collides, e.g. ``n_folds < len(seeds)``)
        then those per-seed Δs are on the SAME/overlapping test data, so a binomial p-value would be
        anti-conservative (the exact non-independence CV was meant to remove; M20 review).
    Computed from the **per-point** ``task_params`` (not the base config) so a sweep/grid that
    overrides ``n_folds`` is honoured.
    """
    if task_name not in ("multilabel", "etth1"):
        return True, "independent (fresh function + rows per seed)"
    # M26 etth1: the expanding-window backtest maps seed → a DISJOINT chronological test block
    # (fold = seed % n_folds), same validity condition as multilabel K-fold — and stronger, since
    # each block's train set is its own past prefix (more independent than K-fold's shared train).
    n_folds = task_params.get("n_folds", 10 if task_name == "etth1" else None)
    if not n_folds:
        return False, "random real-data splits overlap and are not independent"
    folds = [s % int(n_folds) for s in seeds]
    if len(set(folds)) != len(folds):
        return (
            False,
            f"selected seeds map to {len(set(folds))} distinct fold(s) of {len(folds)} seeds "
            f"(seed % n_folds collides; need n_folds ≥ #seeds with distinct residues)",
        )
    kind = "expanding-window backtest blocks" if task_name == "etth1" else "K-fold test sets"
    return (
        True,
        f"DISJOINT {kind} (treat p as indicative, cf. Dietterich 1998)",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="override: run a single seed")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw)

    # Pin CPU threads before any tensor work. For the tiny models here, torch's default
    # (= core count) oversubscribes and is much slower than 1 thread, especially on
    # many-core cloud machines (see TrainConfig.num_threads). Bit-identical to the default.
    if cfg.train.num_threads is not None:
        torch.set_num_threads(cfg.train.num_threads)

    seeds = [args.seed] if args.seed is not None else cfg.seeds
    labels = [a.resolved_label() for a in cfg.arms]

    # Outer axis: a single point, a 1-D `sweep` curve, or an N-D `grid` of configs.
    # Each entry is (human label, task-param overrides) — see ExperimentConfig.axis_points.
    axis_points = cfg.axis_points()

    points = []  # one entry per axis point
    extrap_results = None

    for point_label, overrides in axis_points:
        task_params = copy.deepcopy(cfg.task.params)
        task_params.update(overrides)
        # Gate the paired sign test PER POINT, from the actual (possibly grid-overridden) params and
        # the selected seeds: only emit binomial p-values when every seed lands on a distinct,
        # disjoint test fold (see cv_sign_test_status). Otherwise report Δ ± std without a p-value.
        paired_sign_tests, sign_reason = cv_sign_test_status(cfg.task.name, task_params, seeds)
        print(f"\n=== {point_label} ===")

        per_seed = []
        all_seed_extrap = []
        # Seeds are independent (each self-reseeds), so they run serially or across a process
        # pool per `parallel_workers` — same results either way. Printing stays in seed order.
        seed_outputs = _compute_seeds(cfg, task_params, seeds)
        for seed, (r_rec, seed_extrap) in zip(seeds, seed_outputs):
            per_seed.append(r_rec)
            summary = "  ".join(f"{lbl}={r_rec[lbl]['accuracy']:.3f}" for lbl in labels)
            print(f"  [seed={seed}] {summary} (baseline={r_rec['baseline']:.3f})")
            if seed_extrap is not None:
                all_seed_extrap.append(seed_extrap)

        agg = _aggregate(per_seed, labels)
        print(
            f"  {'majority_baseline':>16}: "
            f"acc {agg['baseline']['mean']:.4f} ± {agg['baseline']['std']:.4f}"
        )
        if "exact_match" in agg.get("baselines", {}):
            b = agg["baselines"]["exact_match"]
            print(f"  {'subset_baseline':>16}: EM  {b['mean']:.4f} ± {b['std']:.4f}")
        if cfg.task.name == "multilabel":
            print(f"  sign tests {'over' if paired_sign_tests else 'skipped:'} {sign_reason}")
        for lbl in labels:
            a = agg[lbl]
            print(f"  {lbl:>16}: acc {a['accuracy_mean']:.4f} ± {a['accuracy_std']:.4f}")

        # Exact-match is a distinct, meaningful signal on multi-output tasks (Task B),
        # so we report its paired Δ *with variance* too — not just a point estimate.
        multi_output = "exact_match" in per_seed[0][labels[0]]

        deltas = {}
        for a, b in cfg.resolved_deltas():
            rep = delta_report(
                [s[a]["accuracy"] for s in per_seed],
                [s[b]["accuracy"] for s in per_seed],
                label="accuracy",
                paired_sign_test=paired_sign_tests,
            )
            deltas[f"{a}-{b}"] = {"accuracy": rep}
            line = f"  Δ({a} − {b}) = {rep['delta_mean']:+.4f} ± {rep['delta_std']:.4f}"
            if multi_output:
                em_rep = delta_report(
                    [s[a]["exact_match"] for s in per_seed],
                    [s[b]["exact_match"] for s in per_seed],
                    label="exact_match",
                    paired_sign_test=paired_sign_tests,
                )
                deltas[f"{a}-{b}"]["exact_match"] = em_rep
                line += f"  [EM {em_rep['delta_mean']:+.4f} ± {em_rep['delta_std']:.4f}]"
                # M20: F1 is the honest co-headline on imbalanced multi-label (EM over-rewards
                # modal label-combos). Δ on micro/macro-F1 reported with the same paired stats.
                for f1k, tag in (("micro_f1", "miF1"), ("macro_f1", "maF1")):
                    if f1k in per_seed[0][a]:
                        f1_rep = delta_report(
                            [s[a][f1k] for s in per_seed],
                            [s[b][f1k] for s in per_seed],
                            label=f1k,
                            paired_sign_test=paired_sign_tests,
                        )
                        deltas[f"{a}-{b}"][f1k] = f1_rep
                        line += f"  [{tag} {f1_rep['delta_mean']:+.4f} ± {f1_rep['delta_std']:.4f}]"
                if "coherence_excess" in per_seed[0][a]:
                    ce_rep = delta_report(
                        [s[a]["coherence_excess"] for s in per_seed],
                        [s[b]["coherence_excess"] for s in per_seed],
                        label="coherence_excess",
                        paired_sign_test=paired_sign_tests,
                    )
                    deltas[f"{a}-{b}"]["coherence_excess"] = ce_rep
                    line += f"  [coh {ce_rep['delta_mean']:+.4f} ± {ce_rep['delta_std']:.4f}]"
            # M26: forecasting-regression Δs (lower MSE/MAE is better, so a NEGATIVE Δ favours the
            # first arm — the opposite sign convention to accuracy; the writeup states this).
            for rk, tag in (("mse", "MSE"), ("mae", "MAE"), ("r2", "R2")):
                if rk in per_seed[0][a]:
                    r_rep = delta_report(
                        [s[a][rk] for s in per_seed],
                        [s[b][rk] for s in per_seed],
                        label=rk,
                        paired_sign_test=paired_sign_tests,
                    )
                    deltas[f"{a}-{b}"][rk] = r_rep
                    line += f"  [{tag} {r_rep['delta_mean']:+.4f} ± {r_rep['delta_std']:.4f}]"
            print(line)

        # Aggregate extrapolation results across seeds
        if cfg.extrapolation is not None:
            extrap_results = _aggregate_extrapolation(
                all_seed_extrap,
                labels,
                cfg.extrapolation.T_values,
                cfg.extrapolation.R_values,
                delta_pairs=cfg.resolved_deltas(),
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

        # M21: aggregate the introspection descriptors (no-op when diagnostics are off).
        diag_agg = _aggregate_diagnostics(per_seed, labels)
        if diag_agg:
            print("  --- diagnostics (mean over seeds) ---")
            for lbl in labels:
                dm = diag_agg.get(lbl, {})
                bits = []
                for k in ("spectral_radius_mean", "operator_norm_mean", "za_alignment",
                          "acc_overunroll_drop", "effective_rank", "lipschitz_product"):
                    if k in dm:
                        bits.append(f"{k}={dm[k]['mean']:.3f}")
                if bits:
                    print(f"  {lbl:>16}: " + "  ".join(bits))

        points.append(
            {
                "label": point_label,
                "overrides": overrides,
                "multi_output": multi_output,
                "agg": agg,
                "deltas": deltas,
                "diagnostics": diag_agg,
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
    # Compute the budget audit BEFORE the JSON dump so the confound-guard table is embedded in
    # the canonical run record (§5.7), not only the side-car CSV (review fix m1).
    audit = budget_audit(points, labels, cfg) if cfg.budget_reference is not None else None
    if audit is not None:
        record["budget_audit"] = audit

    json_path = out_dir / f"{tag}_{stamp}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    # Per-arm curve CSV. `config` holds the axis-point label ("single", "k=4",
    # "rule=30, w=9", ...) so the same writer serves a single run, a 1-D sweep and an
    # N-D grid alike.
    csv_path = out_dir / f"{tag}_{stamp}_curve.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "arm", "metric", "mean", "std", "n_params"])
        for p in points:
            w.writerow(
                [
                    p["label"],
                    "baseline",
                    "accuracy",
                    p["agg"]["baseline"]["mean"],
                    p["agg"]["baseline"]["std"],
                    0,
                ]
            )
            if "exact_match" in p["agg"].get("baselines", {}):
                b = p["agg"]["baselines"]["exact_match"]
                w.writerow([p["label"], "baseline", "exact_match", b["mean"], b["std"], 0])
            for rk in ("mse", "mae", "r2"):  # M26 persistence-forecast baseline
                if rk in p["agg"].get("baselines", {}):
                    b = p["agg"]["baselines"][rk]
                    w.writerow([p["label"], "baseline", rk, b["mean"], b["std"], 0])
            for lbl in labels:
                a = p["agg"][lbl]
                w.writerow(
                    [
                        p["label"],
                        lbl,
                        "accuracy",
                        a["accuracy_mean"],
                        a["accuracy_std"],
                        a["n_params"],
                    ]
                )
                if "train_accuracy_mean" in a:
                    w.writerow(
                        [
                            p["label"],
                            lbl,
                            "train_accuracy",
                            a["train_accuracy_mean"],
                            a["train_accuracy_std"],
                            a["n_params"],
                        ]
                    )
                if "exact_match_mean" in a:
                    w.writerow(
                        [
                            p["label"],
                            lbl,
                            "exact_match",
                            a["exact_match_mean"],
                            a["exact_match_std"],
                            a["n_params"],
                        ]
                    )
                for f1k in ("micro_f1", "macro_f1"):  # M20 multilabel F1 co-headline
                    if f"{f1k}_mean" in a:
                        w.writerow(
                            [
                                p["label"],
                                lbl,
                                f1k,
                                a[f"{f1k}_mean"],
                                a[f"{f1k}_std"],
                                a["n_params"],
                            ]
                        )
                for rk in ("mse", "mae", "r2"):  # M26 forecasting-regression metrics
                    if f"{rk}_mean" in a:
                        w.writerow(
                            [p["label"], lbl, rk, a[f"{rk}_mean"], a[f"{rk}_std"], a["n_params"]]
                        )
                if "avg_segments_mean" in a:  # M23 ACT adaptive-compute diagnostic
                    w.writerow(
                        [
                            p["label"],
                            lbl,
                            "avg_segments",
                            a["avg_segments_mean"],
                            a["avg_segments_std"],
                            a["n_params"],
                        ]
                    )
                if "coherence_excess_mean" in a:
                    w.writerow(
                        [
                            p["label"],
                            lbl,
                            "coherence_excess",
                            a["coherence_excess_mean"],
                            a["coherence_excess_std"],
                            a["n_params"],
                        ]
                    )
                    w.writerow(
                        [
                            p["label"],
                            lbl,
                            "mean_wrong_per_row",
                            a["mean_wrong_per_row_mean"],
                            a["mean_wrong_per_row_std"],
                            a["n_params"],
                        ]
                    )

    # Per-config Δ table — THE deliverable (CLAUDE.md §2: a result is a Δ, never a lone
    # arm number). One row per (config, delta-pair, metric) with paired mean ± std.
    deltas_csv_path = out_dir / f"{tag}_{stamp}_deltas.csv"
    with open(deltas_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "delta",
                "metric",
                "delta_mean",
                "delta_std",
                "n_seeds",
                "sign_pos",
                "sign_neg",
                "sign_p",
            ]
        )
        for p in points:
            for pair, reps in p["deltas"].items():
                for metric, rep in reps.items():
                    st = rep.get("sign_test") or {}
                    w.writerow(
                        [
                            p["label"],
                            pair,
                            metric,
                            rep["delta_mean"],
                            rep["delta_std"],
                            rep["n_seeds"],
                            st.get("n_pos", ""),
                            st.get("n_neg", ""),
                            st.get("p_value", ""),
                        ]
                    )

    print(f"\nResults : {json_path}")
    print(f"Curve   : {csv_path}")
    print(f"Deltas  : {deltas_csv_path}")

    # M21: side-car introspection table — one row per (config, arm, diagnostic) with mean ± std
    # across seeds. Written only when diagnostics ran (the off path skips it entirely).
    if any(p.get("diagnostics") for p in points):
        diag_csv_path = out_dir / f"{tag}_{stamp}_diagnostics.csv"
        with open(diag_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "arm", "metric", "mean", "std", "n_seeds"])
            for p in points:
                for lbl in labels:
                    for metric, st in p.get("diagnostics", {}).get(lbl, {}).items():
                        w.writerow(
                            [p["label"], lbl, metric, st["mean"], st["std"], len(st["per_seed"])]
                        )
        print(f"Diag    : {diag_csv_path}")

    # Budget-parity audit (M3a confound guard): write the realized-param table and flag any
    # matched arm that drifted out of tolerance. `audit` was computed above for the run record.
    if audit is not None:
        params_csv_path = out_dir / f"{tag}_{stamp}_params.csv"
        with open(params_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "arm", "n_params", "ref_params", "ratio", "role", "within_tol"])
            for r in audit["rows"]:
                w.writerow(
                    [
                        r["config"],
                        r["arm"],
                        r["n_params"],
                        r["ref_params"],
                        f"{r['ratio']:.4f}",
                        r["role"],
                        r["within_tol"],
                    ]
                )
        record["budget_audit"] = audit
        print(f"Params  : {params_csv_path}")
        tol_pct = audit["tol"] * 100
        if audit["breaches"]:
            print(
                f"  ⚠ BUDGET PARITY: {len(audit['breaches'])} matched-arm cell(s) "
                f"exceeded ±{tol_pct:.0f}% vs '{audit['reference']}':"
            )
            for cell, lbl, ratio in audit["breaches"]:
                print(f"      [{cell}] {lbl}: ratio {ratio:.4f}")
            print(
                "    NOTE: a breach on the loop/ff_matched is a depth-attribution blocker; "
                "a breach on untied_matched is the expected high-T width-quantization "
                "finding (narrow blocks) — see LOG.md."
            )
        else:
            print(f"  ✓ BUDGET PARITY: all matched arms within ±{tol_pct:.0f}% per cell.")
    # The line-plot only makes sense for a 1-D numeric sweep; a grid is summarised by
    # the CSVs above.
    if cfg.sweep is not None:
        _maybe_plot(points, labels, cfg.sweep.param, out_dir / f"{tag}_{stamp}_curve.png")

    # M3a: when depth is coupled to a swept param, draw accuracy-vs-depth and Δ-vs-depth
    # curves faceted by the remaining grid axes (rule, w) — the depth-budget deliverable.
    if cfg.couple_n_steps_to_param is not None:
        _maybe_plot_depth_budget(
            points,
            labels,
            cfg.couple_n_steps_to_param,
            cfg.resolved_deltas(),
            out_dir / f"{tag}_{stamp}_depth_curve.png",
            out_dir / f"{tag}_{stamp}_depth_deltas.png",
        )

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
                if "exact_match" in extrap_results[T].get("baselines", {}):
                    b = extrap_results[T]["baselines"]["exact_match"]
                    w.writerow([T, 0, "baseline", "exact_match", b["mean"], b["std"]])
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

        # Paired Δ + sign test per (T_test, R') cell — so the extrapolation diagonal carries
        # the same significance call as every other Δ in the run (review fix M2).
        extrap_deltas_path = out_dir / f"{tag}_{stamp}_extrapolation_deltas.csv"
        with open(extrap_deltas_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["T_test", "R_test", "delta", "delta_mean", "delta_std",
                 "sign_pos", "sign_neg", "sign_p"]
            )
            for T in cfg.extrapolation.T_values:
                for R in cfg.extrapolation.R_values:
                    for pair, rep in extrap_results[T][R].get("_deltas", {}).items():
                        st = rep.get("sign_test") or {}
                        w.writerow(
                            [
                                T,
                                R,
                                pair,
                                rep["delta_mean"],
                                rep["delta_std"],
                                st.get("n_pos", ""),
                                st.get("n_neg", ""),
                                st.get("p_value", ""),
                            ]
                        )
        print(f"ExtrapΔ : {extrap_deltas_path}")

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
    xs = [p["overrides"][sweep_param] for p in points]
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


def _group_by_other_axes(points, coupled_param):
    """Group axis points by every grid key except the coupled one (e.g. by (rule, w)).

    Returns an ordered dict {group_label: [(coupled_value, point), ...] sorted by value}.
    Used by the M3a depth-budget plot: one facet per (rule, w), the coupled param (T) on x.
    """
    groups: dict = {}
    for p in points:
        ov = p["overrides"]
        if coupled_param not in ov:
            continue
        other = {k: v for k, v in ov.items() if k != coupled_param}
        glabel = ", ".join(f"{k}={other[k]}" for k in sorted(other)) or "all"
        groups.setdefault(glabel, []).append((ov[coupled_param], p))
    for g in groups.values():
        g.sort(key=lambda t: t[0])
    return groups


def _maybe_plot_depth_budget(points, labels, coupled_param, delta_pairs, out_curve, out_deltas):
    """M3a deliverable: accuracy-vs-T and Δ-vs-T curves, one facet per (rule, w) group."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("Depth Plot: skipped (matplotlib not installed; CSVs hold the curves)")
        return

    groups = _group_by_other_axes(points, coupled_param)
    if not groups:
        return
    glabels = list(groups.keys())

    # --- Figure 1: per-arm test accuracy vs the coupled param (T), faceted by group. ---
    fig, axes = plt.subplots(1, len(glabels), figsize=(5 * len(glabels), 4.2), squeeze=False)
    for j, gl in enumerate(glabels):
        ax = axes[0][j]
        xs = [v for v, _ in groups[gl]]
        for lbl in labels:
            means = np.array([p["agg"][lbl]["accuracy_mean"] for _, p in groups[gl]])
            stds = np.array([p["agg"][lbl]["accuracy_std"] for _, p in groups[gl]])
            ax.plot(xs, means, marker="o", label=lbl)
            ax.fill_between(xs, means - stds, means + stds, alpha=0.2)
        base = np.array([p["agg"]["baseline"]["mean"] for _, p in groups[gl]])
        ax.plot(xs, base, color="gray", linestyle="--", label="majority baseline")
        ax.set_title(gl)
        ax.set_xlabel(coupled_param)
        ax.set_ylabel("test accuracy")
        ax.set_xticks(xs)
        ax.legend(fontsize=8)
    fig.suptitle("M3a: test accuracy vs depth at fixed budget")
    fig.tight_layout()
    fig.savefig(out_curve, dpi=120)
    plt.close(fig)
    print(f"Depth Plot: {out_curve}")

    # --- Figure 2: headline Δ(recurrent − control) vs the coupled param, faceted by group. ---
    fig, axes = plt.subplots(1, len(glabels), figsize=(5 * len(glabels), 4.2), squeeze=False)
    for j, gl in enumerate(glabels):
        ax = axes[0][j]
        xs = [v for v, _ in groups[gl]]
        for a, b in delta_pairs:
            key = f"{a}-{b}"
            means, stds = [], []
            for _, p in groups[gl]:
                rep = p["deltas"].get(key, {}).get("accuracy")
                means.append(rep["delta_mean"] if rep else np.nan)
                stds.append(rep["delta_std"] if rep else 0.0)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(xs, means, marker="o", label=f"Δ({a}−{b})")
            ax.fill_between(xs, means - stds, means + stds, alpha=0.2)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(gl)
        ax.set_xlabel(coupled_param)
        ax.set_ylabel("Δ test accuracy (loop − control)")
        ax.set_xticks(xs)
        ax.legend(fontsize=8)
    fig.suptitle("M3a: Δ(loop − control) vs depth at fixed budget")
    fig.tight_layout()
    fig.savefig(out_deltas, dpi=120)
    plt.close(fig)
    print(f"Δ Plot   : {out_deltas}")


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
