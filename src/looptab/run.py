"""Single entry point: python -m looptab.run --config <yaml> --seed <int>"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import yaml

from .config import ExperimentConfig
from .data.dataset import make_splits, make_loaders
from .eval.metrics import accuracy, exact_match, delta_report
from .registry import get_model
from .train.loop import train


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_single(cfg: ExperimentConfig, seed: int) -> dict:
    """Run one seed: train recurrent + control, return per-seed metrics."""
    torch.manual_seed(seed)

    task_cfg = cfg.task
    train_ds, test_ds = make_splits(
        task=task_cfg.name,
        task_cfg=task_cfg.params,
        task_seed=task_cfg.task_seed,
        train_sample_seed=task_cfg.train_sample_seed + seed * 100,
        test_sample_seed=task_cfg.test_sample_seed + seed * 100,
        n_train=task_cfg.n_train,
        n_test=task_cfg.n_test,
    )
    train_loader, test_loader = make_loaders(train_ds, test_ds, cfg.train.batch_size)

    X_sample, y_sample = train_ds[0]
    in_features = X_sample.shape[0]
    # num_classes: 2 for binary; for multi-output, still 2 (bit prediction)
    num_classes = 2

    device = cfg.train.device

    results = {}
    for role, mcfg in [("recurrent", cfg.model), ("control", cfg.control)]:
        m = get_model(
            mcfg.name,
            in_features=in_features,
            num_classes=num_classes,
            hidden_dim=mcfg.hidden_dim,
            latent_dim=mcfg.latent_dim,
            n_steps=mcfg.n_steps,
            **({"deep_supervision": mcfg.deep_supervision} if mcfg.name == "trm" else {}),
        )
        train(
            m,
            train_loader,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            deep_supervision_weight=cfg.train.deep_supervision_weight,
            device=device,
        )
        acc = accuracy(m, test_loader, device)
        em = exact_match(m, test_loader, device)
        results[role] = {"accuracy": acc, "exact_match": em, "n_params": m.count_params()}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw)

    seeds = [args.seed] if args.seed is not None else cfg.seeds

    per_seed = []
    for seed in seeds:
        print(f"[seed={seed}]")
        r = run_single(cfg, seed)
        r["seed"] = seed
        per_seed.append(r)
        print(f"  recurrent acc={r['recurrent']['accuracy']:.4f}  control acc={r['control']['accuracy']:.4f}")

    rec_accs = [r["recurrent"]["accuracy"] for r in per_seed]
    ctl_accs = [r["control"]["accuracy"] for r in per_seed]
    report = delta_report(rec_accs, ctl_accs, label="accuracy")
    print(f"\nΔ(recurrent − control) = {report['delta_mean']:.4f} ± {report['delta_std']:.4f}")

    record = {
        "config": cfg.model_dump(),
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "seeds": per_seed,
        "summary": report,
    }
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.config).stem
    out_path = out_dir / f"{tag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
