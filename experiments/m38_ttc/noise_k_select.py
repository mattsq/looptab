"""M38b: test-time compute on 9x9 Sudoku — per-step noise, K samples, and selection.

Question (lit-review item 1, Probabilistic TRM arXiv 2605.19943): does sampling K noisy runs of
the loop and SELECTING one buy whole-grid exact match that (a) more deterministic loops at the
same compute, (b) the same sampling on a feedforward control, and (c) the untied mixer do not?

Arms are trained by the runner's own `run_point` (same data seeds, budgets, recipe and AMP as
M38a), from `configs/experiments/m38a_sudoku9_baseline.yaml` plus one extra arm: `trm_mixer`
trained with per-step latent noise. `trm_flat` is dropped (it cannot fit the training set, M38a).
Noise is injected with forward hooks, so no model code changes: after every RMSNorm in the mixers
(once per loop step) and after every GELU in `ff_matched`, out += sigma * rms(out) * N(0, 1), so
sigma is relative to the activation scale in every architecture. sigma = 0 adds nothing and draws
no random numbers, so the noise-free arms train exactly as in M38a.

Test-time strategies, per arm, loop count R and noise level sigma, from K nested samples:
  - `violation`: the sample with the fewest Sudoku rule violations (duplicates in rows, columns
    and boxes, plus changed givens), ties broken by confidence. A violation-free grid that keeps
    the givens IS the unique solution, so this is a verifier, not a learned selector.
  - `confidence`: the sample with the highest mean log max-probability (verifier-free).
  - `vote`: per-cell majority over the K samples (no selection).
  - `oracle`: any of the K samples fully correct (upper bound).
Compute-matched control: the deterministic loop at R' = K * R (tied mixers only).

Writes a run record (JSON, gitignored) and a tracked CSV summary to results/.
Usage: python experiments/m38_ttc/noise_k_select.py --n-givens 30 --seeds 1 2 ... 8
Seed 0 was the pilot used to choose the noise levels; report seeds 1-8 (fresh puzzles).
"""

import argparse
import copy
import csv
import hashlib
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

import looptab.run as runner
from looptab.config import ExperimentConfig
from looptab.models.trm import RMSNorm

BASE_CONFIG = "configs/experiments/m38a_sudoku9_baseline.yaml"
K_VALUES = [1, 2, 4, 8, 16, 32]
DET_R = [8, 16, 32, 64, 128, 256, 512]

# Noise state read by the hooks: training noise per arm label, and the eval noise level.
_TRAIN_SIGMA: dict[int, float] = {}  # id(model) -> sigma during training
_EVAL = {"sigma": 0.0, "gen": None}


def _noise_hook(model_id):
    def hook(module, inputs, out):
        sigma = _TRAIN_SIGMA.get(model_id, 0.0) if module.training else _EVAL["sigma"]
        if sigma == 0.0:
            return None
        rms = out.float().pow(2).mean(-1, keepdim=True).sqrt()
        gen = None if module.training else _EVAL["gen"]
        eps = torch.randn(out.shape, device=out.device, generator=gen)
        return (out.float() + sigma * rms * eps).to(out.dtype)
    return hook


def _attach_hooks(model: nn.Module):
    kinds = (RMSNorm,) if any(isinstance(m, RMSNorm) for m in model.modules()) else (nn.GELU,)
    for mod in model.modules():
        if isinstance(mod, kinds):
            mod.register_forward_hook(_noise_hook(id(model)))


def _patch_runner(noisy_train: dict[str, float], cache_dir: Path):
    build = runner._build_model

    def build_with_hooks(arm, *args, **kwargs):
        m = build(arm, *args, **kwargs)
        _attach_hooks(m)
        _TRAIN_SIGMA[id(m)] = noisy_train.get(arm.resolved_label(), 0.0)
        return m

    split = runner.make_splits

    def cached_splits(**kw):
        key = hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:16]
        path = cache_dir / f"splits_{key}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        out = split(**kw)
        path.write_bytes(pickle.dumps(out))
        return out

    runner._build_model = build_with_hooks
    runner.make_splits = cached_splits


def _units(size=9, box=3):
    idx = np.arange(size * size).reshape(size, size)
    units = [idx[r] for r in range(size)] + [idx[:, c] for c in range(size)]
    for br in range(0, size, box):
        for bc in range(0, size, box):
            units.append(idx[br:br + box, bc:bc + box].ravel())
    return torch.as_tensor(np.stack(units))  # (27, 9)


def violations(pred, givens, units):
    """Duplicates in every row/column/box plus changed givens; 0 <=> the unique solution."""
    counts = torch.nn.functional.one_hot(pred[:, units], 9).sum(2)  # (N, 27, 9)
    dup = (counts - 1).clamp(min=0).sum((1, 2))
    changed = ((givens >= 0) & (pred != givens)).sum(1)
    return dup + changed


@torch.no_grad()
def sample(model, X, R, sigma, K, seed, device):
    """K runs of R loops at noise sigma -> preds (K, N, 81), mean log max-prob (K, N)."""
    _EVAL["sigma"] = sigma
    _EVAL["gen"] = torch.Generator(device=device).manual_seed(10_000 + seed)
    preds, confs = [], []
    for _ in range(K):
        logits, _ = model(X, n_steps=R)
        logp = torch.log_softmax(logits.float(), -1).view(X.shape[0], 81, 9)
        mx, arg = logp.max(-1)
        preds.append(arg)
        confs.append(mx.mean(-1))
    _EVAL["sigma"] = 0.0
    return torch.stack(preds), torch.stack(confs)


def strategies(preds, confs, y, givens, units):
    """Metrics for every K in K_VALUES (nested: the first K samples)."""
    viol = torch.stack([violations(p, givens, units) for p in preds])  # (K, N)
    rows = np.arange(y.shape[0])
    out = {}
    for K in K_VALUES:
        if K > preds.shape[0]:
            break
        p, c, v = preds[:K], confs[:K], viol[:K]
        correct = p == y.unsqueeze(0)  # (K, N, 81)
        picks = {
            "violation": torch.argmin(v.double() * 1e3 - c.double(), 0),
            "confidence": torch.argmax(c, 0),
        }
        res = {}
        for name, k in picks.items():
            chosen = correct[k.cpu().numpy(), rows]
            res[name] = (chosen.float().mean().item(), chosen.all(-1).float().mean().item())
        vote = torch.nn.functional.one_hot(p, 9).sum(0).argmax(-1)
        vc = vote == y
        res["vote"] = (vc.float().mean().item(), vc.all(-1).float().mean().item())
        res["oracle"] = (float("nan"), correct.all(-1).any(0).float().mean().item())
        out[K] = res
    return out


def git_sha():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-givens", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--train-sigma", type=float, nargs="+", default=[0.1])
    ap.add_argument("--eval-sigma", type=float, nargs="+", default=[0.2, 0.4, 0.8])
    ap.add_argument("--noise-arms", nargs="+",
                    default=["trm_mixer", "untied_mixer_matched", "ff_matched"])
    ap.add_argument("--sample-R", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cache-dir", default="results/.m38_cache")
    ap.add_argument("--tag", default="m38b_noise_k_select")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(open(BASE_CONFIG, encoding="utf-8"))
    raw["arms"] = [a for a in raw["arms"] if a["label"] != "trm_flat"]
    # Training noise is its own ingredient (it may just regularise), so every base arm named in
    # --noise-arms also gets a noise-trained copy: the controls see the same training noise.
    noisy = {}
    for s in args.train_sigma:
        for base in args.noise_arms:
            arm = copy.deepcopy(next(a for a in raw["arms"] if a["label"] == base))
            arm["label"] = f"{base}_noise{s:g}"
            raw["arms"].append(arm)
            noisy[arm["label"]] = s
    raw["budget_reference"] = "ff_matched"
    raw["deltas"] = []
    raw.pop("sweep", None)
    raw["task"]["params"]["n_givens"] = args.n_givens
    if args.epochs is not None:
        raw["train"]["epochs"] = args.epochs
    cfg = ExperimentConfig(**raw)
    _patch_runner(noisy, cache)
    torch.set_num_threads(cfg.train.num_threads)
    device = cfg.train.device
    units = _units().to(device)

    per_seed = []
    for seed in args.seeds:
        t0 = time.time()
        r, models, _, _ = runner.run_point(cfg, cfg.task.params, seed)
        _, test = runner.make_splits(
            task=cfg.task.name, task_cfg=cfg.task.params, task_seed=cfg.task.task_seed + seed,
            train_sample_seed=cfg.task.train_sample_seed + seed * 100,
            test_sample_seed=cfg.task.test_sample_seed + seed * 100,
            n_train=cfg.task.n_train, n_test=cfg.task.n_test, seed=seed,
        )
        X = torch.as_tensor(test.X).to(device)
        y = torch.as_tensor(test.y).to(device)
        oh = X.view(-1, 81, 10)
        givens = torch.where(oh[..., 0] > 0.5, -1, oh[..., 1:].argmax(-1))
        rec = {"seed": seed, "main": {k: {m: v[m] for m in ("accuracy", "exact_match",
                                                          "train_accuracy")}
                                      for k, v in r.items() if isinstance(v, dict)
                                      and "accuracy" in v},
               "det": {}, "sampled": {}}
        for lbl, m in models.items():
            m.eval()
            tied = not lbl.startswith(("untied", "ff"))
            if tied:
                for R in DET_R:
                    p, c = sample(m, X, R, 0.0, 1, seed, device)
                    ok = p[0] == y
                    v = violations(p[0], givens, units)
                    rec["det"][f"{lbl}|{R}"] = (ok.float().mean().item(),
                                                ok.all(-1).float().mean().item(),
                                                v.float().mean().item())
            for R in (args.sample_R if tied else [8]):
                for sigma in args.eval_sigma:
                    p, c = sample(m, X, R, sigma, max(K_VALUES), seed, device)
                    rec["sampled"][f"{lbl}|{R}|{sigma:g}"] = strategies(p, c, y, givens, units)
        per_seed.append(rec)
        best = max(rec["sampled"].items(), key=lambda kv: kv[1][32]["violation"][1])
        print(f"seed {seed} ({time.time() - t0:.0f}s) main EM "
              + " ".join(f"{k}={v['exact_match']:.3f}" for k, v in rec["main"].items())
              + f" | best sampled EM@32 {best[0]} = {best[1][32]['violation'][1]:.3f}",
              flush=True)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    tag = f"{args.tag}_g{args.n_givens}_{stamp}"
    record = {"config": cfg.model_dump(), "script_args": vars(args), "git_sha": git_sha(),
              "timestamp": stamp, "seeds": args.seeds, "per_seed": per_seed}
    Path("results").mkdir(exist_ok=True)
    with open(f"results/{tag}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=str)

    # Tracked summary: one row per (arm, mode, R, sigma, strategy, K) with per-seed EM kept so
    # the write-up can run paired sign tests.
    rows = []
    for key in per_seed[0]["det"]:
        vals = np.array([s["det"][key] for s in per_seed])
        lbl, R = key.split("|")
        rows.append([args.n_givens, lbl, "deterministic", R, 0, "-", 1, *_ms(vals[:, 0]),
                     *_ms(vals[:, 1]), ";".join(f"{x:.4f}" for x in vals[:, 1])])
    for key in per_seed[0]["sampled"]:
        lbl, R, sigma = key.split("|")
        for K in K_VALUES:
            for strat in ("violation", "confidence", "vote", "oracle"):
                vals = np.array([s["sampled"][key][K][strat] for s in per_seed])
                rows.append([args.n_givens, lbl, "sampled", R, sigma, strat, K,
                             *_ms(vals[:, 0]), *_ms(vals[:, 1]),
                             ";".join(f"{x:.4f}" for x in vals[:, 1])])
    with open(f"results/{tag}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_givens", "arm", "mode", "R", "sigma", "strategy", "K", "acc_mean",
                    "acc_std", "em_mean", "em_std", "em_per_seed"])
        w.writerows(rows)
    print("record:", f"results/{tag}.json")


def _ms(v):
    return float(np.nanmean(v)), float(np.nanstd(v))


if __name__ == "__main__":
    main()
