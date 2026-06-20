"""Evaluation metrics and the Δ(recurrent − control) comparison."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _predict(
    model: nn.Module, loader: DataLoader, device: str, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        X = X.to(device)
        logits, _ = model(X, **kwargs)
        if logits.ndim == 2:
            pred = logits.argmax(dim=-1).cpu().numpy()
        else:
            # multi-output (B, W, C)
            pred = logits.argmax(dim=-1).cpu().numpy()
        preds.append(pred)
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def accuracy(model: nn.Module, loader: DataLoader, device: str = "cpu", **kwargs) -> float:
    """Token-level (per-bit) accuracy."""
    preds, targets = _predict(model, loader, device, **kwargs)
    return float((preds == targets).mean())


def exact_match(model: nn.Module, loader: DataLoader, device: str = "cpu", **kwargs) -> float:
    """Exact-match (whole-row correct). Only meaningful for multi-output targets."""
    preds, targets = _predict(model, loader, device, **kwargs)
    if targets.ndim == 1:
        return float((preds == targets).mean())  # same as accuracy for single-output
    return float((preds == targets).all(axis=-1).mean())


def majority_baseline(loader: DataLoader) -> float:
    """Compute token-level majority class baseline accuracy."""
    targets = []
    for _, y in loader:
        targets.append(y.numpy())
    if not targets:
        return 0.0
    targets = np.concatenate(targets)
    _, counts = np.unique(targets, return_counts=True)
    if len(counts) == 0:
        return 0.0
    return float(np.max(counts) / targets.size)


def _binom_two_sided_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for k successes in n trials at p=0.5.

    Dependency-free (no scipy): sum the binomial pmf over all outcomes at least as
    extreme as k. Used for the paired sign test below.
    """
    if n == 0:
        return 1.0
    from math import comb

    pmf = [comb(n, i) / (2.0**n) for i in range(n + 1)]
    obs = pmf[k]
    # Tolerance guards float wobble when symmetric outcomes should count as "as extreme".
    return float(min(1.0, sum(p for p in pmf if p <= obs + 1e-12)))


def sign_test(delta_per_seed: list[float]) -> dict:
    """Paired sign test on per-seed Δs (CLAUDE.md §2/§5.2 — a Δ needs a significance call).

    Counts how many seeds favour the recurrent arm (Δ>0) vs the control (Δ<0); ties
    (Δ==0) are dropped, as the sign test prescribes. Reports an exact two-sided binomial
    p-value under H0: P(Δ>0)=0.5. Distribution-free — appropriate for ≥5 small-sample
    seeds where normality is dubious.
    """
    d = np.asarray(delta_per_seed, dtype=float)
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    n_zero = int((d == 0).sum())
    n_eff = n_pos + n_neg
    k = max(n_pos, n_neg)
    p = _binom_two_sided_p(k, n_eff)
    return {"n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero, "n_eff": n_eff, "p_value": p}


def delta_report(
    recurrent_scores: list[float],
    control_scores: list[float],
    label: str = "accuracy",
) -> dict:
    """
    Compute Δ = recurrent − control over multiple seeds.
    Returns mean, sample std (ddof=1), per-seed values, and a paired sign test.
    """
    r = np.array(recurrent_scores)
    c = np.array(control_scores)
    delta = r - c

    def _std(x):
        return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    return {
        "recurrent_mean": float(r.mean()),
        "recurrent_std": _std(r),
        "control_mean": float(c.mean()),
        "control_std": _std(c),
        "delta_mean": float(delta.mean()),
        "delta_std": _std(delta),
        "recurrent_per_seed": r.tolist(),
        "control_per_seed": c.tolist(),
        "delta_per_seed": delta.tolist(),
        "sign_test": sign_test(delta.tolist()),
        "label": label,
        "n_seeds": len(r),
    }
