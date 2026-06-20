"""Evaluation metrics and the Δ(recurrent − control) comparison."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        X = X.to(device)
        logits, _ = model(X)
        if logits.ndim == 2:
            pred = logits.argmax(dim=-1).cpu().numpy()
        else:
            # multi-output (B, W, C)
            pred = logits.argmax(dim=-1).cpu().numpy()
        preds.append(pred)
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def accuracy(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Token-level (per-bit) accuracy."""
    preds, targets = _predict(model, loader, device)
    return float((preds == targets).mean())


def exact_match(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Exact-match (whole-row correct). Only meaningful for multi-output targets."""
    preds, targets = _predict(model, loader, device)
    if targets.ndim == 1:
        return float((preds == targets).mean())  # same as accuracy for single-output
    return float((preds == targets).all(axis=-1).mean())


def delta_report(
    recurrent_scores: list[float],
    control_scores: list[float],
    label: str = "accuracy",
) -> dict:
    """
    Compute Δ = recurrent − control over multiple seeds.
    Returns mean, sample std (ddof=1), and per-seed values.
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
        "label": label,
        "n_seeds": len(r),
    }
