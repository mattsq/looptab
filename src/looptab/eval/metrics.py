"""Evaluation metrics and the Δ(recurrent − control) comparison."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def _predict(
    model: nn.Module, loader: DataLoader, device: str, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    # inference_mode is a strictly-faster no_grad (skips view/version tracking) and is safe
    # here: predictions only feed argmax/numpy, never autograd. Numerically identical.
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        X = X.to(device)
        logits, _ = model(X, **kwargs)
        # argmax over the class dim handles both single-output (B, C) and multi-output (B, W, C).
        preds.append(logits.argmax(dim=-1).cpu().numpy())
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


def multilabel_f1(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Micro- and macro-averaged F1 over labels for a multi-label binary task (positive = 1).

    The standard metrics for multi-label classification — added in M20 because subset accuracy
    (EM) on imbalanced multi-label data rewards getting the *frequent* label-combinations exactly
    right, so an arm can win EM while being *worse* per-label. F1 is the honest co-headline:
      - **micro** pools all per-label decisions (frequency-weighted; dominated by common labels).
      - **macro** averages the per-label F1 (every label counts equally; surfaces rare labels).
    Zero-division is treated as F1=0 for that label (the sklearn ``zero_division=0`` convention) — a
    label with no predicted-and-no-true positives in the slice contributes 0, not NaN.
    ``preds``/``targets`` are ``(N, L)`` arrays of {0,1}; for single-output (1-D) F1 is undefined,
    so this returns zeros (callers gate on multi-output).
    """
    if targets.ndim == 1:
        return {"micro_f1": 0.0, "macro_f1": 0.0}
    p = preds.astype(bool)
    t = targets.astype(bool)
    tp = (p & t).sum(axis=0).astype(np.float64)  # per-label
    fp = (p & ~t).sum(axis=0).astype(np.float64)
    fn = (~p & t).sum(axis=0).astype(np.float64)

    def _f1(tp_, fp_, fn_):
        # Safe divide: F1=0 where the denominator is 0 (zero_division=0), without evaluating the
        # division there (np.where would, emitting a divide warning).
        num = 2 * tp_
        denom = 2 * tp_ + fp_ + fn_
        denom = np.asarray(denom, dtype=np.float64)
        out = np.zeros_like(denom)
        np.divide(num, denom, out=out, where=denom > 0)
        return out

    macro = float(np.mean(_f1(tp, fp, fn)))
    micro = float(_f1(np.atleast_1d(tp.sum()), np.atleast_1d(fp.sum()), np.atleast_1d(fn.sum()))[0])
    return {"micro_f1": micro, "macro_f1": macro}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    *,
    want_exact_match: bool = False,
    want_f1: bool = False,
    **kwargs,
) -> dict:
    """Accuracy (and optional exact-match) from a *single* forward pass over ``loader``.

    ``accuracy`` and ``exact_match`` each run their own ``_predict``, so asking for both —
    which every multi-output (Task B) eval does, on test *and* under the extrapolation
    harness for each R' — used to forward the model over the data twice. This computes the
    predictions once and derives both metrics, halving eval forward passes there. The values
    are identical to calling the two functions separately (same argmax, same reductions).
    """
    preds, targets = _predict(model, loader, device, **kwargs)
    out = {"accuracy": float((preds == targets).mean())}
    if want_exact_match:
        if targets.ndim == 1:
            out["exact_match"] = out["accuracy"]  # whole-row == per-row for single-output
        else:
            correct = preds == targets  # (N, W) per-cell correctness
            out["exact_match"] = float(correct.all(axis=-1).mean())
            # --- Coherence descriptor (M9) -------------------------------------------------
            # `coherence_excess` = EM − token_acc**W, where token_acc**W is the whole-row score
            # expected if per-cell errors were i.i.d. at the arm's GLOBAL token-accuracy. It is a
            # PER-ARM descriptor of how much more coherent an arm's rows are than that i.i.d.
            # reference. IMPORTANT (M9 adversarial review): do NOT use the *cross-arm* Δ of this
            # metric as evidence. Two reasons:
            #   (1) Jensen / per-row-dispersion confound. EM = mean_row(row_acc**W) ≥
            #       (mean_row row_acc)**W, so heterogeneous per-row difficulty inflates
            #       coherence_excess even with NO clustering. Matching two arms' *mean* token-acc
            #       does not match the *variance* of their per-row accuracy, so it does not control
            #       this — a per-row baseline mean_row(row_acc**W) would, but it also cancels the
            #       cross-row clustering that is the actual signal, so it is not used.
            #   (2) At matched token-acc the token_acc**W term is identical across arms, so
            #       Δ(coherence_excess) ≡ Δ(exact_match) — it adds nothing beyond EM.
            # The clean, unconfounded cross-arm mechanism statistic is therefore EM AT MATCHED
            # token-acc (e.g. loop vs ff @ w=24), NOT a coherence_excess Δ. `mean_wrong_per_row`
            # (mean wrong cells per row) is a companion descriptor for the narrative.
            w_out = targets.shape[-1]
            token_acc = out["accuracy"]
            out["coherence_excess"] = out["exact_match"] - float(token_acc**w_out)
            out["mean_wrong_per_row"] = float((~correct).sum(axis=-1).mean())
    # M20: micro/macro-F1 — the standard multi-label metrics, the honest co-headline to EM (which
    # over-rewards modal label-combinations). Gated by `want_f1` (set only for the multilabel task)
    # so every synthetic-task eval is byte-identical to before.
    if want_f1 and targets.ndim > 1:
        out.update(multilabel_f1(preds, targets))
    return out


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


def subset_accuracy_baseline(loader: DataLoader) -> float:
    """Frequency of the single most common whole-row labelset (the EM a best *constant*-row
    predictor scores). The exact-match analogue of ``majority_baseline``, and the honest degeneracy
    tripwire for multi-label data (M20): per-token ``majority_baseline`` is inflated by label
    sparsity — predicting all-zeros scores high token-accuracy — whereas subset accuracy (= EM) is
    what the §9.2 coherence finding is about, so its constant-predictor floor is the right
    reference.

    For single-output (1-D) targets this reduces exactly to ``majority_baseline`` (the most-common
    class).
    """
    targets = []
    for _, y in loader:
        targets.append(y.numpy())
    if not targets:
        return 0.0
    targets = np.concatenate(targets)
    if targets.ndim == 1:
        _, counts = np.unique(targets, return_counts=True)
        return float(np.max(counts) / len(targets)) if len(counts) else 0.0
    # Most common whole row: hash each row to a tuple and count.
    rows, counts = np.unique(targets, axis=0, return_counts=True)
    return float(np.max(counts) / targets.shape[0]) if len(counts) else 0.0


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
    p-value under H0: P(Δ>0)=0.5. Distribution-free — appropriate for small-sample seeds where
    normality is dubious. NOTE: with < 6 non-tied seeds the test cannot reach p<0.05 (a perfect
    5/5 split gives p=0.0625); use >= 8 seeds when significance is the point (CLAUDE.md §5.2).
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
    *,
    paired_sign_test: bool = True,
) -> dict:
    """
    Compute Δ = recurrent − control over multiple seeds.
    Returns mean, sample std (ddof=1), per-seed values, and optionally a paired sign test.
    """
    r = np.array(recurrent_scores)
    c = np.array(control_scores)
    delta = r - c

    def _std(x):
        return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    out = {
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
    if paired_sign_test:
        out["sign_test"] = sign_test(delta.tolist())
    else:
        out["sign_test"] = None
        out["sign_test_note"] = "not_run_non_independent_splits"
    return out
