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


@torch.inference_mode()
def act_predict(
    model: nn.Module,
    loader: DataLoader,
    max_segments: int,
    device: str = "cpu",
    n_steps: int | None = None,
    halt_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Adaptive-segment prediction for an ACT model (M23). Returns (preds, targets, avg_segments).

    Runs the loop segment-by-segment (carrying ``(z, a)`` across segments, like training). After
    each segment the ``halt_head`` gives a per-example halt probability; an example's prediction is
    frozen at the FIRST segment where that probability crosses ``halt_threshold`` (and it stops
    consuming compute). Examples that never halt use the final segment's answer. ``avg_segments`` is
    the mean segments actually used — the adaptive-compute diagnostic (should be lower on easy data,
    higher on hard), the behavioural signature of TRM/HRM halting.
    """
    model.eval()
    preds, targets, seg_counts = [], [], []
    for X, y in loader:
        X = X.to(device)
        B = X.shape[0]
        state = None
        chosen = None
        halted = torch.zeros(B, dtype=torch.bool, device=X.device)
        used = torch.full((B,), max_segments, dtype=torch.long, device=X.device)
        for seg in range(max_segments):
            logits, _, state = model(X, n_steps=n_steps, init_state=state, return_state=True)
            z, _ = state
            if chosen is None:
                chosen = logits.clone()
            halt_p = torch.sigmoid(model.halt_head(z).squeeze(-1))
            fires = (halt_p > halt_threshold) & (~halted)
            chosen[fires] = logits[fires]
            used[fires] = seg + 1
            halted = halted | fires
            if bool(halted.all()):
                break
        chosen[~halted] = logits[~halted]  # never-halted: last segment's answer
        preds.append(chosen.argmax(dim=-1).cpu().numpy())
        targets.append(y.numpy())
        seg_counts.append(used.cpu().numpy())
    return (
        np.concatenate(preds),
        np.concatenate(targets),
        float(np.concatenate(seg_counts).mean()),
    )


def metrics_from_preds(preds: np.ndarray, targets: np.ndarray, *, want_exact_match: bool) -> dict:
    """Accuracy (+ optional exact-match / coherence) from precomputed preds — shared by the
    single-pass ``evaluate`` and the ACT ``evaluate_act`` so both report identical statistics."""
    out = {"accuracy": float((preds == targets).mean())}
    if want_exact_match:
        if targets.ndim == 1:
            out["exact_match"] = out["accuracy"]
        else:
            correct = preds == targets
            out["exact_match"] = float(correct.all(axis=-1).mean())
            w_out = targets.shape[-1]
            out["coherence_excess"] = out["exact_match"] - float(out["accuracy"] ** w_out)
            out["mean_wrong_per_row"] = float((~correct).sum(axis=-1).mean())
    return out


def evaluate_act(
    model: nn.Module,
    loader: DataLoader,
    max_segments: int,
    device: str = "cpu",
    *,
    want_exact_match: bool = False,
    n_steps: int | None = None,
    halt_threshold: float = 0.5,
) -> dict:
    """ACT counterpart of ``evaluate``: metrics from the adaptively-halted predictions, plus the
    mean number of segments used (``avg_segments``) as the adaptive-compute diagnostic."""
    preds, targets, avg_segments = act_predict(
        model, loader, max_segments, device, n_steps=n_steps, halt_threshold=halt_threshold
    )
    out = metrics_from_preds(preds, targets, want_exact_match=want_exact_match)
    out["avg_segments"] = avg_segments
    return out


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


@torch.inference_mode()
def _predict_regression(
    model: nn.Module, loader: DataLoader, device: str, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Raw model outputs as regression predictions (M26). No argmax — the readout values ARE the
    forecast. Returns (preds, targets), both ``(N, M, H)`` float (variable-cells × horizon)."""
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        X = X.to(device)
        out, _ = model(X, **kwargs)
        preds.append(out.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_regression(
    model: nn.Module, loader: DataLoader, device: str = "cpu", **kwargs
) -> dict:
    """Forecasting metrics on standardized targets (M26): MSE, MAE, and R² (the MTS-benchmark
    convention). MSE/MAE are pooled over all (example, variable, horizon) entries; R² = 1 − SSE/SST
    against the per-entry target mean. ``accuracy`` mirrors ``-mse`` so the runner's generic
    "higher is better" plumbing (baseline gap, curve rows) stays meaningful for a regression run
    without special-casing every downstream key — the reported headline metrics are mse/mae/r2.
    (R² uses the TEST-set target mean for SST; on standardized targets that mean ≈ 0, so it is
    indistinguishable from the train-mean convention — MSE is the headline, R² a companion.)"""
    preds, targets = _predict_regression(model, loader, device, **kwargs)
    err = preds - targets
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    sst = float(np.mean((targets - targets.mean()) ** 2))
    r2 = float(1.0 - mse / sst) if sst > 0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": -mse}


def persistence_baseline_mse(loader: DataLoader, lookback: int, n_vars: int) -> dict:
    """Naive-forecast (persistence) baseline for M26: predict every horizon step = the LAST observed
    value of each variable. On standardized inputs/targets this is directly comparable to the arms'
    MSE/MAE. X is the flattened ``(N, M*lookback)`` window (cell i = variable i's history), so the
    last observed value per variable is column ``i*lookback + (lookback-1)``."""
    all_y, all_last = [], []
    for X, y in loader:
        Xr = X.numpy().reshape(len(X), n_vars, lookback)
        last = Xr[:, :, -1:]  # (N, M, 1) last observed value per variable
        yb = y.numpy()  # (N, M, H)
        all_y.append(yb)
        all_last.append(np.broadcast_to(last, yb.shape))
    if not all_y:
        return {"mse": 0.0, "mae": 0.0, "r2": 0.0}
    Y = np.concatenate(all_y)
    P = np.concatenate(all_last)
    err = P - Y
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    sst = float(np.mean((Y - Y.mean()) ** 2))
    r2 = float(1.0 - mse / sst) if sst > 0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


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


def sign_test(delta_per_seed: list[float], eps: float = 0.0) -> dict:
    """Paired sign test on per-seed Δs (CLAUDE.md §2/§5.2 — a Δ needs a significance call).

    Counts how many seeds favour the recurrent arm (Δ>eps) vs the control (Δ<-eps); ties
    (|Δ|<=eps) are dropped, as the sign test prescribes. Reports an exact two-sided binomial
    p-value under H0: P(Δ>0)=0.5. Distribution-free — appropriate for small-sample seeds where
    normality is dubious. NOTE: with < 6 non-tied seeds the test cannot reach p<0.05 (a perfect
    5/5 split gives p=0.0625); use >= 8 seeds when significance is the point (CLAUDE.md §5.2).

    ``eps`` (default 0.0 ⇒ ties are only exact Δ==0, the classic sign test — bit-identical to the
    pre-M29 behaviour) treats any |Δ|<=eps as a PRACTICAL tie and drops it. This is the ceiling-tie
    guard the M29c post-mortem motivated: on near-saturated arms a Δ of one flipped test cell
    (e.g. 1/24000 ≈ 4e-5) is not evidence, yet the eps=0 test counts it as a full "positive vote"
    and can manufacture an 8/0, p<.05 out of a handful of ceiling ties. Pass a small eps (see
    ``delta_report``'s ``near_tie_eps``) to get the robust count.
    """
    d = np.asarray(delta_per_seed, dtype=float)
    n_pos = int((d > eps).sum())
    n_neg = int((d < -eps).sum())
    n_zero = int((np.abs(d) <= eps).sum())
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
    near_tie_eps: float = 1e-3,
) -> dict:
    """
    Compute Δ = recurrent − control over multiple seeds.
    Returns mean, sample std (ddof=1), per-seed values, and optionally a paired sign test.

    Besides the classic ``sign_test`` (exact ties only — the headline number, unchanged), the
    record also carries a CEILING-TIE-ROBUST companion (the M29c post-mortem lesson): ``n_near_tie``
    counts seeds whose |Δ| <= ``near_tie_eps`` (a difference below ~one flipped cell / 0.1pp is
    not evidence on near-saturated arms), and ``sign_test_robust`` re-runs the sign test treating
    those as ties. When ``n_near_tie`` is a large fraction of seeds, an impressive raw sign count
    (e.g. 8/0, p<.05) can collapse under the robust test (e.g. 3/3, ns) — read both before claiming
    significance on arms near the accuracy ceiling. ``near_tie_eps`` is on the metric's own scale
    (accuracy/EM in [0,1]); default 1e-3.
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
        "n_near_tie": int((np.abs(delta) <= near_tie_eps).sum()),
        "near_tie_eps": near_tie_eps,
    }
    if paired_sign_test:
        out["sign_test"] = sign_test(delta.tolist())
        out["sign_test_robust"] = sign_test(delta.tolist(), eps=near_tie_eps)
    else:
        out["sign_test"] = None
        out["sign_test_robust"] = None
        out["sign_test_note"] = "not_run_non_independent_splits"
    return out
