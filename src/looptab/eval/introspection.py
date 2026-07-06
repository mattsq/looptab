"""Latent / weight introspection for the looped models (M21).

Measurement-only diagnostics that open a window into *why* the loop behaves as M0–M20
found — never *what* (that is the accuracy Δ machinery in ``metrics.py``). The repo's
signature pathology is dynamical ("the loop does not settle a stable step operator;
over-unrolling R'>R decays toward baseline", M1/M3b/M7/M8) yet has only ever been read off
accuracy curves. This module instruments the dynamics directly, with three families of
descriptor drawn from the looped-reasoning / deep-equilibrium literature:

  A. Fixed-point & spectral dynamics (recurrent arms only)
     - latent residual trajectory ‖z_{t+1}−z_t‖/‖z_t‖ out to an over-unroll horizon
       (does the latent converge, drift, or cycle — the direct M1/M8 instrument);
     - per-step readout accuracy/EM out past the trained depth (does over-unrolling decay);
     - Jacobian spectral radius ρ(∂z_{t+1}/∂z_t) of the one-step latent map at the
       over-unrolled state (DEQ — Bai 2021 arXiv 2106.14342; STARS — arXiv 2605.26733):
       ρ<1 ⇒ contractive / extrapolation-friendly fixed point, ρ≥1 ⇒ the over-unroll decay
       we see. Estimated by power iteration on autograd Jacobian-vector products. The
       operator norm ‖J‖₂ (a contraction *sufficient* condition) is reported alongside;
     - path independence / asymptotic alignment (Anil 2022 arXiv 2211.09961): unroll from
       many random z0 inits and measure whether they reach the same attractor — correlates
       with upward generalization.

  B. Representation geometry / collapse (all arms)
     - effective rank (Roy & Vetterli: exp of Shannon entropy of the singular-value
       distribution) and participation ratio of the final latent representation. Detects
       whether ``z`` is actually using its dimensions or collapsing.

  C. Weight & training dynamics (all arms)
     - per-Linear top singular value and the product (Lipschitz upper bound on the linear
       part) — the controllable knob behind contraction (Rethinking Deep Thinking
       arXiv 2410.23451).

Design (CLAUDE.md §2/§5/§8): every diagnostic is computed *after* training, reading the
trained model with forward / autograd passes only — it cannot perturb any committed result,
and it touches **no** model code (it rides the existing ``init_state``/``return_state``
resumable-rollout API from M7 and forward hooks, so the off path is byte-identical by
construction). Diagnostics are per-arm *descriptors*; the caller reports them across all arms
and both anchor regimes — the contrast is the finding, never a lone loop number. Stochastic
probes take an explicit ``seed`` so the numbers are reproducible (§5).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..models.controls import FFMatched, UntiedStack, UntiedStackMatched
from ..models.decoupled import TRMDecoupled
from ..models.mixer import TRMMixer
from ..models.trm import TRM

# GELU is ~1.085-Lipschitz; we treat the nonlinearities as 1-Lipschitz and report the
# product of the *linear* layers' spectral norms. Named so the approximation is explicit.
# TRMMixer (M27 mixer re-test) shares the (z, a) resumable API and a per-cell readout, so the
# Family-A dynamics probes apply unchanged — its z is (B, n_cells, latent) instead of (B, latent),
# which the ``reshape(B, -1)`` / last-dim ops throughout this module already handle.
_RECURRENT = (TRM, TRMDecoupled, TRMMixer)


def is_recurrent(model: nn.Module) -> bool:
    """True for arms with a resumable latent state (the Family-A subjects)."""
    return isinstance(model, _RECURRENT)


def _final_repr_module(model: nn.Module) -> nn.Module:
    """The final Linear whose *input* is the model's final latent representation.

    Hooking this layer's input gives a uniform 'final latent' across architectures: the
    refined ``z`` for the recurrent arms, the last block's latent for the untied stack, the
    penultimate activation for the feedforward control.
    """
    if isinstance(model, (TRM, TRMDecoupled, TRMMixer)):
        return model.readout
    if isinstance(model, FFMatched):
        return model.net[-1]
    if isinstance(model, UntiedStackMatched):
        return model.inner.readouts[-1]
    if isinstance(model, UntiedStack):
        return model.readouts[-1]
    raise TypeError(f"no representation module known for {type(model).__name__}")


# --------------------------------------------------------------------------------------
# Family B — representation geometry
# --------------------------------------------------------------------------------------
def _spectrum_stats(feats: np.ndarray) -> dict:
    """Effective rank + participation ratio of a representation matrix ``feats`` (N, D).

    Computed on the *centered* features (so it measures spread, not the mean offset):
      - effective_rank = exp(−Σ p_i log p_i), p_i = s_i / Σ s_j  (Roy & Vetterli, on the
        singular values s of the centered matrix);
      - participation_ratio = (Σ λ_i)² / Σ λ_i², λ_i = s_i²  (the covariance spectrum).
    Both range in [1, min(N, D)] and fall toward 1 under representational collapse.
    """
    F = feats - feats.mean(axis=0, keepdims=True)
    # Guard a degenerate (single-row / all-zero) slice.
    if F.shape[0] < 2 or not np.any(F):
        return {"effective_rank": 1.0, "participation_ratio": 1.0, "n_dims": int(feats.shape[1])}
    s = np.linalg.svd(F, compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    lam = s**2
    part_ratio = float((lam.sum() ** 2) / (lam**2).sum())
    return {
        "effective_rank": eff_rank,
        "participation_ratio": part_ratio,
        "n_dims": int(feats.shape[1]),
    }


@torch.no_grad()
def representation_geometry(model: nn.Module, X: torch.Tensor) -> dict:
    """Effective rank / participation ratio of the final latent representation (Family B)."""
    model.eval()
    captured: list[torch.Tensor] = []

    def hook(_module, inp, _out):
        captured.append(inp[0].detach())

    handle = _final_repr_module(model).register_forward_hook(hook)
    try:
        model(X)
    finally:
        handle.remove()
    # The readout fires once per step for recurrent arms; the final step's input is the
    # refined representation we want.
    feats = captured[-1]
    feats = feats.reshape(feats.shape[0], -1).cpu().numpy()
    return _spectrum_stats(feats)


# --------------------------------------------------------------------------------------
# Family C — weight spectral norms
# --------------------------------------------------------------------------------------
@torch.no_grad()
def weight_spectral_norms(model: nn.Module) -> dict:
    """Top singular value of every Linear weight + the product (Lipschitz bound, linear part).

    The product over linear layers upper-bounds the network's Lipschitz constant up to the
    nonlinearities (GELU ~1.085-Lipschitz, RMSNorm bounded) — a static, non-mutating snapshot
    of how much the map can amplify, the knob behind contraction.
    """
    per_layer: dict[str, float] = {}
    product = 1.0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            sn = float(torch.linalg.matrix_norm(mod.weight.detach(), ord=2).item())
            per_layer[name] = sn
            product *= sn
    return {
        "layer_spectral_norms": per_layer,
        "max_spectral_norm": float(max(per_layer.values())) if per_layer else 0.0,
        "lipschitz_product": float(product),
    }


# --------------------------------------------------------------------------------------
# Family A — fixed-point & spectral dynamics (recurrent arms only)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _unroll_states(model: nn.Module, X: torch.Tensor, horizon: int) -> list[tuple]:
    """Step the recurrent model one outer step at a time, returning the (z, a) state after
    each of ``horizon`` steps. Uses the M7 resumable API, which guarantees stepwise
    composition is bit-identical to a single ``n_steps=horizon`` unroll."""
    states = []
    _, _, (z, a) = model(X, n_steps=1, return_state=True)
    states.append((z, a))
    for _ in range(horizon - 1):
        _, _, (z, a) = model(X, n_steps=1, init_state=(z, a), return_state=True)
        states.append((z, a))
    return states


def _flat_per_row(z: torch.Tensor) -> torch.Tensor:
    return z.reshape(z.shape[0], -1)


def _readout_logits(model: nn.Module, a: torch.Tensor) -> torch.Tensor:
    """Shape the answer state into per-output-cell logits (B, out_features, num_classes).

    ``TRM`` carries the answer *flat* (B, w·C) in its state while ``TRMDecoupled`` already
    carries it as (B, w, C); both reduce to (B, C) for single-output. Mirrors the reshape
    ``TRM.forward`` applies to its returned logits.
    """
    if model.out_features is not None and a.ndim == 2:
        return a.reshape(a.shape[0], model.out_features, model.num_classes)
    return a


@torch.no_grad()
def latent_dynamics(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, trained_steps: int, horizon: int
) -> dict:
    """Latent residual trajectory + per-step readout metrics out to ``horizon`` steps.

    residual_t = mean_row ‖z_{t+1}−z_t‖₂ / ‖z_t‖₂  (convergent ⇒ → 0).
    Readout acc/EM at each step quantify the over-unroll decay at the answer level. The
    ``*_at_trained`` scalars read the trajectory at the depth the model was trained for;
    the ``*_overunroll`` scalars read the final (4×-ish) over-unrolled step.
    """
    model.eval()
    states = _unroll_states(model, X, horizon)
    zs = [s[0] for s in states]
    as_ = [s[1] for s in states]

    residuals = []
    for t in range(len(zs) - 1):
        z0 = _flat_per_row(zs[t])
        z1 = _flat_per_row(zs[t + 1])
        denom = z0.norm(dim=1).clamp_min(1e-12)
        residuals.append(float(((z1 - z0).norm(dim=1) / denom).mean().item()))

    yb = y.to(as_[0].device)
    multi = yb.ndim > 1
    acc_per_step, em_per_step = [], []
    for a in as_:
        preds = _readout_logits(model, a).argmax(dim=-1)
        acc_per_step.append(float((preds == yb).float().mean().item()))
        if multi:
            em_per_step.append(float((preds == yb).all(dim=-1).float().mean().item()))

    ti = min(trained_steps, horizon) - 1  # index of the trained-depth readout
    out = {
        "residual_trajectory": residuals,
        "acc_per_step": acc_per_step,
        # residual entering the trained-depth step, and at the over-unrolled tail
        "residual_at_trained": residuals[min(ti, len(residuals) - 1)] if residuals else 0.0,
        "residual_overunroll": residuals[-1] if residuals else 0.0,
        "acc_at_trained": acc_per_step[ti],
        "acc_overunroll": acc_per_step[-1],
        # decay = how much accuracy is lost by over-unrolling past the trained depth
        "acc_overunroll_drop": acc_per_step[ti] - acc_per_step[-1],
    }
    if multi:
        out["em_per_step"] = em_per_step
        out["em_at_trained"] = em_per_step[ti]
        out["em_overunroll"] = em_per_step[-1]
    return out


def _step_z_fn(model: nn.Module, X1: torch.Tensor, linearize_steps: int):
    """Return ``F`` and the linearization point ``z*`` for the one-step latent map of a single
    example. The asymptotic recurrence is z_{t+1} = update(cat[X, z_t, readout(z_t)]) = F(z_t)
    (the answer ``a`` is slaved to ``z`` after the first step), so ρ(∂F/∂z) governs convergence.

    ``linearize_steps`` is the over-unroll depth at which ``z*`` is taken — passed in (not the old
    hardcoded ``n_steps*4``) so it matches the horizon the residual / readout / path-independence
    probes use, keeping every Family-A diagnostic linearized at the SAME state.
    """
    with torch.no_grad():
        _, _, (z_star, _) = model(X1, n_steps=linearize_steps, return_state=True)
    z_shape = z_star.shape

    def F(z_flat: torch.Tensor) -> torch.Tensor:
        # One full OUTER step via the model's own forward — so the n_latent inner z-updates are
        # included (F is faithful for any n_latent, not just 1). ``a = readout(z)`` supplies the
        # steady-state answer the outer step conditions on (a_t = readout(z_{t-1}) past step 0).
        z = z_flat.reshape(z_shape)
        a = model.readout(z)
        _, _, (z2, _) = model(X1, n_steps=1, init_state=(z, a), return_state=True)
        return z2.reshape(-1)

    return F, z_star.reshape(-1).detach()


def _power_iteration_radius(matvec, d: int, k: int, gen: torch.Generator, dtype) -> float:
    """Estimate |λ_max| of the linear operator ``matvec`` (v ↦ J v) by power iteration.

    The dominant-eigenvalue magnitude is exp(mean log ‖J v_t‖) over the back half of the
    iteration (the per-step amplification of a unit vector converges to |λ_max|)."""
    v = torch.randn(d, generator=gen, dtype=dtype)
    v = v / v.norm().clamp_min(1e-12)
    gains = []
    for _ in range(k):
        Jv = matvec(v)
        g = float(Jv.norm().item())
        gains.append(g)
        if g < 1e-20:
            break
        v = Jv / g
    half = gains[len(gains) // 2 :] or gains
    return float(np.exp(np.mean(np.log(np.clip(half, 1e-20, None))))) if half else 0.0


def _power_iteration_opnorm(matvec, rmatvec, d: int, k: int, gen: torch.Generator, dtype) -> float:
    """Estimate σ_max(J) (largest singular value) by power iteration on JᵀJ.

    ``matvec`` = v ↦ J v, ``rmatvec`` = u ↦ Jᵀ u. σ_max < 1 is a *sufficient* condition for the
    map to be a 2-norm contraction."""
    v = torch.randn(d, generator=gen, dtype=dtype)
    v = v / v.norm().clamp_min(1e-12)
    sigma = 0.0
    for _ in range(k):
        w = rmatvec(matvec(v))
        nw = float(w.norm().item())
        if nw < 1e-20:
            return 0.0
        sigma = float(np.sqrt(nw))  # ‖JᵀJ v‖ ≈ σ² for unit v at convergence
        v = w / nw
    return sigma


def _spectral_radius_one(
    model: nn.Module, X1: torch.Tensor, k: int, gen: torch.Generator, linearize_steps: int
):
    """Power-iterate the Jacobian of the one-step latent map for ONE example.

    Returns (spectral_radius |λ_max(J)|, operator_norm σ_max(J)) of ∂F/∂z at the over-unrolled
    state, via autograd Jacobian-vector (jvp) and vector-Jacobian (vjp) products — no Jacobian is
    ever materialized. NOTE: the |λ_max| estimate is power-iteration on a generically NON-NORMAL
    Jacobian, so at finite ``k`` the magnitude is mildly UPPER-biased (a defective/Jordan block
    converges slowly); the bias is one-directional, so the ``ρ>1`` conclusion is conservative, but
    the precise magnitude and cross-regime ordering of ρ should be read as approximate (use a larger
    ``k`` to tighten). σ_max (JᵀJ power iteration) converges fast and is reliable at ``k≈20``.
    """
    F, z_star = _step_z_fn(model, X1, linearize_steps)
    d = z_star.numel()
    _, vjp_fn = torch.func.vjp(F, z_star)

    def J(v):
        return torch.func.jvp(F, (z_star,), (v,))[1]

    def Jt(u):
        return vjp_fn(u)[0]

    rho = _power_iteration_radius(J, d, k, gen, z_star.dtype)
    sigma = _power_iteration_opnorm(J, Jt, d, k, gen, z_star.dtype)
    return rho, sigma


def jacobian_spectrum(
    model: nn.Module,
    X: torch.Tensor,
    n_examples: int,
    power_iter_steps: int,
    seed: int,
    linearize_steps: int,
) -> dict:
    """Spectral radius / operator norm of the one-step latent map, averaged over examples."""
    model.eval()
    n = min(n_examples, X.shape[0])
    gen = torch.Generator().manual_seed(seed)
    rhos, sigmas = [], []
    for i in range(n):
        X1 = X[i : i + 1]
        rho, sigma = _spectral_radius_one(model, X1, power_iter_steps, gen, linearize_steps)
        rhos.append(rho)
        sigmas.append(sigma)
    return {
        "spectral_radius_mean": float(np.mean(rhos)),
        "spectral_radius_max": float(np.max(rhos)),
        "operator_norm_mean": float(np.mean(sigmas)),
        "operator_norm_max": float(np.max(sigmas)),
        "frac_expanding": float(np.mean([r > 1.0 for r in rhos])),  # ρ>1 ⇒ non-contractive
        "n_examples": n,
    }


@torch.no_grad()
def path_independence(
    model: nn.Module, X: torch.Tensor, n_inits: int, horizon: int, seed: int
) -> dict:
    """Asymptotic alignment: unroll deep from ``n_inits`` random z0 inits and measure whether
    they reach the same attractor (Anil 2022). High alignment ⇒ a genuine init-independent
    fixed point; low ⇒ a learned-depth circuit whose output depends on where it started.

    Returns mean pairwise cosine of the final ``z`` across inits (per row, averaged) and the
    fraction of rows whose final readout agrees across all inits.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    # Default state shapes + a scale for the random inits comparable to the learned trajectory.
    _, _, (z_def, a_def) = model(X, n_steps=1, return_state=True)
    scale = float(z_def.std().clamp_min(1e-6).item())

    finals_z, finals_pred = [], []
    for _ in range(n_inits):
        z0 = torch.randn(z_def.shape, generator=gen, dtype=z_def.dtype) * scale
        a0 = torch.zeros_like(a_def)
        _, _, (zf, af) = model(X, n_steps=horizon, init_state=(z0, a0), return_state=True)
        finals_z.append(_flat_per_row(zf))
        finals_pred.append(_readout_logits(model, af).argmax(dim=-1))

    U = torch.stack(finals_z, dim=0)  # (K, B, D)
    U = U / U.norm(dim=2, keepdim=True).clamp_min(1e-12)
    sims = torch.einsum("ibd,jbd->bij", U, U)  # (B, K, K)
    K = U.shape[0]
    if K > 1:
        offdiag = (sims.sum(dim=(1, 2)) - K) / (K * (K - 1))  # mean pairwise cosine per row
        align = float(offdiag.mean().item())
    else:
        align = 1.0

    preds = torch.stack(finals_pred, dim=0)  # (K, B, ...) row preds across inits
    agree_rows = (preds == preds[0:1]).reshape(K, preds.shape[1], -1).all(dim=(0, 2))
    readout_agreement = float(agree_rows.float().mean().item())

    return {
        "za_alignment": align,
        "readout_agreement": readout_agreement,
        "n_inits": K,
    }


# --------------------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------------------
def run_introspection(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    overunroll_factor: int = 4,
    n_random_inits: int = 5,
    power_iter_steps: int = 20,
    jac_n_examples: int = 8,
    seed: int = 0,
) -> dict:
    """Run the applicable diagnostic families for one trained arm on a fixed batch.

    Controls (feedforward / untied) get Families B + C; the recurrent arms (``trm`` /
    ``trm_decoupled``) additionally get Family A (residual trajectory, per-step readout,
    Jacobian spectrum, path independence). Returns a flat dict of scalars plus a few
    list-valued trajectories; all reproducible for a fixed ``seed``.
    """
    X, y = batch
    was_training = model.training  # restore the arm's mode on exit (no training side-effect)
    out: dict = {}
    try:
        out.update(representation_geometry(model, X))  # B
        out.update(weight_spectral_norms(model))  # C
        if is_recurrent(model):
            horizon = max(2, overunroll_factor * model.n_steps)
            out.update(latent_dynamics(model, X, y, trained_steps=model.n_steps, horizon=horizon))
            out.update(
                jacobian_spectrum(
                    model,
                    X,
                    n_examples=jac_n_examples,
                    power_iter_steps=power_iter_steps,
                    seed=seed,
                    linearize_steps=horizon,  # linearize ρ at the SAME state the probes use
                )
            )
            out.update(
                path_independence(model, X, n_inits=n_random_inits, horizon=horizon, seed=seed + 1)
            )
    finally:
        model.train(was_training)
    return out
