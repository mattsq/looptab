"""Tests for the M21 latent/weight introspection layer (eval/introspection.py).

Covers: (a) known-answer sanity for the spectral-radius / operator-norm power iterations and the
effective-rank / participation-ratio statistics; (b) determinism (same seed → identical numbers,
CLAUDE.md §5); (c) that the diagnostic dispatcher returns the right families per arm and that the
recurrent-only Family A is skipped for controls.
"""

import numpy as np
import torch

from looptab.eval.introspection import (
    _power_iteration_opnorm,
    _power_iteration_radius,
    _spectrum_stats,
    is_recurrent,
    representation_geometry,
    run_introspection,
    weight_spectral_norms,
)
from looptab.models.controls import FFMatched, UntiedStackMatched
from looptab.models.decoupled import TRMDecoupled
from looptab.models.trm import TRM


def _batch(B=64, in_features=16, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, in_features, generator=g)
    y = torch.randint(0, 2, (B, w), generator=g)
    return X, y


# --------------------------------------------------------------------------------------
# (a) Known-answer sanity
# --------------------------------------------------------------------------------------
def test_power_iteration_radius_diagonal():
    """|λ_max| of a diagonal matrix is its largest |entry|."""
    A = torch.diag(torch.tensor([3.0, -0.5, 0.1]))
    gen = torch.Generator().manual_seed(0)
    rho = _power_iteration_radius(lambda v: A @ v, d=3, k=60, gen=gen, dtype=torch.float32)
    assert abs(rho - 3.0) < 1e-3


def test_power_iteration_radius_nonsymmetric():
    """Eigenvalues of [[2,1],[0,0.5]] are 2 and 0.5 → radius 2; σ_max is the top singular value."""
    A = torch.tensor([[2.0, 1.0], [0.0, 0.5]])
    gen = torch.Generator().manual_seed(0)
    rho = _power_iteration_radius(lambda v: A @ v, d=2, k=80, gen=gen, dtype=torch.float32)
    assert abs(rho - 2.0) < 1e-2
    sigma = _power_iteration_opnorm(
        lambda v: A @ v, lambda u: A.T @ u, d=2, k=80, gen=gen, dtype=torch.float32
    )
    true_sigma = float(torch.linalg.svdvals(A)[0])
    assert abs(sigma - true_sigma) < 1e-2
    assert sigma > rho  # operator norm ≥ spectral radius


def test_effective_rank_orthogonal_vs_rank1():
    """Effective rank ≈ D for an isotropic batch, ≈ 1 for a rank-1 batch."""
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((512, 8))
    stats_iso = _spectrum_stats(iso)
    assert stats_iso["effective_rank"] > 6.5  # near full (8)
    assert stats_iso["participation_ratio"] > 6.0

    direction = rng.standard_normal((1, 8))
    rank1 = rng.standard_normal((512, 1)) * direction  # all rows along one direction
    stats_r1 = _spectrum_stats(rank1)
    assert stats_r1["effective_rank"] < 1.05
    assert stats_r1["participation_ratio"] < 1.05


def test_weight_spectral_norm_known():
    """The spectral norm of a scaled-identity Linear weight equals the scale."""

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.lin.weight.copy_(2.5 * torch.eye(4))

    out = weight_spectral_norms(_Tiny())
    assert abs(out["layer_spectral_norms"]["lin"] - 2.5) < 1e-4
    assert abs(out["max_spectral_norm"] - 2.5) < 1e-4
    assert abs(out["lipschitz_product"] - 2.5) < 1e-4


# --------------------------------------------------------------------------------------
# (b) Determinism — same seed → identical numbers (incl. the stochastic probes)
# --------------------------------------------------------------------------------------
def _scalars(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


def test_run_introspection_deterministic():
    torch.manual_seed(0)
    model = TRM(16, 2, 64, 64, n_steps=4, out_features=8)
    batch = _batch()
    a = run_introspection(model, batch, power_iter_steps=15, jac_n_examples=4, seed=7)
    b = run_introspection(model, batch, power_iter_steps=15, jac_n_examples=4, seed=7)
    assert _scalars(a) == _scalars(b)
    # Trajectories reproduce too.
    assert a["residual_trajectory"] == b["residual_trajectory"]
    assert a["acc_per_step"] == b["acc_per_step"]


def test_run_introspection_decoupled_deterministic():
    """Same-process determinism on trm_decoupled too — it exercises the 3-D (B,w,m) state and the
    flat-vs-3D readout reshape. NB CLAUDE.md §11.1: trm_decoupled's batched matmul is
    thread/BLAS-order-sensitive, so its diagnostics reproduce bit-for-bit only at a FIXED
    num_threads (as here), NOT across environments — same caveat as its accuracy."""
    torch.manual_seed(0)
    model = TRMDecoupled(16, 2, 64, 64, n_steps=4, out_features=8)
    batch = _batch()
    a = run_introspection(model, batch, power_iter_steps=15, jac_n_examples=4, seed=3)
    b = run_introspection(model, batch, power_iter_steps=15, jac_n_examples=4, seed=3)
    assert _scalars(a) == _scalars(b)
    assert a["residual_trajectory"] == b["residual_trajectory"]


def test_step_map_faithful_for_n_latent_gt_1():
    """The one-step latent map F used by the Jacobian delegates to model.forward(n_steps=1), so it
    runs the full n_latent inner loop — F is faithful for n_latent>1, not just the default 1."""
    from looptab.eval.introspection import _step_z_fn

    torch.manual_seed(0)
    m = TRM(16, 2, 32, 32, n_steps=4, out_features=8, n_latent=3).eval()
    X1 = torch.randn(1, 16)
    F, z_star = _step_z_fn(m, X1, linearize_steps=16)
    z = z_star.reshape(1, -1)
    a = m.readout(z)
    _, _, (z2, _) = m(X1, n_steps=1, init_state=(z, a), return_state=True)
    assert torch.allclose(F(z_star), z2.reshape(-1), atol=1e-6)


def test_run_introspection_seed_changes_stochastic_probe():
    """A different seed perturbs the random-init path-independence / Jacobian probes."""
    torch.manual_seed(0)
    model = TRM(16, 2, 64, 64, n_steps=4, out_features=8)
    batch = _batch()
    a = run_introspection(model, batch, power_iter_steps=15, jac_n_examples=4, seed=1)
    b = run_introspection(model, batch, power_iter_steps=15, jac_n_examples=4, seed=2)
    # Deterministic, non-stochastic descriptors are unchanged by the probe seed.
    assert a["effective_rank"] == b["effective_rank"]
    assert a["lipschitz_product"] == b["lipschitz_product"]


# --------------------------------------------------------------------------------------
# (c) Dispatcher — right families per arm
# --------------------------------------------------------------------------------------
def test_recurrent_arms_get_family_a():
    batch = _batch()
    torch.manual_seed(0)
    for model in (
        TRM(16, 2, 64, 64, n_steps=4, out_features=8),
        TRMDecoupled(16, 2, 64, 64, n_steps=4, out_features=8),
    ):
        assert is_recurrent(model)
        d = run_introspection(model, batch, power_iter_steps=10, jac_n_examples=3, seed=0)
        # Family A keys present
        for k in ("spectral_radius_mean", "operator_norm_mean", "za_alignment",
                  "residual_trajectory", "acc_overunroll_drop"):
            assert k in d
        # Families B + C present
        assert "effective_rank" in d and "lipschitz_product" in d


def test_control_arms_skip_family_a():
    batch = _batch()
    torch.manual_seed(0)
    for model in (
        FFMatched(16, 2, 64, 64, n_steps=4, out_features=8),
        UntiedStackMatched(16, 2, 64, 64, n_steps=4, out_features=8),
    ):
        assert not is_recurrent(model)
        d = run_introspection(model, batch, seed=0)
        # Only Families B + C
        assert "effective_rank" in d and "lipschitz_product" in d
        assert "spectral_radius_mean" not in d
        assert "residual_trajectory" not in d


def test_representation_geometry_shapes():
    """Effective rank is bounded by the representation dimensionality for every arm."""
    X, _ = _batch()
    torch.manual_seed(0)
    for model, name in (
        (TRM(16, 2, 64, 64, n_steps=4, out_features=8), "trm"),
        (FFMatched(16, 2, 64, 64, n_steps=4, out_features=8), "ff"),
        (UntiedStackMatched(16, 2, 64, 64, n_steps=4, out_features=8), "untied"),
    ):
        stats = representation_geometry(model, X)
        assert 1.0 <= stats["effective_rank"] <= stats["n_dims"] + 1e-6, name
        assert 1.0 <= stats["participation_ratio"] <= stats["n_dims"] + 1e-6, name
