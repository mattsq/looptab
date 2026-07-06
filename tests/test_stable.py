"""M27: the contraction-regularized loop (`trm_stable` via ``train_stable``).

Guards the three properties the milestone rests on: (1) the penalty backpropagates and actually
lowers the M21-measured spectral radius of the one-step latent map; (2) it is deterministic at a
pinned thread count (the same float-reduction caveat as ``trm_decoupled`` / the diagnostics); and
(3) it does not collapse accuracy at a moderate weight. These are the checks that let a cold agent
trust a null result ("even a contractive loop doesn't extrapolate") as real, not a broken routine.
"""

import torch

from looptab.data.dataset import make_loaders, make_splits
from looptab.eval.introspection import jacobian_spectrum
from looptab.eval.metrics import accuracy, evaluate
from looptab.models.trm import TRM
from looptab.train.loop import train, train_stable

torch.set_num_threads(1)  # bit-repro for the jvp-based penalty needs threads pinned (see docstring)


def _converge_loaders(w=24, n_train=1000, n_test=400, seed=0):
    tr, te = make_splits(
        task="converge",
        task_cfg={"w": w, "T": 6, "rule": 78, "distractors": 0},
        task_seed=42,
        train_sample_seed=1,
        test_sample_seed=2,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    trl, tel = make_loaders(tr, te, 256)
    return trl, tel, int(tr[0][0].shape[0]), int(tr.y.shape[-1])


def _build(inf, of):
    torch.manual_seed(0)
    return TRM(
        in_features=inf, num_classes=2, hidden_dim=64, latent_dim=64,
        n_steps=6, deep_supervision=True, out_features=of,
    )


def test_train_stable_runs():
    trl, _, inf, of = _converge_loaders()
    m = _build(inf, of)
    losses = train_stable(m, trl, jac_reg_weight=1e-2, epochs=5, reg_seed=0)
    assert len(losses) == 5
    assert all(isinstance(x, float) for x in losses)


def test_train_stable_deterministic():
    """Same seed + pinned threads ⇒ bit-identical weights (the M27 determinism contract)."""
    trl, _, inf, of = _converge_loaders()
    # Mirror the runner's per-arm protocol: reseed (via _build's manual_seed) immediately before
    # each train call so both start from an identical global RNG state (the loader shuffle draws
    # from it), exactly as run.py does with `torch.manual_seed(seed)` before every arm.
    a = _build(inf, of)
    train_stable(a, trl, jac_reg_weight=1e-2, fixed_point_weight=1e-2, epochs=6, reg_seed=0)
    b = _build(inf, of)
    train_stable(b, trl, jac_reg_weight=1e-2, fixed_point_weight=1e-2, epochs=6, reg_seed=0)
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


def test_jac_penalty_lowers_spectral_radius():
    """The Jacobian penalty must move the actual M21 diagnostic: a nonzero weight yields a smaller
    spectral radius / operator norm of the one-step latent map than the unregularized loop."""
    trl, tel, inf, of = _converge_loaders()
    Xb, _ = next(iter(tel))
    plain = _build(inf, of)
    train(plain, trl, epochs=15)
    stable = _build(inf, of)
    train_stable(stable, trl, jac_reg_weight=1e-1, epochs=15, reg_seed=0)

    js_plain = jacobian_spectrum(plain, Xb, n_examples=8, power_iter_steps=20, seed=0,
                                 linearize_steps=24)
    js_stable = jacobian_spectrum(stable, Xb, n_examples=8, power_iter_steps=20, seed=0,
                                  linearize_steps=24)
    assert js_stable["spectral_radius_mean"] < js_plain["spectral_radius_mean"]
    assert js_stable["operator_norm_mean"] < js_plain["operator_norm_mean"]


def test_train_stable_preserves_accuracy():
    """A moderate contraction weight should not collapse per-cell accuracy (the cost axis is EM/
    coherence, measured separately; token accuracy should stay well above chance)."""
    trl, tel, inf, of = _converge_loaders()
    m = _build(inf, of)
    train_stable(m, trl, jac_reg_weight=1e-2, epochs=30, reg_seed=0)
    assert accuracy(m, trl) > 0.6
    assert evaluate(m, tel, "cpu", want_exact_match=True)["accuracy"] > 0.6
