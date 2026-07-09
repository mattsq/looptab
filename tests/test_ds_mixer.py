"""M29: the canonical detached-carry deep-supervision routine (``train_deep_supervision``, M18
ingredient 1) re-tested on the cross-cell mixer (``trm_mixer``).

M18 found the detached-carry DS *mechanism* essentially inert on the FLAT ``trm`` (the win was just
more optimization). Since ``trm_mixer`` — the architecture where recurrence demonstrably does
algorithmic work — landed, DS has only ever been held FIXED in the recipe, never ablated. Running
the M18 decomposition on the mixer needs the ``n_sup>1`` dispatch gate (run.py) to admit
``trm_mixer``. These tests guard that the routine actually WORKS on the mixer (its per-cell
``(z,a)`` state carries + detaches correctly), is deterministic at a pinned thread count (the
3-D-matmul caveat, as ``trm_decoupled`` / ``trm_stable``), that the ``carry`` flag is really wired,
and that the gate still rejects the non-loop controls — so a cold agent can trust an M29 null.
"""

import pytest
import torch

from looptab.config import ExperimentConfig
from looptab.data.dataset import make_loaders, make_splits
from looptab.models.mixer import TRMMixer
from looptab.run import run_point
from looptab.train.loop import train_deep_supervision

torch.set_num_threads(1)  # bit-repro for the mixer's 3-D matmuls needs threads pinned (docstring)


def _converge_loaders(w=24, n_train=800, n_test=400, seed=0):
    tr, te = make_splits(
        task="converge",
        task_cfg={"w": w, "T": 6, "rule": 78, "distractors": 0},  # distractors=0 ⇒ mixer-divisible
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
    return TRMMixer(
        in_features=inf, num_classes=2, hidden_dim=96, latent_dim=64, token_hidden=48,
        n_steps=6, deep_supervision=False, out_features=of, use_rmsnorm=True,
    )


def test_train_deep_supervision_runs_on_mixer():
    """The detached-carry routine drives the mixer's init_state/return_state API and its per-cell
    (z,a) state carries across passes without a shape error — finite losses over n_sup=4 passes."""
    trl, _, inf, of = _converge_loaders()
    m = _build(inf, of)
    losses = train_deep_supervision(m, trl, n_sup=4, carry=True, epochs=5)
    assert len(losses) == 5
    assert all(isinstance(x, float) and x == x for x in losses)  # finite (x==x rejects NaN)


def test_ds_mixer_deterministic():
    """Same seed + pinned threads ⇒ bit-identical weights (the 3-D-matmul determinism contract)."""
    trl, _, inf, of = _converge_loaders()
    a = _build(inf, of)
    train_deep_supervision(a, trl, n_sup=4, carry=True, epochs=6)
    b = _build(inf, of)
    train_deep_supervision(b, trl, n_sup=4, carry=True, epochs=6)
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


def test_carry_flag_is_wired_on_mixer():
    """carry=True (detached-carry mechanism) vs carry=False (fresh z0 each pass, the compute-matched
    control) must produce DIFFERENT weights — else the M29 mechanism delta measures nothing."""
    trl, _, inf, of = _converge_loaders()
    a = _build(inf, of)
    train_deep_supervision(a, trl, n_sup=4, carry=True, epochs=6)
    b = _build(inf, of)
    train_deep_supervision(b, trl, n_sup=4, carry=False, epochs=6)
    assert any(not torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


def _ds_cfg(arm_name, n_sup=4, **arm_over):
    """A minimal converge run with one N_sup arm (mixer-divisible: distractors=0, w=out)."""
    arm = dict(name=arm_name, label=arm_name, hidden_dim=32, latent_dim=32, n_steps=4, n_sup=n_sup)
    arm.update(arm_over)
    return ExperimentConfig(
        task=dict(
            name="converge",
            params={"w": 12, "T": 6, "rule": 78, "distractors": 0},
            n_train=400, n_test=200, task_seed=42, train_sample_seed=1, test_sample_seed=2,
        ),
        arms=[arm],
        train=dict(epochs=2, lr=1e-3, batch_size=128, device="cpu", num_threads=1),
        seeds=[0],
    )


def test_run_point_dispatches_nsup_on_mixer():
    """End-to-end: trm_mixer with n_sup>1 now routes through train_deep_supervision and returns a
    result (the gate admits it after the M29 allowlist widening)."""
    cfg = _ds_cfg("trm_mixer", token_hidden=8)
    out, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    assert "trm_mixer" in out and "exact_match" in out["trm_mixer"]


def test_nsup_guard_still_rejects_feedforward():
    """The non-loop controls lack the init_state/return_state API — n_sup>1 on them must still raise
    loudly rather than crash inside the routine (the S3 guard, unchanged by the mixer widening)."""
    cfg = _ds_cfg("ff_matched")
    with pytest.raises(ValueError, match="detached-carry"):
        run_point(cfg, cfg.task.params, seed=0)
