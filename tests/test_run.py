"""Tests for the run harness: arms, sweep, Δ reporting, and determinism."""


from looptab.config import ExperimentConfig
from looptab.run import run_point


def _cfg(**over):
    base = dict(
        task=dict(
            name="parity",
            params={"d": 12, "k": 2},
            n_train=400,
            n_test=200,
            task_seed=42,
            train_sample_seed=1,
            test_sample_seed=2,
        ),
        arms=[
            dict(name="trm", label="trm_ds", hidden_dim=16, latent_dim=16, n_steps=3,
                 deep_supervision=True, deep_supervision_weight=1.0),
            dict(name="trm", label="trm_nods", hidden_dim=16, latent_dim=16, n_steps=3,
                 deep_supervision=False, deep_supervision_weight=0.0),
            dict(name="ff_matched", label="ff_matched", hidden_dim=16, latent_dim=16, n_steps=3),
        ],
        train=dict(epochs=3, lr=1e-3, batch_size=128, device="cpu"),
        seeds=[0, 1],
    )
    base.update(over)
    return ExperimentConfig(**base)


def test_run_point_returns_all_arms():
    cfg = _cfg()
    out = run_point(cfg, cfg.task.params, seed=0)
    assert set(out.keys()) == {"trm_ds", "trm_nods", "ff_matched"}
    for v in out.values():
        assert "accuracy" in v and "exact_match" in v and "n_params" in v


def test_run_point_deterministic():
    """C3/§5.3: same seed => identical metrics, bit for bit."""
    cfg = _cfg()
    a = run_point(cfg, cfg.task.params, seed=0)
    b = run_point(cfg, cfg.task.params, seed=0)
    for lbl in a:
        assert a[lbl]["accuracy"] == b[lbl]["accuracy"]
        assert a[lbl]["exact_match"] == b[lbl]["exact_match"]


def test_arm_init_independent_of_order():
    """C3: reseeding before each arm => an arm's result is independent of which
    arms ran before it. Reversing arm order must not change a shared arm's metric."""
    cfg = _cfg()
    rev = _cfg()
    rev.arms = list(reversed(rev.arms))
    out = run_point(cfg, cfg.task.params, seed=0)
    out_rev = run_point(rev, rev.task.params, seed=0)
    assert out["ff_matched"]["accuracy"] == out_rev["ff_matched"]["accuracy"]
    assert out["trm_ds"]["accuracy"] == out_rev["trm_ds"]["accuracy"]


def test_function_varies_across_seeds():
    """I1: different outer seeds use different task_seeds => different informative
    bits => the parity functions differ, so metrics generally differ across seeds."""
    cfg = _cfg()
    s0 = run_point(cfg, cfg.task.params, seed=0)
    s1 = run_point(cfg, cfg.task.params, seed=1)
    # At least one arm should land on a different accuracy (different function).
    assert any(s0[lbl]["accuracy"] != s1[lbl]["accuracy"] for lbl in s0)


def test_resolved_deltas_default():
    cfg = _cfg()
    cfg.deltas = None
    # default: every non-last arm diffed against the last (control)
    assert cfg.resolved_deltas() == [["trm_ds", "ff_matched"], ["trm_nods", "ff_matched"]]
