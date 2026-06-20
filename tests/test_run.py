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
            dict(
                name="trm",
                label="trm_ds",
                hidden_dim=16,
                latent_dim=16,
                n_steps=3,
                deep_supervision=True,
                deep_supervision_weight=1.0,
            ),
            dict(
                name="trm",
                label="trm_nods",
                hidden_dim=16,
                latent_dim=16,
                n_steps=3,
                deep_supervision=False,
                deep_supervision_weight=0.0,
            ),
            dict(name="ff_matched", label="ff_matched", hidden_dim=16, latent_dim=16, n_steps=3),
        ],
        train=dict(epochs=3, lr=1e-3, batch_size=128, device="cpu"),
        seeds=[0, 1],
    )
    base.update(over)
    return ExperimentConfig(**base)


def test_run_point_returns_all_arms():
    cfg = _cfg()
    out, models, baseline = run_point(cfg, cfg.task.params, seed=0)
    assert set(out.keys()) == {"trm_ds", "trm_nods", "ff_matched"}
    for v in out.values():
        assert "accuracy" in v and "n_params" in v
    assert set(models.keys()) == {"trm_ds", "trm_nods", "ff_matched"}
    assert 0.0 <= baseline <= 1.0


def test_exact_match_suppressed_for_single_output():
    """Parity is single-output: exact_match == accuracy, so it isn't reported."""
    cfg = _cfg()
    out, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for v in out.values():
        assert "exact_match" not in v


def test_run_point_deterministic():
    """C3/§5.3: same seed => identical metrics, bit for bit."""
    cfg = _cfg()
    a, _, _ = run_point(cfg, cfg.task.params, seed=0)
    b, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for lbl in a:
        assert a[lbl]["accuracy"] == b[lbl]["accuracy"]


def test_arm_init_independent_of_order():
    """C3: reseeding before each arm => an arm's result is independent of which
    arms ran before it. Reversing arm order must not change a shared arm's metric."""
    cfg = _cfg()
    rev = _cfg()
    rev.arms = list(reversed(rev.arms))
    out, _, _ = run_point(cfg, cfg.task.params, seed=0)
    out_rev, _, _ = run_point(rev, rev.task.params, seed=0)
    assert out["ff_matched"]["accuracy"] == out_rev["ff_matched"]["accuracy"]
    assert out["trm_ds"]["accuracy"] == out_rev["trm_ds"]["accuracy"]


def test_function_varies_across_seeds():
    """I1: different outer seeds use different task_seeds => different informative
    bits => the parity functions differ, so metrics generally differ across seeds."""
    cfg = _cfg()
    s0, _, _ = run_point(cfg, cfg.task.params, seed=0)
    s1, _, _ = run_point(cfg, cfg.task.params, seed=1)
    # At least one arm should land on a different accuracy (different function).
    assert any(s0[lbl]["accuracy"] != s1[lbl]["accuracy"] for lbl in s0)


def test_run_point_multi_output():
    cfg = _cfg()
    cfg.task.name = "iterated"
    cfg.task.params = {"w": 8, "T": 2, "rule": 90, "distractors": 2}
    out, models, baseline = run_point(cfg, cfg.task.params, seed=0)
    for lbl in out:
        assert "exact_match" in out[lbl]
        assert "accuracy" in out[lbl]
        assert out[lbl]["accuracy"] >= 0.0
    assert 0.0 <= baseline <= 1.0


def test_resolved_deltas_default():
    cfg = _cfg()
    cfg.deltas = None
    # default: every non-last arm diffed against the last (control)
    assert cfg.resolved_deltas() == [
        ["trm_ds", "ff_matched"],
        ["trm_nods", "ff_matched"],
    ]


def _untied_arm(cfg):
    return cfg.arms[0].model_copy(
        update={
            "name": "untied_stack",
            "label": "untied_stack",
            "deep_supervision": False,
            "deep_supervision_weight": 0.0,
        }
    )


def test_run_point_with_untied_stack_arm():
    """M2: the untied-stack control (§4b) is a first-class arm the runner trains."""
    cfg = _cfg()
    cfg.arms.append(_untied_arm(cfg))
    out, models, _ = run_point(cfg, cfg.task.params, seed=0)
    assert "untied_stack" in out
    assert out["untied_stack"]["accuracy"] >= 0.0
    # §4b: depth/compute-matched, not param-matched — it has more params than the loop.
    assert out["untied_stack"]["n_params"] > out["trm_nods"]["n_params"]


def _untied_matched_arm(cfg):
    return cfg.arms[0].model_copy(
        update={
            "name": "untied_matched",
            "label": "untied_matched",
            "deep_supervision": False,
            "deep_supervision_weight": 0.0,
        }
    )


def test_untied_matched_arm_is_param_matched_in_runner():
    """The clean control reports a param count close to the loop's (not ~4×)."""
    cfg = _cfg()
    cfg.arms.append(_untied_matched_arm(cfg))
    out, _, _ = run_point(cfg, cfg.task.params, seed=0)
    ratio = out["untied_matched"]["n_params"] / out["trm_nods"]["n_params"]
    assert 0.8 <= ratio <= 1.2, f"untied_matched/loop param ratio = {ratio:.3f}"


def test_untied_arms_deterministic():
    """§5.3: both untied controls reproduce bit-for-bit at a fixed seed."""
    cfg = _cfg()
    cfg.arms.append(_untied_arm(cfg))
    cfg.arms.append(_untied_matched_arm(cfg))
    a, _, _ = run_point(cfg, cfg.task.params, seed=0)
    b, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for lbl in ("untied_stack", "untied_matched"):
        assert a[lbl]["accuracy"] == b[lbl]["accuracy"]


def test_untied_stack_routed_as_fixed_depth_in_extrapolation():
    """The untied stack has fixed depth, so the extrapolation harness must evaluate it
    once and hold it flat across R' (like ff_matched), never over-unrolling it."""
    cfg = _cfg()
    cfg.task.name = "iterated"
    cfg.task.params = {"w": 8, "T": 2, "rule": 90, "distractors": 2}
    cfg.arms.append(_untied_arm(cfg))
    _, models, _ = run_point(cfg, cfg.task.params, seed=0)

    from looptab.run import run_extrapolation_point

    extrap_out, _ = run_extrapolation_point(
        cfg,
        cfg.task.params,
        seed=0,
        models=models,
        T_test=2,
        R_test_values=[3, 5],
        device="cpu",
    )
    # Fixed-depth arm: identical accuracy across all R' (evaluated once, copied).
    assert (
        extrap_out[("untied_stack", 3)]["accuracy"] == extrap_out[("untied_stack", 5)]["accuracy"]
    )


def test_extrapolation_harness_determinism():
    """Verify that run_extrapolation_point at T_test=T_train produces identical
    metrics as the main run_point loop, confirming seed alignment."""
    cfg = _cfg()
    cfg.task.name = "iterated"
    cfg.task.params = {"w": 8, "T": 2, "rule": 90, "distractors": 2}

    out_main, models, baseline_main = run_point(cfg, cfg.task.params, seed=0)

    from looptab.run import run_extrapolation_point

    # R_test MUST equal the arms' trained n_steps (3, from _cfg) for the accuracy
    # equality below to hold: run_point evaluates at the default n_steps, so the
    # extrapolation pass only reproduces it when unrolled to the same depth. This
    # isolates the seed/data-alignment property, not an unroll-invariance one.
    extrap_out, baseline = run_extrapolation_point(
        cfg,
        cfg.task.params,
        seed=0,
        models=models,
        T_test=2,
        R_test_values=[3],
        device="cpu",
    )

    assert baseline == baseline_main
    assert extrap_out[("trm_ds", 3)]["accuracy"] == out_main["trm_ds"]["accuracy"]
    assert extrap_out[("trm_nods", 3)]["accuracy"] == out_main["trm_nods"]["accuracy"]
