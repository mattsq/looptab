"""Tests for the run harness: arms, sweep, Δ reporting, and determinism."""

from pathlib import Path

import pytest
import torch
import yaml

from looptab.config import ExperimentConfig
from looptab.eval.metrics import evaluate
from looptab.run import budget_audit, run_point


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


class _FixedPredModel(torch.nn.Module):
    """Stub whose argmax over the class dim reproduces a preset (N, W) prediction array."""

    def __init__(self, preds, num_classes=2):
        super().__init__()
        onehot = torch.nn.functional.one_hot(torch.as_tensor(preds), num_classes).float()
        self._logits = onehot * 10.0  # (N, W, C); argmax(-1) == preds

    def forward(self, X, **kwargs):
        return self._logits, None


def _one_batch_loader(targets):
    y = torch.as_tensor(targets)
    X = torch.zeros(y.shape[0], 1)  # ignored by the stub
    return [(X, y)]


def test_coherence_excess_diagnostic_m9():
    """M9: coherence_excess = EM − token_acc**W. At matched token-acc, CLUSTERED errors give a
    POSITIVE excess (coherent whole rows) and SPREAD errors give a negative one — the mechanism
    test for the M8 tying-positive."""
    targets = [[0, 0], [0, 0], [0, 0], [0, 0]]  # 4 rows × 2 cells, all class 0

    # Clustered: one row entirely wrong, rest perfect. token_acc=0.75, EM=0.75.
    clustered = _FixedPredModel([[1, 1], [0, 0], [0, 0], [0, 0]])
    out_c = evaluate(clustered, _one_batch_loader(targets), want_exact_match=True)
    assert out_c["accuracy"] == pytest.approx(0.75)
    assert out_c["exact_match"] == pytest.approx(0.75)
    assert out_c["coherence_excess"] == pytest.approx(0.75 - 0.75**2)  # +0.1875
    assert out_c["coherence_excess"] > 0
    assert out_c["mean_wrong_per_row"] == pytest.approx(0.5)

    # Spread: same token_acc=0.75 but errors split across two rows → lower EM, negative excess.
    spread = _FixedPredModel([[1, 0], [1, 0], [0, 0], [0, 0]])
    out_s = evaluate(spread, _one_batch_loader(targets), want_exact_match=True)
    assert out_s["accuracy"] == pytest.approx(0.75)  # matched token-acc
    assert out_s["exact_match"] == pytest.approx(0.5)
    assert out_s["coherence_excess"] == pytest.approx(0.5 - 0.75**2)  # -0.0625
    assert out_c["coherence_excess"] > out_s["coherence_excess"]


def test_coherence_excess_absent_for_single_output():
    """Single-output (1-D targets): no whole-row notion, so coherence_excess isn't emitted."""
    targets = [0, 1, 0, 1]
    model = _FixedPredModel([[0], [1], [1], [1]]).eval()
    # 1-D targets path: build logits of shape (N, C) by squeezing the W=1 dim.
    model._logits = model._logits.squeeze(1)
    out = evaluate(model, _one_batch_loader(targets), want_exact_match=True)
    assert "coherence_excess" not in out
    assert out["exact_match"] == out["accuracy"]


def test_run_point_deterministic():
    """C3/§5.3: same seed => identical metrics, bit for bit."""
    cfg = _cfg()
    a, _, _ = run_point(cfg, cfg.task.params, seed=0)
    b, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for lbl in a:
        assert a[lbl]["accuracy"] == b[lbl]["accuracy"]


def test_parallel_seeds_bit_identical_to_serial():
    """§5.3: running seeds across worker processes must be bit-for-bit identical to serial —
    parallelism is a speed knob only (seeds are independent and self-reseed)."""
    from looptab.run import _compute_seeds

    seeds = [0, 1, 2]
    serial = _compute_seeds(_cfg(parallel_workers=1), _cfg().task.params, seeds)
    parallel = _compute_seeds(_cfg(parallel_workers=3), _cfg().task.params, seeds)
    assert len(serial) == len(parallel) == len(seeds)
    for (r_s, _), (r_p, _) in zip(serial, parallel):
        assert r_s.keys() == r_p.keys()
        for lbl in ("trm_ds", "trm_nods", "ff_matched"):
            assert r_s[lbl]["accuracy"] == r_p[lbl]["accuracy"]


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


def test_axis_points_single():
    """No sweep/grid => exactly one point with no overrides (CLAUDE.md axis_points)."""
    cfg = _cfg()
    assert cfg.axis_points() == [("single", {})]


def test_axis_points_sweep():
    cfg = _cfg(sweep=dict(param="k", values=[2, 3, 4]))
    assert cfg.axis_points() == [("k=2", {"k": 2}), ("k=3", {"k": 3}), ("k=4", {"k": 4})]


def test_axis_points_grid_is_cartesian_product():
    """M2-confirm: a grid replicates the factorial across a rule × w product."""
    cfg = _cfg(grid=dict(params={"rule": [30, 90], "w": [9, 13]}))
    pts = cfg.axis_points()
    assert [ov for _, ov in pts] == [
        {"rule": 30, "w": 9},
        {"rule": 30, "w": 13},
        {"rule": 90, "w": 9},
        {"rule": 90, "w": 13},
    ]
    # labels are human-readable and carry both axes
    assert pts[0][0] == "rule=30, w=9"


def test_sweep_and_grid_are_mutually_exclusive():
    with pytest.raises(ValueError):
        _cfg(sweep=dict(param="k", values=[2]), grid=dict(params={"w": [9]}))


def test_grid_and_extrapolation_are_mutually_exclusive():
    with pytest.raises(ValueError):
        _cfg(
            grid=dict(params={"w": [9]}),
            extrapolation=dict(T_values=[4], R_values=[4]),
        )


def test_grid_point_runs_factorial_with_overrides():
    """Each grid cell trains every arm at its own task config (overrides applied)."""
    cfg = _cfg(grid=dict(params={"k": [2, 3]}))
    for _, overrides in cfg.axis_points():
        params = {**cfg.task.params, **overrides}
        out, _, _ = run_point(cfg, params, seed=0)
        assert set(out.keys()) == {"trm_ds", "trm_nods", "ff_matched"}


def test_grid_cells_deterministic_and_independent():
    """§5.3/§5.8: a grid cell reproduces bit-for-bit, and cells don't leak state into
    each other — a cell's metrics are identical regardless of which cells ran before it
    (the override dict must not mutate cfg.task.params)."""
    cfg = _cfg(grid=dict(params={"k": [2, 4]}))
    points = cfg.axis_points()
    base_params = dict(cfg.task.params)

    def run_cell(overrides):
        return run_point(cfg, {**cfg.task.params, **overrides}, seed=0)[0]

    # Same cell twice => identical (determinism).
    a = run_cell(points[0][1])
    b = run_cell(points[0][1])
    for lbl in a:
        assert a[lbl]["accuracy"] == b[lbl]["accuracy"]

    # Running cell 1 in between must not change cell 0's result (independence) and must
    # not have mutated the shared task params.
    run_cell(points[1][1])
    c = run_cell(points[0][1])
    for lbl in a:
        assert a[lbl]["accuracy"] == c[lbl]["accuracy"]
    assert cfg.task.params == base_params


def test_m4_parity_grid_config_is_2d_d_by_k():
    """M4: the parity grid is the full d × k Cartesian product (3×3 = 9 cells), runs the
    four required arms + the labelled untied_stack ceiling, and wires the budget audit
    with the loop as reference and only untied_stack exempt."""
    path = Path(__file__).resolve().parents[1] / "configs/experiments/m4_parity_grid.yaml"
    with open(path) as f:
        cfg = ExperimentConfig(**yaml.safe_load(f))
    pts = cfg.axis_points()
    assert [ov for _, ov in pts] == [
        {"d": d, "k": k} for d in (20, 40, 80) for k in (3, 4, 5)
    ]
    labels = {a.resolved_label() for a in cfg.arms}
    assert {"trm_ds", "trm_nods", "ff_matched", "untied_matched", "untied_stack"} == labels
    assert cfg.budget_reference == "trm_nods"
    assert cfg.budget_ceiling == ["untied_stack"]
    # The four required M4 deltas must all be present.
    assert ["trm_nods", "ff_matched"] in cfg.deltas
    assert ["trm_nods", "untied_matched"] in cfg.deltas
    assert ["untied_matched", "ff_matched"] in cfg.deltas
    assert ["trm_ds", "trm_nods"] in cfg.deltas
    assert len(cfg.seeds) == 10


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


def _iter_cfg(**over):
    """A small iterated-CA experiment with the four M3a arms (loop + 3 controls)."""
    base = dict(
        task=dict(
            name="iterated",
            params={"w": 8, "rule": 30, "distractors": 2},
            n_train=300,
            n_test=200,
            task_seed=42,
            train_sample_seed=1,
            test_sample_seed=2,
        ),
        arms=[
            dict(
                name="trm",
                label="trm_nods",
                hidden_dim=24,
                latent_dim=24,
                n_steps=4,
                deep_supervision=False,
                deep_supervision_weight=0.0,
            ),
            dict(name="ff_matched", label="ff_matched", hidden_dim=24, latent_dim=24, n_steps=4),
            dict(
                name="untied_matched",
                label="untied_matched",
                hidden_dim=24,
                latent_dim=24,
                n_steps=4,
                deep_supervision=False,
                deep_supervision_weight=0.0,
            ),
            dict(
                name="untied_stack",
                label="untied_stack",
                hidden_dim=24,
                latent_dim=24,
                n_steps=4,
                deep_supervision=False,
                deep_supervision_weight=0.0,
            ),
        ],
        train=dict(epochs=2, lr=1e-3, batch_size=128, device="cpu"),
        seeds=[0, 1],
    )
    base.update(over)
    return ExperimentConfig(**base)


def test_couple_n_steps_sets_model_depth_to_T():
    """M3a: with `couple_n_steps_to_param: T`, each arm's unroll depth tracks the swept T,
    overriding its static n_steps. The loop unrolls T; the untied arms grow to T blocks."""
    cfg = _iter_cfg(
        grid=dict(params={"T": [3, 6]}),
        couple_n_steps_to_param="T",
    )
    for _, overrides in cfg.axis_points():
        params = {**cfg.task.params, **overrides}
        _, models, _ = run_point(cfg, params, seed=0)
        T = overrides["T"]
        # Loop: n_steps coupled to T despite arm config saying 4.
        assert models["trm_nods"].n_steps == T
        # Untied stack: one independent block per step => T blocks.
        assert len(models["untied_stack"].update_nets) == T
        assert len(models["untied_matched"].inner.update_nets) == T


def test_couple_n_steps_absent_uses_static_n_steps():
    """Without coupling, depth stays at the per-arm n_steps (no silent override)."""
    cfg = _iter_cfg(grid=dict(params={"T": [3]}))
    params = {**cfg.task.params, "T": 3}
    _, models, _ = run_point(cfg, params, seed=0)
    assert models["trm_nods"].n_steps == 4  # the arm's static value, not T


def test_train_accuracy_reported():
    """M3a diagnostic: train accuracy is captured per arm alongside test accuracy."""
    cfg = _iter_cfg(grid=dict(params={"T": [3]}), couple_n_steps_to_param="T")
    out, _, _ = run_point(cfg, {**cfg.task.params, "T": 3}, seed=0)
    for lbl in out:
        assert "train_accuracy" in out[lbl]
        assert 0.0 <= out[lbl]["train_accuracy"] <= 1.0


def _fake_points(ref_params, arm_params_by_cell):
    """Build the minimal `points` structure budget_audit consumes."""
    points = []
    for cell, arms in arm_params_by_cell.items():
        agg = {lbl: {"n_params": n} for lbl, n in arms.items()}
        points.append({"label": cell, "agg": agg, "overrides": {}})
    return points


def test_budget_audit_passes_within_tol_and_exempts_ceiling():
    cfg = _iter_cfg(
        budget_reference="trm_nods",
        budget_ceiling=["untied_stack"],
        budget_tol=0.02,
    )
    points = _fake_points(
        None,
        {
            "T=4": {
                "trm_nods": 1000,
                "ff_matched": 1010,  # +1% ok
                "untied_matched": 990,  # -1% ok
                "untied_stack": 4000,  # 4x but exempt
            }
        },
    )
    labels = ["trm_nods", "ff_matched", "untied_matched", "untied_stack"]
    audit = budget_audit(points, labels, cfg)
    assert audit["breaches"] == []
    roles = {r["arm"]: r["role"] for r in audit["rows"]}
    assert roles["trm_nods"] == "reference"
    assert roles["untied_stack"] == "ceiling"


def test_budget_audit_flags_matched_breach_only():
    """A matched arm out of tolerance is flagged; the ceiling never is, however large."""
    cfg = _iter_cfg(
        budget_reference="trm_nods",
        budget_ceiling=["untied_stack"],
        budget_tol=0.02,
    )
    points = _fake_points(
        None,
        {
            "T=16": {
                "trm_nods": 1000,
                "ff_matched": 1000,
                "untied_matched": 930,  # -7% => breach (expected high-T quantization finding)
                "untied_stack": 16000,  # 16x but exempt
            }
        },
    )
    labels = ["trm_nods", "ff_matched", "untied_matched", "untied_stack"]
    audit = budget_audit(points, labels, cfg)
    assert [b[1] for b in audit["breaches"]] == ["untied_matched"]


def test_run_point_curriculum_trains_all_arms():
    """M3b: with a curriculum, run_point trains every arm across a depth range and still
    reports test/train accuracy and exact-match. The step-aligned loop is just another arm."""
    cfg = _iter_cfg(
        curriculum=dict(param="T", T_min=1, T_max=4),
        couple_n_steps_to_param="T",
    )
    cfg.task.params = {"w": 8, "T": 4, "rule": 30, "distractors": 2}
    # Make the loop arm step-aligned with DS on; keep one final-state contrast arm.
    cfg.arms[0].deep_supervision = True
    cfg.arms[0].deep_supervision_weight = 1.0
    cfg.arms[0].ds_mode = "step_aligned"
    cfg.arms[0].label = "trm_stepDS"
    out, models, baseline = run_point(cfg, cfg.task.params, seed=0)
    assert set(out.keys()) == {"trm_stepDS", "ff_matched", "untied_matched", "untied_stack"}
    for lbl in out:
        assert "accuracy" in out[lbl] and "train_accuracy" in out[lbl]
        assert "exact_match" in out[lbl]
    # The step-aligned loop is built to unroll T_max (the reference eval depth).
    assert models["trm_stepDS"].n_steps == 4


def test_run_point_curriculum_deterministic():
    cfg = _iter_cfg(
        curriculum=dict(param="T", T_min=1, T_max=4),
        couple_n_steps_to_param="T",
    )
    cfg.task.params = {"w": 8, "T": 4, "rule": 30, "distractors": 2}
    cfg.arms[0].ds_mode = "step_aligned"
    cfg.arms[0].deep_supervision = True
    a, _, _ = run_point(cfg, cfg.task.params, seed=0)
    b, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for lbl in a:
        assert a[lbl]["accuracy"] == b[lbl]["accuracy"]


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
