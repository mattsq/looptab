"""Tests for the run harness: arms, sweep, Δ reporting, and determinism."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from looptab.config import ExperimentConfig
from looptab.eval.metrics import delta_report, evaluate
from looptab.run import _git_sha, budget_audit, cv_sign_test_status, run_point


def test_git_sha_marks_dirty_tree():
    """§5.7: a run record's git_sha must not claim reproducibility from a dirty tree.

    Mocked (not run against the real repo state) so this test is deterministic regardless of
    whether the working tree happens to be clean when the suite runs.
    """
    with patch("subprocess.check_output") as mock_run:
        mock_run.side_effect = [b"abc1234\n", b" M src/looptab/run.py\n"]
        assert _git_sha() == "abc1234-dirty"

    with patch("subprocess.check_output") as mock_run:
        mock_run.side_effect = [b"abc1234\n", b""]
        assert _git_sha() == "abc1234"


def test_git_sha_fails_conservatively_when_status_check_errors():
    """External review (post-M34): a swallowed `git status` failure must NOT silently report the
    bare (clean-looking) sha -- that recreates the exact false-clean state the dirty guard exists
    to prevent. It must report a distinct, honest "don't know" marker instead."""
    with patch("subprocess.check_output") as mock_run:
        mock_run.side_effect = [b"abc1234\n", subprocess.CalledProcessError(1, "git status")]
        assert _git_sha() == "abc1234-status-unknown"


def test_cv_sign_test_status_gates_on_unique_folds():
    # Synthetic tasks always qualify (fresh function + rows per seed).
    ok, _ = cv_sign_test_status("converge", {}, [0, 1, 2, 3, 4])
    assert ok
    # multilabel random-split mode (no n_folds) — overlapping test sets, suppressed.
    ok, why = cv_sign_test_status("multilabel", {"dataset": "yeast"}, list(range(10)))
    assert not ok and "overlap" in why
    # multilabel 10-fold CV with seeds 0..9 → each on a distinct disjoint fold → valid.
    ok, why = cv_sign_test_status("multilabel", {"n_folds": 10}, list(range(10)))
    assert ok and "DISJOINT" in why
    # COLLISION 1: more seeds than folds (seed 0 and 10 both → fold 0) → suppressed.
    ok, why = cv_sign_test_status("multilabel", {"n_folds": 10}, list(range(11)))
    assert not ok and "collide" in why
    # COLLISION 2: n_folds < #seeds → guaranteed collisions → suppressed.
    ok, _ = cv_sign_test_status("multilabel", {"n_folds": 5}, list(range(10)))
    assert not ok
    # PER-POINT params win over any base config: a grid cell that overrode n_folds away is honoured
    # here because the gate reads the passed task_params, not cfg.task.params.
    ok, _ = cv_sign_test_status("multilabel", {"n_folds": 4}, [0, 1, 2, 3])
    assert ok


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
    out, models, baselines, _ = run_point(cfg, cfg.task.params, seed=0)
    assert set(out.keys()) == {"trm_ds", "trm_nods", "ff_matched"}
    for v in out.values():
        assert "accuracy" in v and "n_params" in v
    assert set(models.keys()) == {"trm_ds", "trm_nods", "ff_matched"}
    assert set(baselines) == {"accuracy"}
    assert 0.0 <= baselines["accuracy"] <= 1.0


def test_exact_match_suppressed_for_single_output():
    """Parity is single-output: exact_match == accuracy, so it isn't reported."""
    cfg = _cfg()
    out, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
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


def test_coherence_excess_dispersion_confound_m9():
    """M9 review guard: coherence_excess's CROSS-ARM Δ is confounded by per-row difficulty
    dispersion, NOT just by token-acc level. Two arms at IDENTICAL mean token-acc (0.875, w=8)
    — one with homogeneous rows (1 wrong/row), one dispersed (half rows perfect, half with 2
    wrong/row) — give very different coherence_excess, purely from the Jensen gap (EM =
    mean_row(row_acc**w) ≥ (mean row_acc)**w). Because the token_acc**w baseline is identical at
    matched acc, Δ(coherence_excess) == Δ(exact_match): the metric adds NOTHING beyond EM here.
    This documents why the clean cross-arm statistic is EM-at-matched-token-acc, not a coh Δ."""
    targets = [[0] * 8 for _ in range(4)]
    homogeneous = _FixedPredModel([[1] + [0] * 7 for _ in range(4)])  # every row 1 wrong
    dispersed = _FixedPredModel(  # 2 perfect rows + 2 rows with 2 wrong
        [[0] * 8, [0] * 8, [1, 1] + [0] * 6, [1, 1] + [0] * 6]
    )
    out_h = evaluate(homogeneous, _one_batch_loader(targets), want_exact_match=True)
    out_d = evaluate(dispersed, _one_batch_loader(targets), want_exact_match=True)

    # (a) matched mean token-acc
    assert out_h["accuracy"] == pytest.approx(0.875)
    assert out_d["accuracy"] == pytest.approx(0.875)
    # (b) at matched acc, Δ(coherence_excess) is exactly Δ(exact_match) (baseline cancels)
    assert (out_d["coherence_excess"] - out_h["coherence_excess"]) == pytest.approx(
        out_d["exact_match"] - out_h["exact_match"]
    )
    # (c) dispersed arm shows far higher coherence_excess despite no clustering — the confound
    assert out_d["coherence_excess"] - out_h["coherence_excess"] == pytest.approx(0.5)


def test_run_point_deterministic():
    """C3/§5.3: same seed => identical metrics, bit for bit."""
    cfg = _cfg()
    a, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    b, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
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
    out, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    out_rev, _, _, _ = run_point(rev, rev.task.params, seed=0)
    assert out["ff_matched"]["accuracy"] == out_rev["ff_matched"]["accuracy"]
    assert out["trm_ds"]["accuracy"] == out_rev["trm_ds"]["accuracy"]


def test_function_varies_across_seeds():
    """I1: different outer seeds use different task_seeds => different informative
    bits => the parity functions differ, so metrics generally differ across seeds."""
    cfg = _cfg()
    s0, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    s1, _, _, _ = run_point(cfg, cfg.task.params, seed=1)
    # At least one arm should land on a different accuracy (different function).
    assert any(s0[lbl]["accuracy"] != s1[lbl]["accuracy"] for lbl in s0)


def test_run_point_multi_output():
    cfg = _cfg()
    cfg.task.name = "iterated"
    cfg.task.params = {"w": 8, "T": 2, "rule": 90, "distractors": 2}
    out, models, baselines, _ = run_point(cfg, cfg.task.params, seed=0)
    for lbl in out:
        assert "exact_match" in out[lbl]
        assert "accuracy" in out[lbl]
        assert out[lbl]["accuracy"] >= 0.0
    assert set(baselines) == {"accuracy", "exact_match"}
    assert 0.0 <= baselines["accuracy"] <= 1.0
    assert 0.0 <= baselines["exact_match"] <= 1.0


def test_aggregate_persists_per_seed_exact_match():
    """M29c audit-gap fix: EM (the headline metric on multi-output tasks) must be persisted per-seed
    so its sign-counts are re-derivable from the committed record, like accuracy already is."""
    from looptab.run import _aggregate

    per_seed = [
        {"m": {"accuracy": 0.9, "exact_match": 0.8, "n_params": 10}},
        {"m": {"accuracy": 0.92, "exact_match": 0.85, "n_params": 10}},
    ]
    agg = _aggregate(per_seed, ["m"])
    assert agg["m"]["accuracy_per_seed"] == [0.9, 0.92]
    assert agg["m"]["exact_match_per_seed"] == [0.8, 0.85]  # newly persisted


def test_delta_report_can_skip_sign_test_for_non_independent_splits():
    rep = delta_report(
        [0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3],
        paired_sign_test=False,
    )
    assert rep["sign_test"] is None
    assert rep["sign_test_note"] == "not_run_non_independent_splits"


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
        out, _, _, _ = run_point(cfg, params, seed=0)
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
    out, models, _, _ = run_point(cfg, cfg.task.params, seed=0)
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
    out, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    ratio = out["untied_matched"]["n_params"] / out["trm_nods"]["n_params"]
    assert 0.8 <= ratio <= 1.2, f"untied_matched/loop param ratio = {ratio:.3f}"


def test_untied_arms_deterministic():
    """§5.3: both untied controls reproduce bit-for-bit at a fixed seed."""
    cfg = _cfg()
    cfg.arms.append(_untied_arm(cfg))
    cfg.arms.append(_untied_matched_arm(cfg))
    a, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    b, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for lbl in ("untied_stack", "untied_matched"):
        assert a[lbl]["accuracy"] == b[lbl]["accuracy"]


def test_untied_stack_routed_as_fixed_depth_in_extrapolation():
    """The untied stack has fixed depth, so the extrapolation harness must evaluate it
    once and hold it flat across R' (like ff_matched), never over-unrolling it."""
    cfg = _cfg()
    cfg.task.name = "iterated"
    cfg.task.params = {"w": 8, "T": 2, "rule": 90, "distractors": 2}
    cfg.arms.append(_untied_arm(cfg))
    _, models, _, _ = run_point(cfg, cfg.task.params, seed=0)

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
        _, models, _, _ = run_point(cfg, params, seed=0)
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
    _, models, _, _ = run_point(cfg, params, seed=0)
    assert models["trm_nods"].n_steps == 4  # the arm's static value, not T


def test_train_accuracy_reported():
    """M3a diagnostic: train accuracy is captured per arm alongside test accuracy."""
    cfg = _iter_cfg(grid=dict(params={"T": [3]}), couple_n_steps_to_param="T")
    out, _, _, _ = run_point(cfg, {**cfg.task.params, "T": 3}, seed=0)
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
    out, models, _, _ = run_point(cfg, cfg.task.params, seed=0)
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
    a, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    b, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for lbl in a:
        assert a[lbl]["accuracy"] == b[lbl]["accuracy"]


def test_extrapolation_harness_determinism():
    """Verify that run_extrapolation_point at T_test=T_train produces identical
    metrics as the main run_point loop, confirming seed alignment."""
    cfg = _cfg()
    cfg.task.name = "iterated"
    cfg.task.params = {"w": 8, "T": 2, "rule": 90, "distractors": 2}

    out_main, models, baseline_main, _ = run_point(cfg, cfg.task.params, seed=0)

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


# --- speed knobs: `amp` / `compile` (§11.1) ------------------------------------------------
# Both change NUMERICS when on, so the contract is: (1) OFF is bit-identical to the pre-knob
# runner, and (2) a configuration that cannot honour the knob FAILS LOUDLY rather than silently
# training some arms differently — a per-arm precision difference would land in the reported Δ.


def test_amp_off_is_bit_identical():
    """amp=False must reproduce the pre-AMP path exactly (autocast/GradScaler disabled are
    documented no-ops). Guards the wrapper itself, on whatever device the suite runs on."""
    # Same train settings as the default _cfg; the ONLY difference is amp being stated explicitly.
    ref_cfg = _cfg()
    ref, _, _, _ = run_point(ref_cfg, ref_cfg.task.params, seed=0)
    cfg = _cfg(train=dict(epochs=3, lr=1e-3, batch_size=128, device="cpu", amp=False))
    got, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    assert set(ref) == set(got)
    for label in ref:
        assert ref[label]["accuracy"] == got[label]["accuracy"]
        assert ref[label]["n_params"] == got[label]["n_params"]


def test_amp_defaults_off():
    cfg = _cfg()
    assert cfg.train.amp is False and cfg.train.compile is False


def test_amp_accepted_on_deep_supervision_family():
    """AMP now covers the DS family (n_sup>1 / use_act / contraction) as well as the standard
    path — each routine autocasts its own forward/loss. On CPU the flag is inert, so these must
    RUN (previously they raised) and produce the same numbers as the un-flagged config."""
    for arm_over in [
        dict(n_sup=2),
        dict(use_act=True),
        dict(jac_reg_weight=0.1),
    ]:
        arm = dict(name="trm", label="a", hidden_dim=16, latent_dim=16, n_steps=3, **arm_over)
        base = _cfg(
            arms=[arm],
            train=dict(epochs=1, lr=1e-3, weight_decay=1e-4, batch_size=64),
        )
        ref, _, _, _ = run_point(base, base.task.params, seed=0)
        cfg = _cfg(
            arms=[arm],
            train=dict(epochs=1, lr=1e-3, weight_decay=1e-4, batch_size=64, amp=True),
        )
        got, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
        assert ref["a"]["accuracy"] == got["a"]["accuracy"]


def test_amp_still_rejects_curriculum_routines():
    """The trajectory curriculum resamples the unroll depth per batch, so it keeps raising —
    the guard must narrow to that case, not disappear."""
    cfg = _iter_cfg(
        curriculum=dict(param="T", T_min=1, T_max=4),
        couple_n_steps_to_param="T",
        train=dict(epochs=1, lr=1e-3, weight_decay=1e-4, batch_size=64, amp=True),
    )
    cfg.task.params = {"w": 8, "T": 4, "rule": 30, "distractors": 2}
    with pytest.raises(ValueError, match="curriculum"):
        run_point(cfg, cfg.task.params, seed=0)


def test_compile_failure_is_actionable():
    """Where torch.compile can't run (old torch / no Triton), the error must name the fix rather
    than surfacing a bare dynamo RuntimeError. Skipped where compile actually works."""
    from looptab.run import _compile_model

    try:
        torch.compile(torch.nn.Linear(2, 2))
    except Exception:
        pass
    else:
        pytest.skip("torch.compile is available here; nothing to assert about the failure path")
    with pytest.raises(RuntimeError, match="train.compile=true"):
        _compile_model(torch.nn.Linear(2, 2))


def test_cuda_graph_off_is_bit_identical():
    """cuda_graph=False must reproduce the pre-cuda_graph path exactly (it's inert on CPU too,
    but this guards the wrapper itself takes the untouched branch when the flag is off)."""
    ref_cfg = _cfg()
    ref, _, _, _ = run_point(ref_cfg, ref_cfg.task.params, seed=0)
    cfg = _cfg(train=dict(epochs=3, lr=1e-3, batch_size=128, device="cpu", cuda_graph=False))
    got, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    assert set(ref) == set(got)
    for label in ref:
        assert ref[label]["accuracy"] == got[label]["accuracy"]
        assert ref[label]["n_params"] == got[label]["n_params"]


def test_cuda_graph_defaults_off():
    cfg = _cfg()
    assert cfg.train.amp is False and cfg.train.compile is False and cfg.train.cuda_graph is False


def test_cuda_graph_accepted_on_deep_supervision_family():
    """cuda_graph now covers the DS family: n_sup / ACT run a fixed number of identically-shaped
    passes per batch, so the PASS is capturable. Inert on CPU, so these must run and match the
    un-flagged numbers."""
    for arm_over in [dict(n_sup=2), dict(use_act=True)]:
        arm = dict(name="trm", label="a", hidden_dim=16, latent_dim=16, n_steps=3, **arm_over)
        base = _cfg(arms=[arm], train=dict(epochs=1, lr=1e-3, weight_decay=1e-4, batch_size=64))
        ref, _, _, _ = run_point(base, base.task.params, seed=0)
        cfg = _cfg(
            arms=[arm],
            train=dict(epochs=1, lr=1e-3, weight_decay=1e-4, batch_size=64, cuda_graph=True),
        )
        got, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
        assert ref["a"]["accuracy"] == got["a"]["accuracy"]


def test_cuda_graph_still_rejects_unshaped_routines():
    """What stays rejected is genuinely shape-variable: the contraction probe (a fresh
    torch.func.jvp graph per batch) and the trajectory curriculum (per-batch depth)."""
    arm = dict(name="trm", label="a", hidden_dim=16, latent_dim=16, n_steps=3, jac_reg_weight=0.1)
    cfg = _cfg(
        arms=[arm],
        train=dict(epochs=1, lr=1e-3, weight_decay=1e-4, batch_size=64, cuda_graph=True),
    )
    with pytest.raises(ValueError, match="contraction-reg"):
        run_point(cfg, cfg.task.params, seed=0)


def test_cuda_graph_amp_combination_accepted():
    """The static-loss-scale capture path replaced the old mutual exclusion; on CPU both flags
    are inert, so the run must match the plain config rather than raising."""
    ref_cfg = _cfg(train=dict(epochs=2, lr=1e-3, batch_size=64))
    ref, _, _, _ = run_point(ref_cfg, ref_cfg.task.params, seed=0)
    cfg = _cfg(train=dict(epochs=2, lr=1e-3, batch_size=64, amp=True, cuda_graph=True))
    got, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for label in ref:
        assert ref[label]["accuracy"] == got[label]["accuracy"]


def test_speed_mode_is_recorded_in_results():
    """Every arm's metrics must say which speed path produced them — under `speed: auto` arms can
    land on different modes, and a Δ that hides a per-arm precision difference is exactly what
    §11.3's uniformity contract exists to prevent."""
    cfg = _cfg(train=dict(epochs=1, lr=1e-3, batch_size=64))
    out, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for label in out:
        assert out[label]["speed_mode"] == "eager"


def test_speed_mode_is_recorded_for_regression_results():
    """The regression branch must not continue before attaching speed provenance."""
    cfg = ExperimentConfig(
        task=dict(
            name="etth1",
            objective="regression",
            params={"lookback": 12, "horizon": 3, "n_folds": 10, "test_frac": 0.3},
            n_train=64,
            n_test=32,
            task_seed=0,
        ),
        arms=[dict(name="ff_matched", label="ff", hidden_dim=8, latent_dim=8, n_steps=2)],
        train=dict(epochs=1, lr=1e-3, batch_size=32, device="cpu"),
        seeds=[0],
    )
    out, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    assert out["ff"]["speed_mode"] == "eager"


def test_speed_auto_is_inert_on_cpu():
    """`speed: auto` on CPU has nothing to choose between (both knobs are inert there), so it must
    resolve to eager and reproduce the manual run exactly rather than burning probe time."""
    ref_cfg = _cfg(train=dict(epochs=2, lr=1e-3, batch_size=64))
    ref, _, _, _ = run_point(ref_cfg, ref_cfg.task.params, seed=0)
    cfg = _cfg(train=dict(epochs=2, lr=1e-3, batch_size=64, speed="auto"))
    got, _, _, _ = run_point(cfg, cfg.task.params, seed=0)
    for label in ref:
        assert ref[label]["accuracy"] == got[label]["accuracy"]
        assert "cpu" in got[label]["speed_mode"]


def test_speed_rejects_unknown_mode():
    with pytest.raises(Exception, match="speed"):
        _cfg(train=dict(epochs=1, lr=1e-3, batch_size=64, speed="turbo"))


def _probe_loader():
    """A small real loader for the autotuner tests (same shape as the `_cfg` parity task)."""
    from looptab.data.dataset import make_loaders, make_splits

    train_ds, test_ds = make_splits(
        task="parity", task_cfg={"d": 12, "k": 2}, task_seed=42,
        train_sample_seed=1, test_sample_seed=2, n_train=400, n_test=200,
    )
    train_loader, _ = make_loaders(train_ds, test_ds, batch_size=64)
    return train_loader


def _probe_model():
    from looptab.models.trm import TRM

    return TRM(in_features=12, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=3)


def test_autotune_probe_returns_a_mode_and_timings():
    """Exercise the autotuner directly. On CPU the candidates are all really the same eager path,
    so this asserts the CONTRACT (a valid winner, one finite timing per candidate, throwaway
    models only) rather than which mode wins — the winner is a GPU question."""
    from looptab.run import _autotune_speed_mode, _speed_mode_label

    built = []

    def _build():
        m = _probe_model()
        built.append(m)
        return m

    candidates = [(False, False), (True, False)]
    best, timings = _autotune_speed_mode(
        _build, _probe_loader(), candidates=candidates, device="cpu", lr=1e-3,
        weight_decay=1e-4, deep_supervision_weight=1.0, loss_type="ce", probe_steps=4,
    )
    assert best in candidates
    assert set(timings) == {_speed_mode_label(a, g) for a, g in candidates}
    assert all(t < float("inf") for t in timings.values())
    # One throwaway model per candidate — probing must never train the arm's real weights.
    assert len(built) == len(candidates)
    # CPU candidates are the same eager path, so restored RNG + identical batches must leave
    # their trained weights exactly equal; only the requested speed mode may vary.
    for a, b in zip(built[0].parameters(), built[1].parameters()):
        assert torch.equal(a, b)


def test_autotune_probe_leaves_rng_recoverable():
    """The probe trains throwaway models, which CONSUMES the global RNG (InMemoryLoader draws its
    per-epoch shuffle from it). run_point therefore re-seeds and rebuilds the arm after probing.
    This asserts the property that makes that safe: re-seeding restores the exact stream, so a
    post-probe seeded train is identical to one that never probed. Without the re-seed, an
    autotuned run would silently train on a different shuffle order than a manual one."""
    from looptab.run import _autotune_speed_mode
    from looptab.train.loop import train

    loader = _probe_loader()

    def _weights_after_seeded_train():
        torch.manual_seed(7)
        m = _probe_model()
        train(m, loader, epochs=2, lr=1e-3, device="cpu")
        return [p.detach().clone() for p in m.parameters()]

    clean = _weights_after_seeded_train()
    _autotune_speed_mode(
        _probe_model, loader, candidates=[(False, False)], device="cpu", lr=1e-3,
        weight_decay=1e-4, deep_supervision_weight=1.0, loss_type="ce", probe_steps=4,
    )
    after_probe = _weights_after_seeded_train()
    for a, b in zip(clean, after_probe):
        assert torch.equal(a, b)


def test_cuda_graph_tail_defaults_off_and_is_inert_on_cpu():
    """cuda_graph_tail changes WHICH rows are trained on (it stops dropping the ragged batch), so
    it must default off — committed cuda_graph results dropped that batch."""
    cfg = _cfg()
    assert cfg.train.cuda_graph_tail is False
    ref_cfg = _cfg(train=dict(epochs=2, lr=1e-3, batch_size=64))
    ref, _, _, _ = run_point(ref_cfg, ref_cfg.task.params, seed=0)
    got_cfg = _cfg(
        train=dict(epochs=2, lr=1e-3, batch_size=64, cuda_graph=True, cuda_graph_tail=True)
    )
    got, _, _, _ = run_point(got_cfg, got_cfg.task.params, seed=0)
    for label in ref:
        assert ref[label]["accuracy"] == got[label]["accuracy"]


def test_trm_mixer_fused_rejects_amp():
    """The fused kernel's extension is fp32-only with no autocast registration, so under amp it
    would train at a different precision than the other arms' autocast nn.Linear ops — a per-arm
    precision difference smuggled into the reported Δ (PR #35 review). Must raise, not silently
    compare kernel-vs-eager confounded with fp32-vs-fp16. Needs a multi-output task (TRMMixer's
    out_features = n_cells comes from the task's y shape, not an arm field) at a kernel-supported
    (n_cells, token_hidden) shape — `iterated` with w=6, no distractors gives n_cells=6.
    """
    arm = dict(
        name="trm_mixer_fused", label="a", hidden_dim=8, latent_dim=4, n_steps=2, token_hidden=6,
    )
    cfg = _cfg(
        task=dict(
            name="iterated",
            params={"w": 6, "T": 2, "rule": 30, "distractors": 0},
            n_train=100,
            n_test=50,
            task_seed=42,
            train_sample_seed=1,
            test_sample_seed=2,
        ),
        arms=[arm],
        train=dict(epochs=1, lr=1e-3, weight_decay=1e-4, batch_size=32, amp=True),
    )
    with pytest.raises(ValueError, match="not supported with trm_mixer_fused"):
        run_point(cfg, cfg.task.params, seed=0)
