"""Determinism tests for every task generator. Same seeds => identical bytes."""

import numpy as np
import pytest

from looptab.data.generators import (
    _build_hopfield_weights,
    _ring_band_mask,
    _threshold_step,
    ca_step,
    make_converge,
    make_hopfield,
    make_iterated,
    make_linear,
    make_mixed_converge,
    make_multi_parity,
    make_parity,
    mixed_ca_step,
)


def test_linear_determinism():
    a = make_linear(n=100, d=10, task_seed=7, sample_seed=3)
    b = make_linear(n=100, d=10, task_seed=7, sample_seed=3)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_linear_sample_seed_varies():
    a = make_linear(n=100, d=10, task_seed=7, sample_seed=3)
    b = make_linear(n=100, d=10, task_seed=7, sample_seed=99)
    assert not np.array_equal(a[0], b[0])


def test_linear_task_seed_consistent():
    """Different sample seeds but same task_seed => same decision boundary."""
    X1, y1 = make_linear(n=500, d=5, task_seed=1, sample_seed=10)
    X2, y2 = make_linear(n=500, d=5, task_seed=1, sample_seed=20)
    # Both datasets use same w, so manually recompute and check labels
    w = np.random.default_rng(1).standard_normal(5)
    np.testing.assert_array_equal(y1, (X1 @ w > 0).astype(np.int64))
    np.testing.assert_array_equal(y2, (X2 @ w > 0).astype(np.int64))


def test_parity_determinism():
    a = make_parity(n=200, d=16, k=3, task_seed=5, sample_seed=9)
    b = make_parity(n=200, d=16, k=3, task_seed=5, sample_seed=9)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_parity_informative_shared_across_splits():
    """Train and test share task_seed => same informative bits."""
    _, _, inf_train = make_parity(n=100, d=20, k=4, task_seed=42, sample_seed=1)
    _, _, inf_test = make_parity(n=100, d=20, k=4, task_seed=42, sample_seed=2)
    np.testing.assert_array_equal(inf_train, inf_test)


def test_parity_labels_correct():
    X, y, informative = make_parity(n=300, d=10, k=2, task_seed=0, sample_seed=0)
    expected = X[:, informative].sum(axis=1).astype(int) % 2
    np.testing.assert_array_equal(y, expected)


def test_parity_symmetric():
    X, y, _ = make_parity(n=50, d=8, k=2, task_seed=0, sample_seed=0, symmetric=True)
    assert X.min() == -1
    assert X.max() == 1


def test_parity_noise():
    # With noise=0.5 all bits are random; just check shape and dtype
    X, y, _ = make_parity(n=100, d=8, k=2, task_seed=0, sample_seed=0, noise=0.5)
    assert y.dtype == np.int64
    assert X.shape == (100, 8)


def test_multi_parity_determinism():
    a = make_multi_parity(n=200, d=20, k=4, w=4, task_seed=5, sample_seed=9)
    b = make_multi_parity(n=200, d=20, k=4, w=4, task_seed=5, sample_seed=9)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_multi_parity_shapes():
    X, y, informative = make_multi_parity(n=50, d=20, k=3, w=8, task_seed=0, sample_seed=0)
    assert X.shape == (50, 20)
    assert y.shape == (50, 8)
    assert informative.shape == (8, 3)
    assert y.dtype == np.int64


def test_multi_parity_informative_shared_across_splits():
    """Train and test share task_seed => same w informative subsets (§3)."""
    _, _, inf_train = make_multi_parity(n=100, d=20, k=4, w=4, task_seed=42, sample_seed=1)
    _, _, inf_test = make_multi_parity(n=100, d=20, k=4, w=4, task_seed=42, sample_seed=2)
    np.testing.assert_array_equal(inf_train, inf_test)


def test_multi_parity_labels_correct():
    X, y, informative = make_multi_parity(n=300, d=16, k=3, w=5, task_seed=1, sample_seed=2)
    for j in range(informative.shape[0]):
        expected = X[:, informative[j]].sum(axis=1).astype(int) % 2
        np.testing.assert_array_equal(y[:, j], expected)


def test_multi_parity_w1_reduces_to_parity():
    """w=1 reduces exactly to make_parity, modulo the trailing output axis."""
    Xm, ym, infm = make_multi_parity(n=120, d=20, k=4, w=1, task_seed=7, sample_seed=3)
    Xp, yp, infp = make_parity(n=120, d=20, k=4, task_seed=7, sample_seed=3)
    np.testing.assert_array_equal(Xm, Xp)
    np.testing.assert_array_equal(ym[:, 0], yp)
    np.testing.assert_array_equal(infm[0], infp)


def test_multi_parity_symmetric():
    X, _, _ = make_multi_parity(n=50, d=12, k=2, w=3, task_seed=0, sample_seed=0, symmetric=True)
    assert X.min() == -1
    assert X.max() == 1


def test_multi_parity_determinism_noise_and_symmetric_paths():
    """§5.8: the optional noise/symmetric RNG paths are also a pure function of the seeds."""
    kw = dict(n=150, d=14, k=3, w=4, task_seed=11, sample_seed=23, noise=0.2, symmetric=True)
    a = make_multi_parity(**kw)
    b = make_multi_parity(**kw)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_converge_determinism():
    a = make_converge(n=200, w=16, task_seed=5, sample_seed=9, rule=92, distractors=4)
    b = make_converge(n=200, w=16, task_seed=5, sample_seed=9, rule=92, distractors=4)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("rule", [13, 78, 92, 140, 232, 69, 79, 93, 141, 197])
def test_converge_target_is_a_fixed_point(rule):
    """The target s_inf must satisfy ca_step(s_inf)==s_inf for every converge rule used.

    13/78/92 are the M8/M8b families; 140/232 are the M11 additions (140 = deep-spread,
    232 = majority/balanced). 69/79/93/141/197 are the M12 orbit-mates: the complete set of
    balanced+deep converging ECAs is two symmetry orbits — {13,69,79,93} and {78,92,141,197} —
    all screened for the fixed-point property at w∈{24,32} before being gridded.
    """
    X, s_inf = make_converge(n=300, w=32, task_seed=1, sample_seed=2, rule=rule)
    np.testing.assert_array_equal(ca_step(s_inf, rule), s_inf)


def test_converge_raises_on_non_converging_rule():
    """A non-converging rule must fail loudly, not return a non-fixed state as the target."""
    # rule 184 (traffic) reaches a period-2 shift on a ring, never a fixed point => must raise.
    with pytest.raises(ValueError):
        make_converge(n=64, w=16, task_seed=0, sample_seed=0, rule=184, max_steps=40)


def test_converge_shapes_and_distractors():
    X, y = make_converge(n=50, w=32, task_seed=0, sample_seed=0, rule=92, distractors=8)
    assert X.shape == (50, 40)  # w + distractors
    assert y.shape == (50, 32)  # fixed point is over the w CA cells only
    assert y.dtype == np.int64


def test_converge_trajectory_chains_to_fixed_point_tail():
    """Trajectory frames are successive CA steps; once at the fixed point they stay there."""
    X, s_inf, traj = make_converge(
        n=40, w=16, task_seed=3, sample_seed=4, rule=92, T=20, return_trajectory=True
    )
    assert traj.shape == (40, 20, 16)
    for i in range(1, traj.shape[1]):
        np.testing.assert_array_equal(traj[:, i, :], ca_step(traj[:, i - 1, :], 92))
    # T=20 exceeds the convergence cap for w=16, so the last frame has reached s_inf.
    np.testing.assert_array_equal(traj[:, -1, :], s_inf)


def _hopfield_update01(s01, W, gamma):
    """Reference threshold update in {0,1} coding, for the fixed-point assertion."""
    s = s01.astype(np.int64) * 2 - 1
    return (_threshold_step(s, W, gamma) + 1) // 2


@pytest.mark.parametrize("weights,kw", [("hebbian", {"n_patterns": 8}), ("random", {})])
def test_hopfield_determinism(weights, kw):
    a = make_hopfield(
        n=200, w=24, task_seed=5, sample_seed=9, weights=weights, gamma=12, distractors=4, **kw
    )
    b = make_hopfield(
        n=200, w=24, task_seed=5, sample_seed=9, weights=weights, gamma=12, distractors=4, **kw
    )
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("w", [24, 32])
def test_hopfield_target_is_a_fixed_point(w):
    """s_inf must satisfy threshold_update(s_inf)==s_inf for every row (the defining property)."""
    W = _build_hopfield_weights(
        w, task_seed=1, weights="hebbian", n_patterns=8, weight_scale=1, density=1.0
    )
    gamma = 12
    X, s_inf = make_hopfield(n=300, w=w, task_seed=1, sample_seed=2, n_patterns=8, gamma=gamma)
    np.testing.assert_array_equal(_hopfield_update01(s_inf, W, gamma), s_inf)


def test_hopfield_auto_gamma_converges_and_is_a_fixed_point():
    """gamma=None auto-derives a PSD-guaranteeing self-coupling; the result is a fixed point."""
    W = _build_hopfield_weights(
        20, task_seed=7, weights="hebbian", n_patterns=6, weight_scale=1, density=1.0
    )
    lam_min = float(np.linalg.eigvalsh(W.astype(np.float64)).min())
    gamma = max(int(np.ceil(-lam_min - 1e-9)) + 1, 0)
    X, s_inf = make_hopfield(n=200, w=20, task_seed=7, sample_seed=3, n_patterns=6, gamma=None)
    np.testing.assert_array_equal(_hopfield_update01(s_inf, W, gamma), s_inf)


def test_hopfield_raises_on_nonconverging():
    """Too small a gamma can leave a 2-cycle; the generator must fail loudly, not return it."""
    with pytest.raises(ValueError):
        make_hopfield(
            n=64,
            w=24,
            task_seed=0,
            sample_seed=0,
            weights="random",
            weight_scale=4,
            gamma=0,
            max_steps=30,
        )


def test_hopfield_shapes_distractors_and_coding():
    X, y = make_hopfield(
        n=50, w=32, task_seed=0, sample_seed=0, n_patterns=10, gamma=15, distractors=8
    )
    assert X.shape == (50, 40)  # w + distractors
    assert y.shape == (50, 32)  # fixed point over the w net cells only
    assert y.dtype == np.int64
    assert set(np.unique(y)).issubset({0, 1})  # mapped to {0,1} for the binary heads


def test_hopfield_balance():
    """The ±1/zero-bias symmetry => each output cell is ~balanced across rows (majority near .5)."""
    X, y = make_hopfield(n=4000, w=32, task_seed=3, sample_seed=4, n_patterns=12, gamma=15)
    cell_means = y.mean(axis=0)
    assert np.all((cell_means > 0.25) & (cell_means < 0.75))


def test_hopfield_trajectory_chains_to_fixed_point_tail():
    W = _build_hopfield_weights(
        20, task_seed=3, weights="hebbian", n_patterns=6, weight_scale=1, density=1.0
    )
    gamma = 12
    X, s_inf, traj = make_hopfield(
        n=40,
        w=20,
        task_seed=3,
        sample_seed=4,
        n_patterns=6,
        gamma=gamma,
        T=40,
        return_trajectory=True,
    )
    assert traj.shape == (40, 40, 20)
    for i in range(1, traj.shape[1]):
        np.testing.assert_array_equal(
            traj[:, i, :], _hopfield_update01(traj[:, i - 1, :], W, gamma)
        )
    # T=40 exceeds the convergence depth, so the last frame has reached s_inf.
    np.testing.assert_array_equal(traj[:, -1, :], s_inf)


# --- M14: bandwidth / locality knob -------------------------------------------------


@pytest.mark.parametrize("bandwidth", [1, 2, 4])
def test_hopfield_bandwidth_determinism(bandwidth):
    """Banded weights keep the generator bit-exact: same seeds + bandwidth => identical bytes."""
    a = make_hopfield(
        n=200, w=24, task_seed=5, sample_seed=9, n_patterns=12, gamma=16, bandwidth=bandwidth
    )
    b = make_hopfield(
        n=200, w=24, task_seed=5, sample_seed=9, n_patterns=12, gamma=16, bandwidth=bandwidth
    )
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("bandwidth", [1, 3, 5])
def test_hopfield_bandwidth_zeros_distant_couplings(bandwidth):
    """W must be exactly zero beyond ring distance `bandwidth`, and symmetric, diagonal zeroed."""
    w = 24
    W = _build_hopfield_weights(
        w, task_seed=1, weights="hebbian", n_patterns=12, weight_scale=1, density=1.0,
        bandwidth=bandwidth,
    )
    mask = _ring_band_mask(w, bandwidth)
    assert np.all(W[~mask] == 0)  # nothing leaks past the band
    np.testing.assert_array_equal(W, W.T)  # still symmetric
    assert np.all(np.diag(W) == 0)


def test_hopfield_bandwidth_half_equals_dense():
    """bandwidth = w//2 spans the whole ring => identical to the dense (M13) net."""
    w = 24
    dense = _build_hopfield_weights(
        w, task_seed=2, weights="hebbian", n_patterns=12, weight_scale=1, density=1.0
    )
    full = _build_hopfield_weights(
        w, task_seed=2, weights="hebbian", n_patterns=12, weight_scale=1, density=1.0,
        bandwidth=w // 2,
    )
    np.testing.assert_array_equal(dense, full)


@pytest.mark.parametrize("bandwidth", [1, 2, 4])
def test_hopfield_banded_target_is_a_fixed_point(bandwidth):
    """The banded net still settles to a genuine fixed point (auto-gamma keeps W+γI PSD)."""
    w = 24
    W = _build_hopfield_weights(
        w, task_seed=3, weights="hebbian", n_patterns=12, weight_scale=1, density=1.0,
        bandwidth=bandwidth,
    )
    lam_min = float(np.linalg.eigvalsh(W.astype(np.float64)).min())
    gamma = max(int(np.ceil(-lam_min - 1e-9)) + 1, 0)
    X, s_inf = make_hopfield(
        n=300, w=w, task_seed=3, sample_seed=4, n_patterns=12, bandwidth=bandwidth, gamma=None
    )
    np.testing.assert_array_equal(_hopfield_update01(s_inf, W, gamma), s_inf)


# --- M15: per-position mixed-CA fixed-point task (deep + non-uniform + local) -------


def test_mixed_ca_step_matches_ca_step_when_uniform():
    """A constant per-position rule vector reduces mixed_ca_step to plain ca_step."""
    s = np.random.default_rng(0).integers(0, 2, size=(50, 16))
    rules = np.full(16, 78, dtype=np.int64)
    np.testing.assert_array_equal(mixed_ca_step(s, rules), ca_step(s, 78))


def test_mixed_converge_determinism():
    """Same seeds (incl. the rejection filter) => identical bytes."""
    a = make_mixed_converge(n=200, w=24, task_seed=42, sample_seed=1, distractors=8)
    b = make_mixed_converge(n=200, w=24, task_seed=42, sample_seed=1, distractors=8)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_mixed_converge_is_non_uniform():
    """The per-position rule assignment uses >1 distinct rule (it is NOT a single CA)."""
    rules = np.asarray((78, 92, 141, 197))
    fn_rng = np.random.default_rng(42)
    pos_rules = rules[fn_rng.integers(0, len(rules), size=48)]
    assert len(np.unique(pos_rules)) > 1


@pytest.mark.parametrize("w", [24, 32])
def test_mixed_converge_target_is_a_fixed_point(w):
    """Every kept row's target is a genuine fixed point of its own per-position rule."""
    rules = np.asarray((78, 92, 141, 197))
    pos_rules = rules[np.random.default_rng(7).integers(0, len(rules), size=w)]
    X, s_inf = make_mixed_converge(n=400, w=w, task_seed=7, sample_seed=2)
    np.testing.assert_array_equal(mixed_ca_step(s_inf, pos_rules), s_inf)


def test_mixed_converge_balance_and_nontrivial():
    """Balanced (majority ~0.5) and non-trivial (most rows actually move off s0)."""
    X, y = make_mixed_converge(n=3000, w=32, task_seed=42, sample_seed=1)
    cell_means = y.mean(axis=0)
    assert np.all((cell_means > 0.25) & (cell_means < 0.75))
    moved = ~(y == X[:, :32]).all(axis=1)
    assert moved.mean() > 0.9  # screened triv ~0% => >90% of rows are non-identity


def test_mixed_converge_shapes_distractors_and_coding():
    X, y = make_mixed_converge(n=50, w=24, task_seed=0, sample_seed=0, distractors=8)
    assert X.shape == (50, 32)  # w + distractors
    assert y.shape == (50, 24)
    assert set(np.unique(y)).issubset({0, 1})


def test_mixed_converge_trajectory_chains_to_fixed_point_tail():
    rules = np.asarray((78, 92, 141, 197))
    pos_rules = rules[np.random.default_rng(3).integers(0, len(rules), size=24)]
    X, s_inf, traj = make_mixed_converge(
        n=80, w=24, task_seed=3, sample_seed=4, T=96, return_trajectory=True
    )
    assert traj.shape == (80, 96, 24)
    for i in range(1, traj.shape[1]):
        np.testing.assert_array_equal(traj[:, i, :], mixed_ca_step(traj[:, i - 1, :], pos_rules))
    # T=96 = 4*w exceeds the (filtered) convergence depth, so the tail has reached s_inf.
    np.testing.assert_array_equal(traj[:, -1, :], s_inf)


def test_mixed_converge_raises_when_cycling():
    """A non-converging rule_set must fail loudly rather than return non-fixed states."""
    with pytest.raises(ValueError):
        # rule 90 (XOR) never settles to a fixed point from random inputs => 0 convergent rows.
        make_mixed_converge(
            n=200, w=24, task_seed=0, sample_seed=0, rule_set=(90,), max_draw_factor=4
        )


def test_mixed_converge_trajectory_path_matches_splits_path():
    """Curriculum-alignment invariant: the rejection filter must accept the SAME rows whether
    called for the (X, s_inf) target or for the trajectory — else the step-aligned DS target
    would not correspond to the s_inf the other arms are trained on (M15)."""
    X1, y1 = make_mixed_converge(n=400, w=24, task_seed=42, sample_seed=1, distractors=8)
    X2, y2, traj = make_mixed_converge(
        n=400, w=24, task_seed=42, sample_seed=1, distractors=8, T=6, return_trajectory=True
    )
    np.testing.assert_array_equal(X1, X2)  # identical accepted inputs (incl. distractors)
    np.testing.assert_array_equal(y1, y2)  # identical fixed-point targets


@pytest.mark.parametrize("cap", [2, 4, 6])
def test_mixed_converge_accept_max_depth_caps_depth(cap):
    """accept_max_depth keeps only rows converging within `cap` steps (M15b depth-matched control).

    Uniform rule_set=(78,) => a true CA through the same pipeline; every accepted row must reach its
    fixed point in <= cap steps."""
    rules = np.full(24, 78, dtype=np.int64)
    X, s_inf = make_mixed_converge(
        n=300, w=24, task_seed=42, sample_seed=1, rule_set=(78,), accept_max_depth=cap
    )
    s = X[:, :24].astype(np.int64)
    reached = np.zeros(len(s), dtype=bool)
    for _ in range(cap):
        s = mixed_ca_step(s, rules)
        reached |= (s == s_inf).all(axis=1)
    assert reached.all()  # every accepted row hits its fixed point within `cap` steps


def test_mixed_converge_accept_max_depth_none_is_unchanged():
    """The cap is additive: accept_max_depth=None must reproduce the uncapped output bit-for-bit.

    This checks additivity *within the current code*. The cross-version guard (that the M15b
    depth-tracking rewrite still reproduces the pre-M15b committed output) is the golden-hash test
    below."""
    a = make_mixed_converge(n=200, w=24, task_seed=42, sample_seed=1, distractors=8)
    b = make_mixed_converge(
        n=200, w=24, task_seed=42, sample_seed=1, distractors=8, accept_max_depth=None
    )
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_mixed_converge_golden_hash():
    """Pin the committed M15 output bytes so the M15b depth-tracking rewrite (and any future change)
    cannot silently alter the generated data the committed results rest on."""
    import hashlib

    X, y = make_mixed_converge(n=200, w=24, task_seed=42, sample_seed=1, distractors=8)
    assert hashlib.sha256(X.tobytes()).hexdigest()[:16] == "7b862a85c0038032"
    assert hashlib.sha256(y.tobytes()).hexdigest()[:16] == "40e789486f31084a"


def test_ca_step_rule90():
    """Rule 90 is XOR of neighbors."""
    s = np.array([[1, 0, 0, 1, 0]])
    out = ca_step(s, 90)
    left = np.roll(s, 1, -1)
    right = np.roll(s, -1, -1)
    np.testing.assert_array_equal(out, left ^ right)


def test_iterated_determinism():
    a = make_iterated(n=100, w=8, T=3, task_seed=1, sample_seed=2)
    b = make_iterated(n=100, w=8, T=3, task_seed=1, sample_seed=2)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_iterated_shapes():
    X, y = make_iterated(n=50, w=10, T=2, task_seed=0, sample_seed=0, distractors=5)
    assert X.shape == (50, 15)
    assert y.shape == (50, 10)


def test_iterated_T0_identity():
    """T=0 => output equals input (no steps applied)."""
    X, y = make_iterated(n=30, w=6, T=0, task_seed=0, sample_seed=0)
    np.testing.assert_array_equal(X, y)


def test_iterated_trajectory_shape_and_consistency():
    """M3b: the trajectory is (n, T, w) and its last frame is the canonical s_T target."""
    X, y, traj = make_iterated(
        n=40, w=7, T=5, task_seed=1, sample_seed=2, rule=30, return_trajectory=True
    )
    assert traj.shape == (40, 5, 7)
    assert X.shape == (40, 7)  # no distractors here, so X is just s0
    # Consistency check against the canonical single-target output (per the M3b spec).
    np.testing.assert_array_equal(traj[:, -1, :], y)


def test_iterated_trajectory_is_step_chain():
    """Each trajectory frame is one CA step from the previous (s_{i} = ca_step(s_{i-1}))."""
    _, _, traj = make_iterated(
        n=25, w=8, T=4, task_seed=3, sample_seed=4, rule=110, return_trajectory=True
    )
    for i in range(1, traj.shape[1]):
        np.testing.assert_array_equal(traj[:, i, :], ca_step(traj[:, i - 1, :], 110))


def test_iterated_trajectory_determinism():
    a = make_iterated(n=30, w=6, T=3, task_seed=7, sample_seed=8, return_trajectory=True)
    b = make_iterated(n=30, w=6, T=3, task_seed=7, sample_seed=8, return_trajectory=True)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_iterated_trajectory_first_frame_is_one_step():
    """First trajectory frame s_1 is one CA step from s0 (the input's CA part)."""
    X, _, traj = make_iterated(
        n=20, w=9, T=2, task_seed=0, sample_seed=0, rule=90, return_trajectory=True
    )
    s0 = X[:, :9].astype(np.int64)
    np.testing.assert_array_equal(traj[:, 0, :], ca_step(s0, 90))
