"""Determinism tests for every task generator. Same seeds => identical bytes."""

import numpy as np
import pytest

from looptab.data.generators import (
    ca_step,
    make_converge,
    make_iterated,
    make_linear,
    make_multi_parity,
    make_parity,
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
