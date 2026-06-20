"""Determinism tests for every task generator. Same seeds => identical bytes."""

import numpy as np
import pytest
from looptab.data.generators import make_linear, make_parity, make_iterated, ca_step


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
