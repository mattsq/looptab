"""Canonical synthetic task generators. Treat as spec — match semantics exactly."""

import numpy as np


def make_linear(n: int, d: int, task_seed: int, sample_seed: int):
    """Task 0: smoke test. Linear threshold in d dimensions."""
    w = np.random.default_rng(task_seed).standard_normal(d)
    X = np.random.default_rng(sample_seed).standard_normal((n, d))
    y = (X @ w > 0).astype(np.int64)
    return X.astype(np.float32), y


def make_parity(
    n: int,
    d: int,
    k: int,
    task_seed: int,
    sample_seed: int,
    noise: float = 0.0,
    symmetric: bool = False,
):
    """Task A: k-sparse parity in d bits with (d-k) distractors."""
    fn_rng = np.random.default_rng(task_seed)
    row_rng = np.random.default_rng(sample_seed)
    informative = fn_rng.choice(d, size=k, replace=False)
    X = row_rng.integers(0, 2, size=(n, d))
    y = X[:, informative].sum(axis=1) % 2
    if noise > 0:
        flip = row_rng.random(n) < noise
        y = np.where(flip, 1 - y, y)
    if symmetric:
        X = 2 * X - 1
    return X.astype(np.float32), y.astype(np.int64), informative


def ca_step(s: np.ndarray, rule: int) -> np.ndarray:
    """Single elementary CA step with periodic boundary."""
    left, center, right = np.roll(s, 1, -1), s, np.roll(s, -1, -1)
    idx = (left << 2) | (center << 1) | right
    return (rule >> idx) & 1


def make_iterated(
    n: int,
    w: int,
    T: int,
    task_seed: int,
    sample_seed: int,
    rule: int = 90,
    distractors: int = 0,
    return_trajectory: bool = False,
):
    """Task B: predict T steps of elementary CA rule from initial state.

    Default return is ``(X, s_T)`` — the input (s0 + optional distractors) and the state
    after T steps. With ``return_trajectory=True`` also return the full intermediate
    trajectory ``[s1, …, s_T]`` of shape ``(n, T, w)`` as a third element, for step-aligned
    deep supervision (M3b: loop step i ↔ CA state s_i). The trajectory's last frame is, by
    construction, identical to the canonical single-target ``s_T`` (asserted in tests).
    """
    row_rng = np.random.default_rng(sample_seed)
    s0 = row_rng.integers(0, 2, size=(n, w))
    s = s0.copy()
    traj = [] if return_trajectory else None
    for _ in range(T):
        s = ca_step(s, rule)
        if return_trajectory:
            traj.append(s.copy())
    X = s0
    if distractors > 0:
        fn_rng = np.random.default_rng(task_seed)
        noise = fn_rng.integers(0, 2, size=(n, distractors))  # static, uninformative
        X = np.concatenate([s0, noise], axis=-1)
    if return_trajectory:
        # Stack to (n, T, w); the last frame equals the canonical s_T target.
        trajectory = np.stack(traj, axis=1).astype(np.int64) if T > 0 else s0[:, :0, :]
        return X.astype(np.float32), s.astype(np.int64), trajectory
    return X.astype(np.float32), s.astype(np.int64)
