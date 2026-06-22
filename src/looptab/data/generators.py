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


def make_multi_parity(
    n: int,
    d: int,
    k: int,
    w: int,
    task_seed: int,
    sample_seed: int,
    noise: float = 0.0,
    symmetric: bool = False,
):
    """Both-axes probe (M6a): w independent k-sparse parities in d bits.

    Predict ``w`` binary outputs from the same ``d`` input bits. Output ``j`` is the parity
    of its own size-``k`` informative subset; the ``w`` subsets are drawn independently from
    ``task_seed`` (so they may overlap) and are shared train/test (§3). Stresses depth (each
    output is order-``k`` → shallow MLPs fail at k≥4) and width (``w`` parallel computations →
    narrow untied blocks bottleneck) at once.

    ``w == 1`` reduces exactly to ``make_parity`` (a (n, 1) target instead of (n,)).
    """
    fn_rng = np.random.default_rng(task_seed)
    row_rng = np.random.default_rng(sample_seed)
    informative = np.stack([fn_rng.choice(d, size=k, replace=False) for _ in range(w)])
    X = row_rng.integers(0, 2, size=(n, d))
    y = np.stack([X[:, informative[j]].sum(axis=1) % 2 for j in range(w)], axis=1)  # (n, w)
    if noise > 0:
        flip = row_rng.random((n, w)) < noise
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


def make_converge(
    n: int,
    w: int,
    task_seed: int,
    sample_seed: int,
    rule: int = 232,
    distractors: int = 0,
    T: int | None = None,
    return_trajectory: bool = False,
    max_steps: int | None = None,
):
    """Variable-compute FIXED-POINT task (M8): map s0 to the CA's converged fixed point s_inf.

    Unlike `make_iterated` (target = s_T, a *moving* target that changes with T — non-convergent
    for chaotic rules), here the target is the **fixed point** s_inf reached by iterating a
    *converging* rule (default 232 = majority/voting) until ``ca_step(s)==s``. Two properties make
    this the substrate for the adaptive-computation test:
      - **Per-instance varying depth:** the convergence time differs per row, so some instances
        need more refinement steps than others — a fixed-depth net cannot match a loop that
        unrolls more at test time.
      - **A genuine fixed point:** once s_inf is reached, further steps leave it unchanged, so
        over-unrolling R' past convergence should HOLD (contrast `make_iterated`/rule 30, where
        over-unrolling decays). This is the property Deep Thinking's progressive loss targets.

    Returns ``(X, s_inf)`` — input (s0 + optional static distractors) and the fixed point. With
    ``return_trajectory=True`` also returns ``[s1..s_T]`` of shape ``(n, T, w)`` for step-aligned
    DS (NOTE its last frame is s_T, which for slow-converging rows is *not* yet s_inf — that gap
    is the in-curriculum / out-of-curriculum split the experiment exploits). ``T`` sets only the
    trajectory length; the target always iterates to the true fixed point.
    """
    row_rng = np.random.default_rng(sample_seed)
    s0 = row_rng.integers(0, 2, size=(n, w))
    if max_steps is None:
        max_steps = 4 * w  # generous cap; majority (rule 232) converges in O(w)

    # Iterate to the global fixed point (all rows stationary) for the TARGET.
    s = s0.copy()
    for _ in range(max_steps):
        nxt = ca_step(s, rule)
        if np.array_equal(nxt, s):  # every row stationary => s is the per-row fixed point
            break
        s = nxt
    s_inf = s
    # Guard the defining property: a non-converging rule (e.g. a period-2 cycle) would silently
    # return a non-fixed state as the "fixed point". Fail loudly instead — `converge` requires a
    # rule whose target genuinely satisfies ca_step(s_inf)==s_inf for every row.
    if not np.array_equal(ca_step(s_inf, rule), s_inf):
        raise ValueError(
            f"make_converge: rule {rule} did not reach a fixed point within {max_steps} steps "
            f"(w={w}); it may be non-converging. Pick a converging rule or raise max_steps."
        )

    X = s0
    if distractors > 0:
        fn_rng = np.random.default_rng(task_seed)
        noise = fn_rng.integers(0, 2, size=(n, distractors))  # static, uninformative
        X = np.concatenate([s0, noise], axis=-1)

    if return_trajectory:
        traj_len = T if T is not None else max_steps
        cur = s0.copy()
        frames = []
        for _ in range(traj_len):
            cur = ca_step(cur, rule)
            frames.append(cur.copy())
        traj = np.stack(frames, axis=1).astype(np.int64) if traj_len > 0 else s0[:, :0, :]
        return X.astype(np.float32), s_inf.astype(np.int64), traj
    return X.astype(np.float32), s_inf.astype(np.int64)
