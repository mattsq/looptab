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


def _ring_band_mask(w: int, bandwidth: int) -> np.ndarray:
    """Boolean (w, w) mask: True where the ring distance min(|i-j|, w-|i-j|) ≤ bandwidth.

    The M14 locality knob. On a ring of ``w`` cells, ``bandwidth`` controls how far a cell's
    coupling reaches: ``1`` = nearest-neighbour only (spatially LOCAL, like a CA's 3-neighbour
    stencil — but with per-position irregular weights, so NON-CA), ``w//2`` = fully dense (no
    masking, the M13 regime). The mask is symmetric (ring distance is symmetric) so it preserves
    W's symmetry, and keeps the diagonal (distance 0), which is zeroed separately.
    """
    idx = np.arange(w)
    dist = np.abs(idx[:, None] - idx[None, :])
    ring = np.minimum(dist, w - dist)
    return ring <= bandwidth


def _build_hopfield_weights(
    w: int,
    task_seed: int,
    weights: str,
    n_patterns: int,
    weight_scale: int,
    density: float,
    bandwidth: int | None = None,
) -> np.ndarray:
    """Integer symmetric zero-diagonal weight matrix W (the 'function', fixed by task_seed).

    Two families (both all-integer ⇒ the generator is bit-exact, no float matmul determinism
    risk — contrast the M11 ``trm_decoupled`` caveat):
      - ``"hebbian"``: classic Hopfield W = Σ_μ ξ^μ (ξ^μ)^T over ``n_patterns`` random ±1
        patterns, diagonal zeroed. ``n_patterns`` is the ff-hardness dial — few patterns → few
        attractors a shallow MLP can map; many (≳0.14·w, the Hopfield capacity) → spurious
        attractors + complex basins → ff-hard.
      - ``"random"``: symmetric integer matrix, entries in {-weight_scale..weight_scale} at the
        given off-diagonal ``density``. ``weight_scale``/``density`` are the hardness dials.

    ``bandwidth`` (M14 locality probe): if not None, zero every coupling with ring distance
    > ``bandwidth`` (see ``_ring_band_mask``), turning the dense net into a *local-but-non-CA*
    threshold net. ``None`` = dense = the M13 regime. The mask is applied to the already-built
    integer W, preserving symmetry and the all-integer (bit-exact) property; the PSD-guaranteeing
    ``gamma`` in ``make_hopfield`` is derived from the *masked* W, so convergence still holds.
    """
    fn_rng = np.random.default_rng(task_seed)
    if weights == "hebbian":
        patterns = fn_rng.integers(0, 2, size=(n_patterns, w)).astype(np.int64) * 2 - 1
        W = patterns.T @ patterns  # (w, w) integer, symmetric, PSD
    elif weights == "random":
        M = fn_rng.integers(-weight_scale, weight_scale + 1, size=(w, w)).astype(np.int64)
        if density < 1.0:
            M = M * (fn_rng.random((w, w)) < density)
        U = np.triu(M, 1)
        W = U + U.T
    else:
        raise ValueError(f"make_hopfield: unknown weights mode {weights!r} (use hebbian|random)")
    if bandwidth is not None:
        W = W * _ring_band_mask(w, bandwidth)
    np.fill_diagonal(W, 0)
    return W.astype(np.int64)


def _threshold_step(s: np.ndarray, W: np.ndarray, gamma: int) -> np.ndarray:
    """Synchronous threshold update with self-coupling; s in {-1,+1}. Tie (field==0) → keep.

    ``field = s·W + γ·s`` (W symmetric ⇒ s·W == W·s per cell). The self-coupling γ·s damps
    parallel 2-cycles: with γ ≥ -λ_min(W), W+γI is PSD and the parallel energy is non-increasing
    ⇒ convergence to a fixed point. The 'keep current' tie-break is the convergence-safe choice.
    """
    field = s @ W + gamma * s  # integer (n, w)
    return np.where(field > 0, 1, np.where(field < 0, -1, s))


def make_hopfield(
    n: int,
    w: int,
    task_seed: int,
    sample_seed: int,
    weights: str = "hebbian",
    n_patterns: int = 8,
    weight_scale: int = 1,
    density: float = 1.0,
    bandwidth: int | None = None,
    gamma: int | None = None,
    gamma_margin: int = 1,
    distractors: int = 0,
    T: int | None = None,
    return_trajectory: bool = False,
    max_steps: int | None = None,
):
    """Non-ECA hard-convergence FIXED-POINT task (M13): map s0 → a threshold-net attractor.

    The M13 substrate for testing whether the joint-state coherence result (M8–M12) is a property
    of the hard-convergence *regime* or of cellular automata specifically. Unlike ``make_converge``
    (a *local* 3-neighbour CA), here the update is a **dense, fully-coupled** binary threshold /
    Hopfield network — maximally unlike a local CA, and basin-of-attraction is *intrinsically* a
    whole-row property, the strongest possible probe of the joint-state hypothesis. The contract
    (signature shape, ``(X, s_inf[, traj])`` return, loud non-convergence guard, and
    ``return_trajectory``) mirrors ``make_converge`` so it slots into the existing
    dataset/trajectory machinery unchanged.

    Function (fixed by ``task_seed``): integer symmetric zero-diagonal ``W`` (see
    ``_build_hopfield_weights``) + integer self-coupling ``gamma``. Rows (fixed by ``sample_seed``):
    s0 ∈ {-1,+1}^(n,w), iterated synchronously to the global fixed point. Outputs are mapped to
    {0,1} to match the binary readout heads and the ``coherence_excess`` metric.

    ``bandwidth`` (M14 locality probe): zero couplings beyond ring distance ``bandwidth`` to make
    the net *local-but-non-CA* (``1`` = nearest-neighbour, ``None``/``w//2`` = dense = M13) — the
    knob that tests whether the M8–M12 joint-state coherence edge needs *locality* or the full CA.

    ``gamma``: pass an explicit int for committed runs (keeps the generator purely integer ⇒
    bit-exact). ``None`` auto-derives ``ceil(-λ_min(W)) + gamma_margin`` (guarantees synchronous
    convergence by making W+γI PSD) — this path uses a float eigen-solve, so it is for *screening*;
    the loud guard + a multi-seed screen (M12 lesson) verify the pinned int gamma converges.
    """
    W = _build_hopfield_weights(
        w, task_seed, weights, n_patterns, weight_scale, density, bandwidth
    )
    if gamma is None:
        lam_min = float(np.linalg.eigvalsh(W.astype(np.float64)).min())
        gamma = max(int(np.ceil(-lam_min - 1e-9)) + gamma_margin, 0)

    row_rng = np.random.default_rng(sample_seed)
    s0 = row_rng.integers(0, 2, size=(n, w)).astype(np.int64) * 2 - 1  # {-1,+1}
    if max_steps is None:
        max_steps = 8 * w  # generous cap; PSD self-coupling converges, depth is the screened axis

    s = s0.copy()
    for _ in range(max_steps):
        nxt = _threshold_step(s, W, gamma)
        if np.array_equal(nxt, s):  # every row stationary => global fixed point
            break
        s = nxt
    s_inf = s
    if not np.array_equal(_threshold_step(s_inf, W, gamma), s_inf):
        raise ValueError(
            f"make_hopfield: did not reach a fixed point within {max_steps} steps (w={w}, "
            f"weights={weights}, gamma={gamma}); raise gamma/gamma_margin or max_steps, or "
            f"reduce n_patterns/weight_scale."
        )

    def _to01(arr):
        return (arr + 1) // 2  # {-1,+1} -> {0,1}

    X = _to01(s0)
    if distractors > 0:
        # static, uninformative distractor bits drawn from the function RNG (shared train/test)
        fn_rng = np.random.default_rng(task_seed)
        noise = fn_rng.integers(0, 2, size=(n, distractors))
        X = np.concatenate([X, noise], axis=-1)

    if return_trajectory:
        traj_len = T if T is not None else max_steps
        cur = s0.copy()
        frames = []
        for _ in range(traj_len):
            cur = _threshold_step(cur, W, gamma)
            frames.append(_to01(cur).copy())
        traj = np.stack(frames, axis=1).astype(np.int64) if traj_len > 0 else _to01(s0)[:, :0, :]
        return X.astype(np.float32), _to01(s_inf).astype(np.int64), traj
    return X.astype(np.float32), _to01(s_inf).astype(np.int64)
