# CLAUDE.md

This file is the single source of truth for any agent working in this repo, including
cloud agents that arrive with **no conversation context**. Read it in full before
making changes. When you land a capability, update §11.

---

## 1. What this is

A research repo investigating one question:

> Does **iterative latent refinement + deep supervision** (à la TRM / HRM) provide a
> useful inductive bias on **tabular** problems, *over a parameter- and depth-matched
> feedforward control*?

This is greenfield. There is no established "HRM/TRM-on-tabular" result. We therefore
start on a **synthetic canonical suite** (§3) where difficulty is a dial and the ground
truth is a known algorithm, *before* touching real tabular data. The grid-reasoning
origins of these models (Sudoku, mazes, ARC) do not transfer for free; tabular is its
own regime (irregular targets, uninformative features, no rotational invariance).

## 2. Prime directive

**A recurrent model's accuracy in isolation is not a result.** The repo exists to measure
the *delta* between a recurrent-refinement model and matched non-recurrent controls,
across multiple seeds, with variance reported. Every experiment produces
`Δ = metric(recurrent) − metric(control)`, never a lone recurrent number.

Prior lesson that shapes our ordering: the ARC-Prize autopsy of HRM found the **outer
refinement loop and deep supervision** — *not* the high-level/low-level hierarchy — were
the active ingredients. So we validate the loop before we ever build the hierarchy (§9).

## 3. The canonical task suite (the thing we orient around)

Synthetic because cloud agents need: no downloads, full determinism, low cost, an
explicit difficulty knob, and a built-in out-of-distribution / extrapolation axis.

| Task | Name | Role | Tests |
|------|------|------|-------|
| 0 | `linear` | smoke test / tripwire | pipeline, metrics, training loop |
| A | `parity` | **core diagnostic** | irregular target + uninformative features |
| B | `iterated` | **mechanistic test** | loops ≈ algorithm steps; depth-extrapolation |
| C | `compositional` | hierarchy probe | **DEFERRED — phase 2 only** (see §9) |

**Seed discipline (applies to all tasks).** Each task instance is a pure function of two
seeds: a `task_seed` that defines *the function* (e.g. which bits are informative), and a
`sample_seed` that defines *the rows*. Train and test split on `sample_seed` and **share
the same `task_seed`** — same function, different rows. Never resample the function per
split.

**Metric.** Classification accuracy, plus exact-match (whole-row correct) for multi-output
targets. Report both.

### Reference generators — these *are* the spec

Treat the code below as canonical. If you reimplement, match the semantics exactly and
add a determinism test (§5).

```python
import numpy as np

# --- Task A: k-sparse parity in distractors -------------------------------------
# Hits two Grinsztajn failure modes at once: uninformative features + irregular target.
# Difficulty ladder is over k (interaction order). Distractors are the d-k noise bits.
def make_parity(n, d, k, task_seed, sample_seed, noise=0.0, symmetric=False):
    fn_rng  = np.random.default_rng(task_seed)            # fixes the function
    row_rng = np.random.default_rng(sample_seed)          # fixes the rows
    informative = fn_rng.choice(d, size=k, replace=False)  # shared train/test
    X = row_rng.integers(0, 2, size=(n, d))
    y = X[:, informative].sum(axis=1) % 2                  # parity of k bits
    if noise > 0:
        flip = row_rng.random(n) < noise
        y = np.where(flip, 1 - y, y)
    if symmetric:
        X = 2 * X - 1                                      # {0,1} -> {-1,+1}
    return X.astype(np.float32), y.astype(np.int64), informative

# --- Task B: emulate a T-step computation ---------------------------------------
# Elementary cellular automaton, periodic boundary. `rule` in 0..255 selects the
# update (90 = XOR neighbors, linear/easy; 110 = Turing-complete/hard; 30 = chaotic).
# The whole HRM/TRM thesis in one task: N refinement loops should emulate N CA steps.
def ca_step(s, rule):                                      # s: (..., w) of {0,1}
    left, center, right = np.roll(s, 1, -1), s, np.roll(s, -1, -1)
    idx = (left << 2) | (center << 1) | right              # neighborhood -> 0..7
    return (rule >> idx) & 1

def make_iterated(n, w, T, task_seed, sample_seed, rule=90, distractors=0):
    row_rng = np.random.default_rng(sample_seed)
    s0 = row_rng.integers(0, 2, size=(n, w))
    s = s0.copy()
    for _ in range(T):                                     # T = difficulty, match to loops
        s = ca_step(s, rule)
    X = s0
    if distractors > 0:
        fn_rng = np.random.default_rng(task_seed)
        noise = fn_rng.integers(0, 2, size=(n, distractors))  # static, uninformative
        X = np.concatenate([s0, noise], axis=-1)
    return X.astype(np.float32), s.astype(np.int64)        # multi-output: state after T

# --- Task 0: smoke test ----------------------------------------------------------
def make_linear(n, d, task_seed, sample_seed):
    w = np.random.default_rng(task_seed).standard_normal(d)
    X = np.random.default_rng(sample_seed).standard_normal((n, d))
    y = (X @ w > 0).astype(np.int64)
    return X.astype(np.float32), y
```

### The depth-extrapolation diagnostic (Task B) — our strongest evidence

Train the model unrolled for `R` refinement steps on computations of length `≤ T_train`.
At **test time**, unroll for `R' ≥ R` on `T > T_train`. If the recurrence has learned the
*step operator* (rather than baking in a fixed depth-`T` circuit), giving it more loops
should recover longer computations. Accuracy-vs-`(R', T)` that holds up under extrapolation
is the cleanest signal the loop is doing algorithmic work and not merely acting as a deep
net. If extra test-time loops do **not** help, the recurrence is just learned depth —
report that plainly.

## 4. Model & control contract

**Model API.** `forward(X) -> prediction`, and optionally a **sequence of per-step
predictions** (one readout per refinement step) so deep supervision is possible.

**Recurrent core (TRM-style default).** Maintain a latent `z` and an answer state. For `N`
steps: (i) update `z` given `(input, current answer, z)`; (ii) update the answer given `z`.
Read out from the answer state. Deep supervision = a loss on the readout at each step (or
the final step). Optional learned halting (ACT/PonderNet-style) is a later knob, not v0.

**Controls (MANDATORY — at least one, ideally both):**
- **(a) param-matched feedforward** — same parameter budget, no weight sharing, no loop.
  Isolates *does refinement help beyond raw capacity*.
- **(b) depth/compute-matched untied stack** — the same block stacked `N` times *without*
  weight tying. Isolates *does weight-tied recurrence help beyond depth*.

**Ablate deep supervision separately from the loop.** They are two ingredients; do not
confound them.

## 5. Invariants (non-negotiable)

1. **Control contract** (§4): no recurrent result ships without its matched control(s).
2. **Seeds & variance:** ≥ 5 seeds per config; report mean ± std (or the full
   distribution). No single-seed claims — tabular/small-model results are seed-sensitive.
   *Caveat for the paired sign test:* the two-sided exact binomial cannot reach p<0.05 with
   fewer than **6** seeds (5/5 → p=0.0625). Use ≥ 8 seeds when a significant sign-test call is
   the point; M3a/M3b use 10. The 5-seed floor is for variance bands, not significance.
3. **Determinism:** data is a pure function of `(config, seed)`; seed every RNG; pin
   dependency versions.
4. **No network in the task path:** the canonical suite must run with zero downloads.
5. **Tiny-first:** start at the smallest `(d, k, T, model size)` that exhibits the effect;
   scale only once the phenomenon is visible. Respect cloud compute.
6. **One knob per ablation:** vary a single axis; fix and log everything else.
7. **Run records:** every run writes `{config, metrics, seed, git SHA}` to `results/`.
8. **Every new task generator gets a determinism test** (same seeds → identical bytes).

## 6. Repo layout

```
.
├── CLAUDE.md                 # this file — keep §11 current
├── README.md
├── pyproject.toml
├── configs/                  # a run = one YAML (task+model+control+train+eval) + seed
│   ├── tasks/                # canonical task definitions
│   ├── models/
│   └── experiments/          # runnable: pairs a model WITH its control(s) on a task
├── src/looptab/              # package name `looptab` is a placeholder — rename freely
│   ├── data/                 # synthetic generators (parity, iterated, linear, ...)
│   ├── models/               # recurrent refinement core + controls
│   ├── train/                # loop, deep supervision, (later) halting
│   ├── eval/                 # metrics, extrapolation harness, control comparison, Δ
│   ├── registry.py           # name -> object, so configs stay declarative
│   └── run.py                # single entry point
├── experiments/              # thin launch scripts
├── results/                  # run records (raw gitignored; summaries tracked)
└── tests/                    # generator determinism, shapes, metrics
```

**Config-driven.** A single config plus a seed must fully determine a run. A registry
resolves string names (`task: parity`, `model: trm`, `control: [ff_matched, untied_stack]`)
to objects. This mirrors the substrate-over-scripts approach: agents add a new task/model
by registering it, not by editing the training loop.

## 7. Conventions

- Python ≥ 3.11, PyTorch. Env via `uv`. Lint/format via `ruff`. Tests via `pytest`.
- Configs as `pydantic` models (or dataclasses); type hints everywhere; small modules.
- One entry point: `python -m looptab.run --config configs/experiments/<x>.yaml --seed 0`.
- Don't add a dependency without a reason worth writing down in the PR/commit.

## 8. Methodology guardrails

- The recurrent model **trivially degenerates into a deep net** — the matched controls are
  the entire point of the comparison. Guard them jealously.
- **Verify, don't assume, the extrapolation claim** on Task B (§3). It's the difference
  between "recurrence is algorithmic" and "recurrence is dressed-up depth."
- **Deep supervision is itself a strong ingredient** (per the ARC autopsy). Attribute its
  effect separately so a positive result isn't silently credited to the loop.
- **No tuning on the held-out config-level meta-test.** Tune on validation; report on a
  config you didn't touch.
- **Report negative results plainly.** A clean null (loop ≈ control across the difficulty
  ladder) is a genuine, publishable-internally finding here, not a failure to be buried.

## 9. Scope discipline — do NOT do these yet

- No real / downloaded datasets (ARC, Sudoku, OpenML, the Grinsztajn suite) — synthetic
  first. Real tabular comes *after* the synthetic story is clear.
- No RL extension.
- **No H/L hierarchy** (Task C, two-module HRM split) until the single-loop refinement
  beats its control on Task A *and* Task B. The autopsy says earn the hierarchy.
- No large models. No speculative architecture zoo. Tiny-first, one variable at a time.

## 10. For agents working in this repo

1. Read this whole file before editing.
2. Validate any change against **Task 0 first, then Task A**.
3. **Always emit the control** alongside the recurrent run; report `Δ` and variance.
4. Write a determinism test for any new generator.
5. Keep diffs scoped to one milestone item.
6. Update §11 when you land something. That section is how the next (context-free) agent
   knows where the repo stands.

## 11. Project status / next milestone

Full per-milestone narratives (tables, readings, caveats for M0/M1/M2/M2-confirm and
later) live in **`results/LOG.md`**. Keep this section terse: current state, the
behaviour-changing conclusions, and the next pointer. Append detail to LOG.md, not here.

### (a) Current state — what exists, where

- **Tasks:** Task 0 `linear` (smoke), Task A `parity`, Task B `iterated` (CA), `multi_parity`
  (M6a both-axes probe: `w` independent k-parities), and `converge` (M8 variable-compute
  FIXED-POINT task: map s0 → the converged fixed point s_inf of a *converging* CA, e.g. rule 92;
  per-instance depth varies, target is a genuine fixed point), and `hopfield` (M13 NON-ECA
  hard-convergence probe: a dense binary threshold / Hopfield attractor net, s0 → converged
  attractor; integer symmetric `W` + integer self-coupling `γ` for guaranteed synchronous
  convergence — all-integer ⇒ bit-exact; built to test whether the joint-state result leaves the
  ECA family; **M14 adds a `bandwidth` knob** — zeros couplings beyond ring distance b, giving a
  *local-but-non-CA* threshold net (b small = local, b=w//2 = dense), the locality probe), and
  `mixed_converge` (M15 DEEP+NON-UNIFORM+LOCAL probe: a per-position MIXED CA — each cell runs its
  own radius-1 rule from a `rule_set`, default orbit1 {78,92,141,197} — iterated to a fixed point,
  **rejection-filtered to the convergent basin** since a spatial mix is not globally convergent;
  local + temporally-uniform but spatially non-uniform; `mixed_ca_step` is the per-position step).
  Generators in `src/looptab/data/generators.py`, determinism-tested in
  `tests/test_generators.py`; `make_trajectory_dataset` dispatches iterated/converge/hopfield/
  mixed_converge. Task C (hierarchy) is **gated, unbuilt** (§9).
- **Models/arms:** `trm` (weight-tied refinement loop, optional per-step readouts),
  `ff_matched` (§4a param-matched shallow MLP), `untied_stack` (§4b untied, ~`n_steps`×
  params — a confounded ceiling, NOT param-matched), `untied_matched` (§4b untied,
  width-shrunk to the loop's budget — the *clean* tying control), and `trm_decoupled` (M10
  mechanism ablation: the loop with **per-cell** refinement — each output cell has its own
  latent slice and sees only its own answer, severing the joint multi-output state; budget-
  matched, multi-output only). In `src/looptab/models/{trm,controls,decoupled}.py`, registered
  in `src/looptab/registry.py`. **Determinism exception (M11):** unlike every 2-D arm, `trm_decoupled`'s
  3-D batched matmul `(B,w,m)` has thread/BLAS-order-sensitive reductions — it is reproducible only at a
  fixed `num_threads` (committed runs use 1) and its EM does NOT reproduce bit-for-bit across environments
  (M10 vs M11 differ ~±0.015; the effect sizes dwarf this). The "bit-identical" guarantees below cover the
  2-D arms only.
- **Train/eval:** deep supervision is a **per-arm weight** (`src/looptab/train/loop.py`),
  not a global flag. Three training routines: `train` (standard), `train_curriculum` (M3b
  depth-curriculum + step-aligned DS), and `train_progressive` (M7 Deep Thinking progressive
  loss: detach `(T−k)` steps, gradient on `k`, modes `progressive_final`/`progressive_step`).
  `TRM.forward` takes optional `init_state`/`return_state` (additive, bit-identical when unused)
  so a rollout can be detached and resumed (M7). Metrics `accuracy` / `exact_match` / `majority_baseline` and the
  single-pass `evaluate` (`src/looptab/eval/metrics.py`). Paired Δ with variance is
  `delta_report`. **Data loading uses a custom in-memory `InMemoryLoader`** (not torch
  `DataLoader`) for the RAM-resident synthetic suite — it reproduces `DataLoader`'s exact
  per-epoch RNG protocol so training is **bit-identical**, just faster (see §5.3); if you
  ever swap it back, re-check determinism against the committed results. The runner also
  **pins CPU threads to 1** (`TrainConfig.num_threads`, default 1) — these models are below
  torch's matmul-parallelization threshold, so the default (= core count) only oversubscribes
  (≈3× slower at 8 threads, worse on many-core cloud boxes); bit-identical, so set
  `num_threads: null` only if you scale the models up. The per-seed loop can also run across a
  **process pool** (`ExperimentConfig.parallel_workers`, default 1 = serial) — bit-identical to
  serial (seeds self-reseed), ~Ncores× faster; raise it on ≥5-seed sweeps/grids.
- **Runner:** one entry point `python -m looptab.run --config <yaml>`. Supports a 1-D
  `sweep`, an N-D `grid`, and a depth-`extrapolation` harness (`grid` and `extrapolation`
  are mutually exclusive). Emits per-arm curve CSV, per-config Δ CSV, JSON record (config +
  metrics + seed + git SHA), and PNGs if matplotlib present (`src/looptab/run.py`).
- **Configs:** `configs/experiments/` (m0…m2-confirm, m3a/m3b, m4_parity_grid,
  m5_parity_wall_n16k/n64k, m6a_multi_parity_grid, m7_progressive_extrapolation, m7b_progressive_alpha1,
  m8_converge_adaptive, m8b_converge_grid, m8c_converge_fair, m9_converge_width,
  m10_decoupled_converge, m11_size_{small,base,large}, m12_hardconv_orbit,
  m13_hopfield_{screen,converge,large}, m14_{local_screen,local_ladder,dense_anchor},
  m15_{mixed_screen,mixed_converge,uniform_anchor}).
- **`hopfield` `bandwidth` regime (M14) — locked setting:** the local ladder needs **w=48** (w≤32 has
  no clean local regime — convergence-vs-triviality tension); b∈{2,4,8} at `γ=10` all 10/10
  convergent, balanced, non-trivial (triv ≤5%), settle ≤6 steps; the dense end (b=24) needs `γ=16`
  (a single γ can't span local+dense, hence the split local-ladder/dense-anchor configs).
- **`converge` rules — verified converging families:** {13, 78, 92} (M8), {140, 232} (M11), {69, 79, 93,
  141, 197} (M12). **The "balanced+deep converging" ECAs are EXACTLY 8 rules = two symmetry orbits** (M12
  256-rule screen): orbit 0 {13,69,79,93}, orbit 1 {78,92,141,197} — all are ff-hard and show the joint-state
  mechanism; {140,232} are ff-easy and do not. **w=16 is unusable for rules 13 & 232** — limit-cycle initial
  states on a w=16 ring (never reach a fixed point; the generator raises). Screen new rules over MULTIPLE
  seeds at the run's `n` (a single-seed screen missed this); **w∈{24,32} verified clean for all these rules**
  (0 unconverged / 480k draws), convergence depth ≤22 ≪ the `max_steps=4·w` cap.
- **Metrics:** `accuracy` / `exact_match` / `majority_baseline`, the single-pass `evaluate`, and
  (M9) `coherence_excess = EM − token_acc**w` (whole-row coherence beyond what independent per-cell
  errors would give; >0 = errors clustered) with a `mean_wrong_per_row` companion — all in
  `src/looptab/eval/metrics.py`; paired Δ with variance + sign test is `delta_report`.
- **Tests:** `tests/` — generator determinism, model shapes/param-ratios (incl. the M10
  decoupled no-cross-cell-leakage invariant), runner determinism/independence, coherence-metric
  math, and (M13) `make_hopfield` determinism/fixed-point/balance. Run `uv run --extra dev pytest -q`
  (121 tests); lint `uv run ruff check`.
- **`hopfield` regime (M13) — locked setting:** `weights=hebbian, n_patterns=12, γ=16, distractors=8`,
  w∈{24,32}. Screened multi-seed over the real task_seeds 42..51: **0/10 non-convergence raises**,
  balanced (majority ~0.50), per-row convergence depth typical **median ~2–3** (batch-max ~10 ≪ the
  8·w cap; >87% of rows settle in ≤4 steps, so `n_steps=6` is ample, NOT starving — comparable to
  rule 78's median ~3), and **ff-HARD** (ff EM ~0.26 @ w=24 / ~0.14 @ w=32 — the same hard regime as
  the hard-convergence ECAs). Committed runs
  pin an explicit integer `γ` so the generator is bit-exact; `γ=None` auto-derives `ceil(-λ_min)+margin`
  (float eigen-solve, for screening only). **`γ` too small → 2-cycles → the loud guard raises** (test
  covers it). Despite clearing every precondition, the loop's coherence result does NOT reproduce here
  (M13 §11(b)).

### (b) Behaviour-changing conclusions to date (read before re-running anything)

- **Always use `untied_matched`, not `untied_stack`, as the §4b control.** `untied_stack`
  has ~4× the loop's params; a "clean" Δ against it confounds tying with capacity — a §8
  trap that flipped M2's first-round conclusion. `untied_stack` is a labelled ceiling only.
- **Deep supervision is HORIZON-DEPENDENT, not inert (M3b overturns the prior null at short
  rollout).** Final-state DS (every step ↔ s_T) stays neutral-to-negative everywhere
  (M0/M1/M2/M2-confirm + M3b Δ(finalDS−nods)=−0.017 at the T=8 reference, p=.002). **But
  step-aligned DS (step i ↔ sᵢ) is a large WIN at short horizon** — under a T~1..8 curriculum,
  on the R′=T diagonal, T=4 acc 0.838 / EM 0.285 vs nods 0.676 / 0.046 (paired Δ=+0.162, 10/0
  seeds, sign-test p=.002). The sign of Δ(stepDS−nods) FLIPS with depth: **+0.162 at T=4** vs
  **−0.064 at the T=8 reference** (0/10, p=.002 each — two distinct, oppositely-signed paired
  tests). So "DS is inert" was an artifact of mis-specified (final-state) DS; step-alignment
  matters, but only where the rollout is short.
- **Tying beats a *fair untied* stack on Task B at SHALLOW depth only** (Δ(loop −
  untied_matched) > 0, p=.002, in all M2-confirm cells *and* every M3a T=4 cell) — **but the
  edge does NOT scale with depth; it vanishes by T≥8 (M3a)**. The M2 framing "tied recurrence
  buys depth *and* width from one budget" is now **softened**: only the *width* half holds;
  the *depth* half is **unsupported**.
- **At Task B depth T≥8 the s₀→s_T target is unlearnable one-shot for EVERY arm** (M3a: test
  AND train collapse to baseline; even the 16× `untied_stack` ceiling fits only ~0.78 train /
  ~0.52 test). An optimization/learnability wall, not a depth-capacity separation. A T~1..8
  curriculum (M3b) does lift the in-range reference (T=8 nods 0.65 vs M3a's ~0.52 one-shot) but
  does not crack the wall past the trained horizon.
- **The loop does NOT beat its §4a control `ff_matched` on Task B** at any depth (M2-confirm
  wins 1/6 by noise; M3a ≤0 at every T, significantly negative at T=4/T=8). On Task A it's the
  mirror: beats `ff_matched` but only ties `untied_matched`. So on neither task does
  the loop beat *both* controls → the literal §9 gate is unmet. The loop's defensible property
  is "never the worst param-matched arm" (robustness), not dominance. The gate as literally
  worded may be unsatisfiable by a generalist judged against single-axis specialists.
- **Task A "depth helps, tying neutral" REPLICATES across d × k and strengthens with k (M4).**
  Over d∈{20,40,80} × k∈{3,4,5} (10 seeds), Δ(loop − ff_matched) reproduces at d=20,k=4 (+0.228,
  6/0, p=.031) and grows at k=5 (**+0.497**, 10/0, p=.002; ff at pure chance 0.503 while the loop
  is perfect) — no longer a single-`d` artifact. But Δ(loop − untied_matched) is **non-significant
  in all 9 cells**, and the depth Δ(untied_matched − ff_matched) carries the same sign/significance
  as Δ(loop − ff) — so the win is **depth, not tying**, and the loop beats *both* controls in **0**
  cells. Raising d mostly hits an **unlearnability wall** (d≥40,k≥4: all arms at test-chance — deep
  arms overfit train, ff_matched underfits it) rather than separating arms, so the clean signal
  lives at d=20 (all k).
- **The M4 d≥40 wall is SAMPLE-complexity-bound and lifts to ALL-arms-solve with NO separation
  (M5).** Sweeping `n_train` 4k→16k→64k on M4's d≥40 cells, four of the five walled cells
  (d=40,k=4; d=40,k=5; d=80,k=3; d=80,k=4) go chance→1.000 for *every* arm together — there is no
  hidden architectural edge behind the wall. **M4's d=80,k=3 "depth hint" was just `ff_matched`
  sample-starvation** (dissolves to 1.000-all by 16k), and the **16k d=80,k=4 "loop-beats-both
  hint" was a transient sample-efficiency ordering** (the tied deep arms generalize at smaller `n`
  than the controls, but it is ns at ±0.21 and erased to 1.000-all by 64k). So **no significant
  loop-beats-both cell exists anywhere on the ladder.** The lone exception, **d=80,k=5, is a
  CAPACITY wall not a sample wall**: it stays at test-chance even at 64k and train accuracy *drops*
  (overfit→underfit flip) — the ~14k-param arms can't fit the (80-choose-5) parity at all, so more
  data is the wrong lever (needs a bigger model, out of scope). Tying stays neutral and DS inert at
  scale (M5 confirms M4).
- **The §9 gate is unmet in EVERY tested both-axes cell, and the loop's "never-worst" robustness
  claim is DEFINITIVELY FALSIFIED (M6a).** The `multi_parity` both-axes probe (`w` independent
  k-parities; depth via k, width via w) was built to be the one task needing *both* axes at a fixed
  budget — where a generalist should beat both single-axis controls. It yields **zero
  loop-beats-both cells**: where Δ(loop−ff) is significant (k=4, all w: +0.18…+0.27, p=.002),
  Δ(loop−um) is ns (+0.02…+0.04). `multi_parity` **unifies Task A and Task B**, with `k` as the
  axis dial: k=4 reproduces Task A (loop beats shallow ff, ties deep um) and *extends it to
  multi-output*; k=3 reproduces Task B (wide shallow ff is **best**, loop **significantly beaten**
  by it, −0.15…−0.19, p≤.004). **Strength of the two claims differs — keep them distinct:** (1)
  "loop never beats both" is **strong-but-not-proven** — the k=4 width cells have Δ(loop−um) *positive
  in the predicted direction* but **under-powered (ns, 7/3)**, and the grid is coarse (one d, one
  budget, w 1→4→8), so this is "unmet in every tested cell with a structural k-dial reason," not a
  proof of impossibility (higher seeds on k=4/w≥4 would settle it). (2) "never-worst is falsified" is
  **conclusive** (k=3 wide, loop significantly < ff, p≤.004). Two casualties: (a) the M2 "tying buys
  width" half does **not** replicate on parity (Δ(loop−um) ns at all w — the CA tying edge was
  CA-specific); (b) the loop is **depth-positive** (owns high-order parity vs shallow ff) and
  **tying-neutral** — NOT a robust generalist. This is strong evidence to relax the §9 wording
  (§11(c)(i)), with the under-powered k=4 cells the place to push for a stricter result.
- **No transferable step operator; the loop does NOT extrapolate in depth (M1 + M3b + M7).**
  Over-unrolling R′>R decays to baseline, and OOD depth T>T_train collapses to baseline for
  every arm. M3b applied the two named levers (T-curriculum + step-aligned DS) and the OOD
  collapse (T≥12) STILL holds — a stronger, cleaner null than M1's. Within the curriculum
  there's only weak compositional signal (R′=T tracks for T≤8, tops out ~0.58).
- **Deep Thinking's progressive loss is INERT on Task B (M7) — clean negative.** The progressive
  loss (recall + detach-`(T−k)`-then-gradient-`k`; Bansal 2022) was applied to the M1/M3b null. The
  load-bearing fact is **in-distribution**: the detach adds nothing where we can measure it —
  prog_final ≈ nods and prog_step ≈ stepDS at T=4 *and* T=8 (all Δ ns), so the mechanism is inert
  independent of any extrapolation/convergence argument. It also does not crack the OOD T≥12 collapse
  (every arm at baseline). **A hypothesis for *why* (NOT tested here): ** progressive loss is built
  to instill *path-independence* (converge to a fixed attractor, stable under over-unrolling), but
  Task B's CA is **non-convergent** — `s_T` is a *moving* target (`s_{T+1}≠s_T`, no attractor), so a
  steady-state bias may be mismatched. **This is a rationalization, not a finding** — M7 has no
  convergent-target control, so it cannot distinguish "wrong bias for CA" from "inert at this
  scale/tuning." **α was swept {0.5, 1.0} (M7b):** α=1 (pure progressive, no anchor) is *strictly
  worse* — OOD still collapses, and in-distribution the detach now *significantly underperforms* its
  non-detach counterpart (Δ(prog_step−stepDS)=−0.082, p=.002) — so the "you didn't tune α" objection
  is closed; still one rule/width. The non-convergence hypothesis was then TESTED in M8 (below) and
  **FALSIFIED** — over-unrolling decays even on a convergent fixed-point target, so the decay is
  intrinsic to the learned operator, not caused by task non-convergence. (Compute note: progressive
  arms run ~2× grad forwards/batch — uneven, but cuts toward the null.)
- **FIRST loop-beats-both signal in the project: whole-row EXACT-MATCH on a fixed-point task, driven
  by recurrence/tying coherence — but adaptive computation FAILS and the M7 hypothesis dies (M8/M8b).**
  The `converge` task (map s0 → the converged fixed point s_inf of a converging CA; per-instance depth
  varies) was built to let the loop win via *adaptive compute* (unroll more at test on hard instances).
  That **failed**: over-unrolling decays (stepDS 0.922→0.857 as R′ 6→24) even though the target is a
  genuine fixed point — falsifying M7's "decay is because CA is non-convergent" (the decay is intrinsic
  to baked-in trained depth). BUT a genuine, properly-isolated **tying-positive** surfaces (M8c, after
  an adversarial review caught a supervision confound in M8b's headline arm): the weight-tied loop beats
  a **fair untied stack** on whole-row exact-match in **ALL 6 grid cells at EQUAL supervision** —
  Δ(nods−untied)=+0.05…**+0.37** EM (also +0.06…+0.09 token-acc), and Δ(stepDS−untied_stepDS)=+0.07…+0.30
  EM, both 10/0, p=.002. It is **tying, not depth** (the *deep* untied control is WORST on EM everywhere)
  — the cleanest tying-positive in the project (parity was tying-neutral). The stronger "beats **both**
  controls" claim is real but NARROWER than M8b stated: clean (plain loop, equal supervision) it holds in
  **3/6 cells, all w=24, EM-only**; at w=32 the plain loop *loses* to wide shallow ff on token-acc
  (−0.02, p=.002), so M8b's "4/6 via stepDS" was partly supervision-driven (the §8 trap). Mechanism =
  whole-row coherence from recurrence, **not** adaptive computation. Doesn't satisfy the literal §9 gate
  (Tasks A/B), but is a concrete counterexample to "the loop never beats both anywhere" (w=24, EM).
- **The M8 tying-positive STRENGTHENS under a width sweep, and the "whole-row coherence" mechanism is
  CONFIRMED at matched token-accuracy (M9).** Sweeping `w∈{12,16,24,32,48}` on `converge` (rule 78, M8c
  fair arms) with a new `coherence_excess = EM − token_acc**w` diagnostic: (1) **tying > fair untied is
  width-robust** — Δ(loop−untied) positive on token-acc in all 5 widths (10/0, p=.002) and on EM in 9/10
  width×regime cells (the one exception is near-saturation at w=12); it is the project's durable
  architectural pro-loop fact, NOT a w=24/32 artifact. (2) **Clean loop-beats-both (plain loop, equal
  supervision, EM) is a w≤24 REGIME** (holds w=12/16/24, broader than M8c's single w=24 snapshot) and
  **vanishes by w≥32**, where the wide shallow `ff_matched` overtakes the loop on token-acc (Δ(nods−ff)
  acc +0.038→+0.003(ns)→−0.021→−0.031, crossover ~w=24; EM is the loop's *durable* edge, lasting one
  width-step longer). *(rule 78 only — the w≤24 boundary is not swept over rule or model size.)* (3)
  **Mechanism — the clean statistic is EM-at-matched-token-acc:** loop vs ff @ w=24 (token-acc tied,
  Δacc +0.003 ns) wins EM +0.133 (10/0, p=.002) — at equal per-cell accuracy, recurrence/tying makes
  coherent whole rows the MLP can't. **[Adversarial-review correction]** the `coherence_excess` metric
  does NOT add an independent confirmation: at matched acc Δ(coh) ≡ Δ(EM) (same fact twice), and its
  *cross-arm* Δ is confounded by token-acc *level* AND per-row *dispersion* (Jensen: EM =
  mean_row(row_acc**w) ≥ (mean row_acc)**w inflates it without clustering). So `coherence_excess` is a
  **per-arm descriptor only** (its w≈24 "peak" is mechanical — EM saturates at small w, collapses at
  large w); the cross-arm mechanism claim rests on EM-at-matched-acc, not a coh Δ. Loop's sharpened
  value: **tied recurrence buys whole-row coherence on multi-output fixed-point targets — width-robust
  over a fair untied stack, and over a shallow MLP at matched token-acc for w≤24 (rule 78) — NOT a
  token-acc edge at large w, NOT adaptive compute, NOT depth-extrapolation.**
- **The whole-row-coherence mechanism is the JOINT multi-output state, not recurrence per se (M10).**
  `trm_decoupled` (the loop refining each output cell in its *own* latent, seeing only its own answer —
  budget/recurrence/recall/supervision all matched to `trm_nods`, ONLY the joint state removed) **loses
  the coherence**: Δ(nods−decoupled_nods) EM +0.51/+0.39/+0.09 at w=16/24/32 (10/0, p=.002), and
  +0.32/+0.32/+0.14 at *equal step-aligned supervision* (where the decoupled arm trains stably, so it is
  NOT an optimization artifact). Stronger than the pre-registered fork required: the decoupled loop falls
  **below the shallow §4a MLP** on both token-acc AND EM in all 3 widths (Δ(decoupled−ff) 0/10, p=.002) —
  so the joint coupling is not a bonus on top of recurrence, it is the thing carrying the loop's entire
  value here. M9's loop-vs-ff @ w=24 anchor reproduces (Δacc +0.003 ns / ΔEM +0.133). **Sharpened value
  statement: tied recurrence with a JOINT multi-output state buys whole-row coherence** — the "joint"
  qualifier is now load-bearing and demonstrated. (Caveat: rule 78 / one size; `decoupled_nods` is
  fragile under final-loss-only, so lean on the step-aligned pair for the trainability-controlled Δ.)
- **The joint-state coherence result GENERALIZES across model size (and STRENGTHENS with it — NOT a
  tiny-model artifact) but is OPERATOR-FAMILY-SPECIFIC, and "loop-beats-both" is capacity-contingent (M11).**
  Swept model size (hidden=latent 32/64/128) × operator family ({13,78,92} + new {140,232}) on `converge`
  at w∈{24,32}. **(1) Size:** for {13,78,92} the **joint-state mechanism (M10)** holds 10/0, p<.05 at ALL
  THREE sizes — both Δ(nods−decoupled) and the trainability-clean Δ(stepDS−decoupled_stepDS) — and the gap
  *grows* with size (large Δ(nods−decoup) EM +0.53…+0.66 vs base +0.37); the **tying-positive P1**
  (Δ(nods−untied) EM>0) also holds at all sizes. The one capacity-contingent leg is **loop-beats-both
  (Δ(nods−ff))**: **NEGATIVE at small** (ff *beats* the loop) — −0.04…−0.07, p<.05 at w=24 (all 3 rules);
  weaker at w=32 (rule-13/w32 ns) — positive w≤24 at base (M9), **strongly positive at BOTH widths at large**
  (+0.12…+0.25). So scaling up does NOT erase the
  edge — it amplifies it; small models simply lack capacity for the joint refinement to beat a shallow MLP.
  (Overfit guard: large train−test gap small, no wall — the size effect is real.) **(2) Operator family:**
  the result does **NOT** transfer to the two new families. On **rule 232** (majority/shallow, balanced)
  `ff_matched` **dominates** (Δ(nods−ff) EM −0.45…−0.52) and the mechanism is **reversed** (decoupling
  helps); on **rule 140** (deep but ff-easy) the loop **ties** ff and decoupling does **not** collapse
  coherence (Δ(nods−decoup) ns). Cause: both new rules are per-cell *easy* (ff EM 0.82–0.83 — the MLP
  already makes coherent rows), so there is no coherence gap to fill. **The loop's joint-state edge appears
  only where a shallow per-cell map FAILS on coherence** (i.e. {13,78,92}, ff EM ~0.31) — so the result is
  about a **subclass of hard-convergence operators**, not "multi-output fixed-point targets" in general.
  Sharpened value statement: **whole-row coherence via the JOINT state, on HARD multi-output fixed-point
  targets — robust over a fair untied stack and GROWING with model size, but NOT universal across operator
  families and NOT a capacity-independent "beats-both".**
- **"Hard convergence" (ff-hardness) is the operative axis, confirmed on the full untested membership of both
  ECA orbits (M12).** A 256-rule screen proves the **balanced+deep converging ECAs are EXACTLY two symmetry
  orbits** — {13,69,79,93} and {78,92,141,197} (8 rules); {13,78,92} sampled both. Running the M10 arm set on
  the **5 untested orbit-mates** {69,79,93,141,197} reproduces the whole result: all 5 are **ff-hard** (ff EM
  0.30–0.34 vs M11's ff-easy 0.82–0.83), and at w=24 the loop **beats both controls** on EM (Δ(nods−ff)
  +0.14…+0.19, Δ(nods−untied) +0.32…+0.42, all 10/0) with the **joint-state mechanism** intact (Δ(nods−decoup)
  +0.35…+0.45; trainability-clean Δ(stepDS−dec_sDS) +0.28…+0.35, 10/0; decoupled < ff, 0/10). So the result is
  a property of the **hard-convergence REGIME**, not 3 lucky rule numbers. Caveat: the orbit-mates are
  mirror/complement images of {13,78,92} (genuinely different datasets to our non-equivariant models, but not
  *dynamically* independent) — the screen proves no balanced+deep converging operator exists OUTSIDE this
  closure among 3-neighbour ECAs; a truly independent test requires leaving the ECA family.
- **The joint-state coherence result is CA/LOCAL-UPDATE specific — it does NOT leave the ECA family
  (M13, clean NEGATIVE).** Built `make_hopfield`, a **non-CA** hard-convergence target (a dense,
  fully-coupled binary threshold / Hopfield attractor net — basin-of-attraction is *intrinsically* a
  whole-row property, the strongest probe of the joint-state hypothesis) and ran the M10 arm set at
  base (hidden=64) AND large (128), w∈{24,32}, 10 seeds. The substrate clears **every** precondition:
  genuine multi-output fixed point, balanced, **ff-HARD** (ff EM ~0.26/0.14 — the same hard regime as
  {13,78,92}). Yet the result does **NOT** reproduce. **(1) Loop-beats-both FAILS** — the loop never
  beats `ff_matched` on EM; it is *significantly worse* at base/w24 (−0.063, 0/10, p=.002) and large/w32
  (−0.029, p=.021); per-arm EM is **ff 0.256 > loop 0.193 > decoupled 0.148 > untied 0.113** (base/w24)
  — the shallow MLP is the **best** coherence arm, the exact inverse of the ECAs (M9: loop beat ff
  +0.133). **(2) The JOINT-STATE MECHANISM essentially does not transfer** — the trainability-clean
  Δ(stepDS−decoupled_stepDS) is **ns in all 4 size×width cells** (p=.11/.11/1.0/.75) vs the +0.32…+0.66,
  10/0 it showed on ECAs (M10/M11); the final-loss Δ(nods−decoupled) is weakly positive (EM +0.03…+0.06,
  p≈.02–.04) but small, fragile, and **does NOT grow with size** (base ≈ large). **(3) CAPACITY DOES NOT
  REVIVE IT — the decisive M11 contrast:** on ECAs scaling 64→128 *strengthened* everything; here it does
  nothing (loop-beats-both stays negative, clean mechanism stays null), so the failure is **intrinsic to
  the substrate, not a tiny-model artifact**. **(4) The tying-positive P1 SURVIVES** (Δ(nods−untied) EM
  10/0, p=.002 at base both widths + large/w24; vanishes only at large/w32) — the loop beats a fair
  untied stack on coherence in 3/4 cells. **Budget caveat (do NOT call base P1 "budget-clean"):** at
  BASE the `untied_matched` control is the width-quantization breach the audit flags — **+2.46%/+3.08%
  OVER budget** (`within_tol=False`), so those cells are **conservative** (the loop beats a control with
  a small capacity *advantage*), not clean; the strictly-budget-clean P1 evidence is the **LARGE run
  (untied within tol, ratio 0.988/0.998), where large/w24 P1 = +0.047, 10/0, p=.002**. So **the M8–M12 result is CA/local-update specific, NOT a
  property of hard-convergence multi-output fixed points in general**; the loop's coherence edge appears
  to need a *local, spatially-structured* per-cell map (where the shallow MLP makes spatially-correlated
  errors the joint state repairs), absent on a dense net where the MLP already sees the whole row — and
  it is NOT a depth effect (per-row depth is comparable, median ~2–3, to rule 78 where the loop WON). The
  only regime-independent survivor is the tying-positive P1. (Hypothesis for the locality requirement is
  untested — §8; would need an intermediate local-but-non-CA substrate.)
- **The M13 locality hypothesis is FALSIFIED — the loop's edge is NOT about coupling locality (M14, clean
  NEGATIVE).** Built the intermediate substrate M13 called for: a `bandwidth` knob on the threshold net
  giving a *local-but-non-CA* attractor (b small = nearest-neighbour but per-position-irregular, b=w//2 =
  dense). Ran the M10 arm set across a **local→dense ladder b∈{2,4,8,24} at w=48, 10 seeds**. **(1)
  Locality does NOT revive the loop's edge:** Δ(loop−ff) is **negative at EVERY bandwidth on accuracy**
  (all 0/10, p=.002), `ff_matched` is the **best arm across the whole ladder**, and the direct test of
  M13's hypothesis — the trainability-clean joint-state Δ(stepDS−decoupled_stepDS) — is **null at both the
  local and dense ends** (ns), nothing like the ECA's +0.32…+0.66. (The dramatic b=2 ΔEM −0.80 is the
  LEAST informative cell: an *easy* task, ff acc 0.999, where BOTH recurrent arms collapse — recurrence is
  pointless there. The load-bearing cells are the hard end b=8/dense, where the loop still loses and the
  mechanism Δ is null.) **(2) ff tracks per-cell FAN-IN, and bandwidth↔depth is CONFOUNDED:** ff-easiness
  ⇔ small fan-in ≈ light-cone ≈ bandwidth × convergence-depth; the banded net drops the uniform rule AND
  collapses depth (median 1–2 vs the ECA's tail to ~22) at once, so M14 **cannot separate** "iterated
  translation-invariant local rule" from "deep convergence" as the loop-edge ingredient — the uniform-rule
  reading is a HYPOTHESIS, not isolated here. **(3) The joint-state mechanism does NOT transfer** (as M13).
  **(4) The tying-positive P1 SURVIVES** across the full local→dense ladder (Δ(loop−untied) acc 10/0 p=.002
  at all four b; budget within ±2%, though `untied` is ~1.7% *under* budget so P1 is conservative not
  exactly matched, and the local-ladder γ=10 is loud-guard-empirical not PSD-guaranteed) — the one
  regime-independent leg, demonstrated off-CA at both ends. **Net: the M8–M12 result is bounded to
  LOCAL-UPDATE (CA) hard-convergence — NOT explained by coupling locality, NOT hard-convergence fixed
  points in general; only P1 generalizes. Whether the CA-specific ingredient is the uniform rule or the
  deep convergence is NOT resolved by M14** (resolved in M15 below).
- **The M8–M12 result SPLITS into TWO mechanisms with DIFFERENT requirements (M15 — resolves the M14
  confound).** Built `mixed_converge` (a per-position MIXED CA: each cell its own radius-1 rule from
  orbit1, iterated to a fixed point, rejection-filtered to the convergent basin) — **deep + local +
  ff-hard but spatially NON-uniform (not a CA)** — and contrasted it with the uniform `converge` rule-78
  anchor at IDENTICAL protocol (only translation-invariance differs). **(1) The joint-state coherence
  mechanism (joint refinement ≫ per-cell, M10) is driven by DEEP+LOCAL structure, NOT translation-
  invariance — it TRANSFERS to the non-uniform mixed-CA:** trainability-clean Δ(stepDS−decoupled_stepDS)
  EM **+0.206 (10/0, p=.002) at w=24** (+0.053 at w=32), decoupled falls BELOW ff (−0.237, 0/10) — the
  M10/ECA signature on a non-CA. With M13 (dense ⇒ null) and M14 (shallow ⇒ null), the mechanism needs
  **local + deep** (a wide light-cone from composing a local update over depth), uniform or not. **(2)
  Loop-beats-the-shallow-MLP (the loop-beats-both headline, M8/M9) REQUIRES the UNIFORM rule:** at w=24
  the loop beats ff on EM-at-matched-token-acc for the uniform rule (**+0.133, 10/0, p=.002**) but only
  TIES on the mixed version (**−0.028, ns**) — same depth/hardness/rule-family/protocol. Uniformity makes
  the one-step operator maximally shared, which the weight-tied loop exploits to beat a one-shot MLP;
  remove it and the MLP catches up. (w≤24 effect, as M9 — both tie at w=32.) **(3) P1 survives on both**
  (conservative; untied over-budget). **Net: "tied loop + joint state buys whole-row coherence on hard
  CA targets" = a deep+local joint-state mechanism (transfers off-CA) PLUS a uniform-rule loop-beats-MLP
  edge (does not).**
- Each leg still rests on few configs: Task A now multi-`d`/multi-`k` (M4) and the d≥40 wall has
  been swept over `n_train` (M5 — it is sample-bound and lifts to all-solve, except d=80,k=5 which
  is capacity-bound); Task B depth swept (M3a) but unlearnable past T=4 one-shot; M3b on one rule
  (30) / one width (9).

### (c) Next milestone

**M3a, M3b, M4, M5, and M6a are all DONE** (full narratives in LOG.md). M3a falsified "loop edge
grows with depth" (it vanishes by T≥8; deep CA is unlearnable one-shot for all arms). M3b applied
the T-curriculum + step-aligned DS levers: step-aligned DS is a *real* short-horizon win (DS was
mis-specified, not inert) but did **not** yield a transferable operator. M4 replicated the Task A
parity leg across d∈{20,40,80} × k∈{3,4,5}: the loop's edge over `ff_matched` is **robust and
grows with k** but is purely **depth** (ties `untied_matched`), loop-beats-both in **0** cells.
M5 swept `n_train` on M4's d≥40 wall: **sample-bound, lifts to all-arms-solve with NO separation**.
**M6a** ran the top lever as an experiment: the `multi_parity` both-axes probe yields **zero
loop-beats-both cells** and **falsifies** the loop's "never-worst" claim (k=3 wide: loop significantly <
`ff_matched`). **M7/M7b** applied Deep Thinking progressive loss to the depth-extrapolation null — inert
(α∈{0.5,1}). **M8/M8b/M8c** built the variable-compute `converge` (fixed-point) task: adaptive computation
**FAILED** (over-unrolling decays even on a convergent target → falsifies M7's non-convergence
hypothesis), **but** surfaced a clean **tying-positive** — the tied loop beats a fair untied stack on
whole-row exact-match in **6/6** cells at equal supervision (M8c, after review caught a supervision
confound in M8b's headline). **M9** swept output width `w∈{12,16,24,32,48}` on `converge` + added a
`coherence_excess` diagnostic: the tying-positive is **width-robust** (token-acc 10/10 cells, EM 9/10),
clean **loop-beats-both is a w≤24 regime** (broader than M8c's lone w=24 cell, gone by w≥32), and the
**whole-row-coherence mechanism is CONFIRMED at matched token-acc** (loop vs ff @ w=24: +0.107
coherence_excess at equal accuracy, 10/0, p=.002). M8/M9 are the FIRST loop-beats-both anywhere — real
but narrow (multi-output fixed-point, EM/coherence, w≤24), and NOT adaptive-compute. **M10** built the
`trm_decoupled` ablation and **isolated the coherence mechanism to the JOINT multi-output state**:
refining cells *independently* (same budget/recurrence/supervision) loses the coherence (ΔEM +0.09…+0.51,
10/0; +0.14…+0.32 at equal step-aligned supervision) and falls *below* the shallow §4a MLP (0/10) — so
the "joint" qualifier on "tied recurrence buys coherence" is now load-bearing and demonstrated. **M11**
swept model size (32/64/128) × operator family ({13,78,92}+{140,232}) on `converge`: the joint-state
mechanism + tying-positive **generalize across size and STRENGTHEN with it** (NOT a tiny-model artifact),
but the result is **operator-family-specific** (the two new families are ff-dominated — the loop's edge
needs a per-cell-*hard* target) and **"loop-beats-both" is capacity-contingent** (ff wins at small; loop
wins and widens at large). **M12** confirmed **"hard convergence" (ff-hardness) is the operative axis**: a
256-rule screen shows balanced+deep converging ECAs are exactly two symmetry orbits {13,69,79,93}/
{78,92,141,197}, and the mechanism reproduces on **all 5 untested orbit-mates** (loop-beats-both + joint-state
collapse, 10/0) — so it is a property of the regime, not 3 lucky rules. **M13** left the ECA family entirely
(a dense threshold/Hopfield attractor net) and found the joint-state coherence result is **CA/local-update
specific** — at base AND large size it does NOT reproduce (loop-beats-both fails, ff is the *best* coherence
arm; the trainability-clean joint-state mechanism is ns in all 4 cells; scaling does not revive it), with only
the tying-positive P1 surviving. The M8–M12 result is now BOUNDED to local-update (CA) hard-convergence.
**M14** built the intermediate substrate M13 asked for (a `bandwidth` knob → a *local-but-non-CA* threshold
net) and showed **a local-but-non-CA net does NOT revive the loop's edge**: across a local→dense ladder
(b∈{2,4,8,24}, w=48) the loop loses to ff at every b and the trainability-clean joint-state Δ is null at
the hard end — so the M8–M12 result is **not explained by coupling locality**. (M14 confounds bandwidth
with convergence depth — separated in M15.) Only the conservative **tying-positive P1** survives off-CA.
**M15** built `mixed_converge` (a per-position MIXED CA — deep + local + ff-hard but spatially NON-uniform)
and contrasted it with uniform rule 78 at identical protocol, **resolving the M14 confound into a clean
DECOMPOSITION**: the **joint-state coherence mechanism** (joint ≫ per-cell, M10) is driven by **deep +
local** structure and **TRANSFERS** to the non-uniform mixed-CA (trainability-clean ΔEM +0.206, 10/0,
p=.002 @ w=24; decoupled < ff), while **loop-beats-the-MLP** (the loop-beats-both headline) **REQUIRES the
uniform rule** (loop beats ff +0.133 EM on uniform-78 but only ties on the mixed version, same
depth/hardness/protocol). P1 survives both. So the M8–M12 result = a deep+local joint-state mechanism PLUS
a uniform-rule loop-beats-MLP edge.

**No milestone is currently in flight.** Open threads, in rough priority:
- **THE highest-value remaining action — relax the literal §9 gate wording (now even better-scoped after the
  M13 bound).** M6a showed "beats both on A and B" is structurally unsatisfiable. The reframing the evidence
  supports: the loop's value is **whole-row coherence via the JOINT refinement state, on LOCAL-UPDATE (CA)
  HARD multi-output fixed-point targets** — robust over a fair untied stack (P1), growing with model size
  (M11), mechanism = the joint state (M10, all sizes), across the entire hard-convergence ECA regime (M12),
  **PLUS the broader-but-not-uniform tying-positive P1** (M13: P1 survives off-CA in 3/4 cells). **Scope it
  precisely:** NOT universal across operator families (ff-dominates easy {140,232}), NOT a capacity-independent
  "beats-both" (ff wins at small), **NOT a property of hard-convergence fixed points in general — it is
  CA/local-update specific (M13)**, NOT token-acc, NOT adaptive compute, NOT depth-extrapolation. Rewrite §9's
  gate around this; do **NOT** build Task C. *(This is a writing task — the experiments are done.)* Note
  M14/M15 further tighten the scope: it is **not about coupling locality** (M14 — a local non-CA net is
  ff-easy and the loop loses), and the M8–M12 result **splits into two mechanisms** (M15): a **joint-state
  coherence mechanism** needing *deep + local* structure (transfers off-CA to a non-uniform local deep
  map) and a **loop-beats-the-MLP** edge needing the *uniform translation-invariant rule* (does not
  transfer to a per-position-mixed rule). Frame the §9 rewrite around this two-part decomposition.
- **The experimental program is effectively complete.** No open experimental thread remains after M14
  closed the locality question and M15 separated the uniform-rule-vs-depth confound (the
  *local-AND-hard non-CA substrate* M14 called for — `mixed_converge` — is now built and run). Lowest-value
  leftovers only: a finer/larger size sweep; more rule families / a radius-2 mix; whether the uniform-rule
  loop-beats-MLP edge is recoverable at larger w. Default next action is the §9 rewrite.
- **Closed levers (do not redo):** depth-extrapolation via progressive loss / path-independence (M7/M8 —
  decay is intrinsic, not convergence-related); adaptive compute on a fixed-point target (M8 — decays);
  "lift the M4 sample wall" (M5); "re-judge via a both-axes task" (M6a); the decoupled-head mechanism
  question (M10 — coherence is the joint state); **the model-size axis (M11 — generalizes & strengthens);
  the rule/operator generality within ECAs (M11/M12 — family-specific to hard-convergence = two symmetry
  orbits, all 8 members confirmed, no other balanced+deep ECA exists); LEAVING the ECA family (M13 — the
  joint-state result is CA/local-update specific, does not transfer to a dense threshold net at any size);
  the LOCALITY hypothesis (M14 — a local-but-non-CA threshold net is ff-easy and the loop loses; the
  M8–M12 edge is NOT explained by coupling locality); the UNIFORM-RULE-vs-DEPTH question (M15 — a
  per-position MIXED CA, deep+local+ff-hard but non-uniform, SPLITS the result: the joint-state mechanism
  is deep+local and TRANSFERS to the mixed task, but loop-beats-the-MLP needs the uniform rule and does
  NOT — do not re-conflate these two legs).**
  A bigger-model probe of d=80,k=5 must scale the budget for *all* arms (M5).

## 12. Key references (for grounding a cold agent)

- **TRM** — Jolicoeur-Martineau 2025, arXiv 2510.04871. Base the recurrent core on this:
  single tiny 2-layer net, iterative answer refinement via a latent. Simpler/stronger than HRM.
- **HRM** — Wang et al. 2025, arXiv 2506.21734. The H/L hierarchy + 1-step-gradient + ACT.
- **ARC-Prize HRM autopsy** — arcprize.org/blog/hrm-analysis. *Why we test the loop before
  the hierarchy.*
- **Deep Equilibrium Models** — Bai, Kolter, Koltun 2019, arXiv 1909.01377. Fixed-point /
  1-step gradient that HRM's training rests on.
- **Universal Transformers** — Dehghani et al. 2018, arXiv 1807.03819. Weight-tied depth
  recurrence + adaptive compute, the archetype.
- **ACT** (Graves 2016, arXiv 1603.08983) / **PonderNet** (Banino et al. 2021,
  arXiv 2107.05407). Halting machinery for the later halting knob.
- **"Can you learn an algorithm?"** — Schwarzschild et al. 2021, arXiv 2108.06011. The
  easy-to-hard extrapolation logic behind Task B.
- **Grinsztajn et al. 2022** — arXiv 2207.08815. The three tabular failure modes Task A is
  built to probe.
- **Gorishniy "Revisiting DL Models for Tabular Data"** — 2021, arXiv 2106.11959. Baseline
  architectures (FT-Transformer / tuned MLP) for when real tabular arrives.

### Depth-extrapolation / "transferable step operator" mechanisms (added for the post-M6a mechanism hunt)

These target our central null — over-unrolling decays to baseline and OOD depth collapses (M1/M3b);
"the loop does not settle a stable fixed point." All are tiny-model, synthetic-algorithmic-task
results (no downloads), so they fit §5 tiny-first and the §4 control contract.

- **Deep Thinking nets ("End-to-end Algorithm Synthesis…: Logical Extrapolation Without
  Overthinking")** — Bansal, Schwarzschild, Borgnia, Emam, Huang, Goldblum, Goldstein 2022,
  arXiv 2202.05826 (NeurIPS'22). *The direct fix for our exact failure.* Two ingredients:
  **(1) recall** — concatenate the input to the recurrent module's input at *every* step so it
  can't be forgotten (our TRM already does a form via `cat[X, z, a]`); **(2) progressive loss** —
  per batch pick random `n, k` with `n+k < T_max`, run `n` steps with **gradients detached**, then
  `k` steps **with** gradients, and apply the loss on that output (combined with the usual
  max-iteration loss). This penalizes iteration-count-specific behavior and pushes the loop toward a
  *repeatable* step operator / steady state — i.e. it directly attacks "overthinking." Code:
  github.com/aks2203/deep-thinking. **Top candidate for the next milestone.**
- **Path Independence in equilibrium models** — Anil, Pokle et al. 2022, arXiv 2211.09961
  (NeurIPS'22). Shows upward (harder-than-trained) generalization *correlates with* path
  independence (convergence to the same attractor regardless of init/over-unroll); interventions
  that promote it improve extrapolation, those that penalize it hurt. Gives a **measurable
  diagnostic** (path independence on OOD samples ⇒ accuracy) and a regularizer to try.
- **Looped Transformers for Length Generalization** — Fan, Du, Ramchandran, Lee 2024,
  arXiv 2409.15647. Weight-tied looping + an **adaptive step count** length-generalizes on
  algorithmic (n-RASP-L) tasks where required depth grows with input length — the architectural
  cousin of our Task B depth-extrapolation claim; their adaptive-halting tie-in is the alternative
  to progressive loss.
- **Rethinking Deep Thinking (Lipschitz-constrained stable algorithm learning)** — 2024,
  arXiv 2410.23451. Stabilizes the deep-thinking recurrence with Lipschitz constraints to curb
  overthinking — a follow-up lever if progressive loss alone is unstable at our scale.
- **Recurrent-depth latent reasoning ("Scaling up Test-Time Compute…")** — Geiping et al. 2025,
  arXiv 2502.05171 (NeurIPS'25 spotlight). Large-scale evidence that iterating a recurrent block to
  arbitrary test-time depth scales reasoning; the per-token adaptive-compute view. Context for why
  a transferable operator matters; out of our tiny-first scope to replicate, in scope to borrow from.
