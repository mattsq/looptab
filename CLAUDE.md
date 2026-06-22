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
  per-instance depth varies, target is a genuine fixed point). Generators in
  `src/looptab/data/generators.py`, determinism-tested in `tests/test_generators.py`;
  `make_trajectory_dataset` dispatches iterated/converge. Task C (hierarchy) is **gated, unbuilt** (§9).
- **Models/arms:** `trm` (weight-tied refinement loop, optional per-step readouts),
  `ff_matched` (§4a param-matched shallow MLP), `untied_stack` (§4b untied, ~`n_steps`×
  params — a confounded ceiling, NOT param-matched), `untied_matched` (§4b untied,
  width-shrunk to the loop's budget — the *clean* tying control). In
  `src/looptab/models/{trm,controls}.py`, registered in `src/looptab/registry.py`.
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
  m8_converge_adaptive, m8b_converge_grid, m8c_converge_fair, m9_converge_width).
- **Metrics:** `accuracy` / `exact_match` / `majority_baseline`, the single-pass `evaluate`, and
  (M9) `coherence_excess = EM − token_acc**w` (whole-row coherence beyond what independent per-cell
  errors would give; >0 = errors clustered) with a `mean_wrong_per_row` companion — all in
  `src/looptab/eval/metrics.py`; paired Δ with variance + sign test is `delta_report`.
- **Tests:** `tests/` — generator determinism, model shapes/param-ratios, runner
  determinism/independence, coherence-metric math. Run `uv run --extra dev pytest -q` (98 tests);
  lint `uv run ruff check`.

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
but narrow (multi-output fixed-point, EM/coherence, w≤24), and NOT adaptive-compute.

**No milestone is currently in flight.** Open threads, in rough priority:
- **A DECISION (now the highest-value live question): relax the literal §9 gate wording.** M6a showed
  "beats both on A and B" is structurally unsatisfiable; **M8+M9 supply the *useful* reframing with
  evidence**: the loop's value is **whole-row/structural coherence on multi-output fixed-point targets**
  (width-robust tying-over-untied; loop-beats-both at matched token-acc for w≤24), NOT single-axis
  dominance and NOT token-acc. Rewrite §9's gate around this; do **NOT** build Task C.
- **M10 — decoupled-head ablation (the deepest remaining mechanism question).** M9 confirmed tying buys
  whole-row coherence at matched token-acc but did NOT isolate *why*: is it the *joint* multi-output
  readout (all `w` cells share one latent/answer state, refined together) vs per-cell-independent heads?
  Build a TRM variant with decoupled per-cell refinement and compare on `converge` @ w≈24 (the coherence
  peak). Needs new model code — kept out of M9's single-knob scope.
- **Lower-value extensions of M9:** more operator families / a model-size axis (M8c had 3 rules at
  w∈{24,32}; M9 has 1 rule × 5 widths — neither covers rule×width×size jointly); (deferred) the strictly-
  budget-clean `untied` fix — currently over-budget at w=24/32, which only *handicaps* the control, so the
  tying-positive is already conservative.
- **Closed levers (do not redo):** depth-extrapolation via progressive loss / path-independence (M7/M8 —
  decay is intrinsic, not convergence-related); adaptive compute on a fixed-point target (M8 — decays);
  "lift the M4 sample wall" (M5); "re-judge via a both-axes task" (M6a). A bigger-model probe of d=80,k=5
  must scale the budget for *all* arms (M5).

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
