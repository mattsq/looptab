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

- **Tasks:** Task 0 `linear` (smoke), Task A `parity`, Task B `iterated` (CA) — generators
  in `src/looptab/data/generators.py`, determinism-tested in `tests/test_generators.py`.
  Task C (hierarchy) is **gated, unbuilt** (§9).
- **Models/arms:** `trm` (weight-tied refinement loop, optional per-step readouts),
  `ff_matched` (§4a param-matched shallow MLP), `untied_stack` (§4b untied, ~`n_steps`×
  params — a confounded ceiling, NOT param-matched), `untied_matched` (§4b untied,
  width-shrunk to the loop's budget — the *clean* tying control). In
  `src/looptab/models/{trm,controls}.py`, registered in `src/looptab/registry.py`.
- **Train/eval:** deep supervision is a **per-arm weight** (`src/looptab/train/loop.py`),
  not a global flag. Metrics `accuracy` / `exact_match` / `majority_baseline`
  (`src/looptab/eval/metrics.py`). Paired Δ with variance is `delta_report`.
- **Runner:** one entry point `python -m looptab.run --config <yaml>`. Supports a 1-D
  `sweep`, an N-D `grid`, and a depth-`extrapolation` harness (`grid` and `extrapolation`
  are mutually exclusive). Emits per-arm curve CSV, per-config Δ CSV, JSON record (config +
  metrics + seed + git SHA), and PNGs if matplotlib present (`src/looptab/run.py`).
- **Configs:** `configs/experiments/` (m0…m2-confirm, m3a/m3b, m4_parity_grid,
  m5_parity_wall_n16k/n64k).
- **Tests:** `tests/` — generator determinism, model shapes/param-ratios, runner
  determinism/independence. Run `uv run pytest -q`; lint `uv run ruff check`.

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
- **No transferable step operator; the loop does NOT extrapolate in depth (M1 + M3b).**
  Over-unrolling R′>R decays to baseline, and OOD depth T>T_train collapses to baseline for
  every arm. M3b applied the two named levers (T-curriculum + step-aligned DS) and the OOD
  collapse (T≥12) STILL holds — a stronger, cleaner null than M1's. Within the curriculum
  there's only weak compositional signal (R′=T tracks for T≤8, tops out ~0.58).
- Each leg still rests on few configs: Task A now multi-`d`/multi-`k` (M4) and the d≥40 wall has
  been swept over `n_train` (M5 — it is sample-bound and lifts to all-solve, except d=80,k=5 which
  is capacity-bound); Task B depth swept (M3a) but unlearnable past T=4 one-shot; M3b on one rule
  (30) / one width (9).

### (c) Next milestone

**M3a, M3b, M4, and M5 are all DONE** (full narratives in LOG.md). M3a falsified "loop edge grows
with depth" (it vanishes by T≥8; deep CA is unlearnable one-shot for all arms). M3b applied
the T-curriculum + step-aligned DS levers: step-aligned DS is a *real* short-horizon win (DS was
mis-specified, not inert) but did **not** yield a transferable operator. M4 replicated the Task A
parity leg across d∈{20,40,80} × k∈{3,4,5}: the loop's edge over `ff_matched` is **robust and
grows with k** but is purely **depth** (ties `untied_matched` in all 9 cells), loop-beats-both in
**0** cells. M5 swept `n_train` (4k→16k→64k) on M4's d≥40 wall: it is **sample-bound and lifts to
all-arms-solve with NO separation** — M4's d=80,k=3 hint was `ff` sample-starvation and the 16k
d=80,k=4 loop hint was a transient sample-efficiency ordering, both erased by more data; the lone
non-lifting cell (d=80,k=5) is a **capacity** wall (needs a bigger model, not more data).

**No milestone is currently in flight.** The §9 gate is still unmet (no task where the loop
beats *both* controls) and M5 confirmed that raising `n_train` dissolves the apparent Task A hints
rather than turning them into a loop-beats-both cell. Open levers for whoever picks this up next, in
rough priority: (i) **re-judge the §9 gate wording itself** — now the **highest-value** question:
both tasks say the loop is robust-not-dominant, and M4/M5 strengthen the suspicion that "beats both
single-axis controls" is unsatisfiable by a generalist vs single-axis specialists (decide whether
to relax it — e.g. to "never-worst across both tasks" or "beats both on one task" — do NOT build
Task C on the current evidence); (ii) **broaden M3b** to more rules/widths and try a *fixed-point or
halting* objective (PonderNet/ACT) against the depth-extrapolation null; (iii) if a bigger-model
probe of d=80,k=5 is ever wanted, it must scale the budget for *all* arms together (confound guard).
The "lift the M4 sample wall" lever is now **closed** (M5). Earn the hierarchy later, not now.

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
