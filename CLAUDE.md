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
| C | `nested_converge` | hierarchy probe, **re-imagined** | earn H/L *against the single loop* — substrate BUILT (M17), but the gate FAILS its controls (M18g/i/j): the single-loop "insufficiency" is a sample wall (the single loop SOLVES it at 64k, EM 0.99), not a timescale deficit → **H/L build (M19) NOT earned; Task C re-DEFERRED** (see §9.3) |

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
the final step). Optional learned halting (ACT/PonderNet-style) was a later knob — **now BUILT
(M23): `use_act` + `train_act`/`evaluate_act`; it is faithful and demonstrably adaptive (segments
scale with difficulty) but did NOT unlock the canonical hard-solving win — §11(b) M23-ACT.**

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

## 9. Scope discipline — what's settled, what's earned, what's still out

**The synthetic story is now clear (M0–M15c).** This section used to be a flat "do-not-do"
list gated on a single criterion. The program ran that criterion to ground and it is now
rewritten around what was actually established. Full narratives live in `results/LOG.md`.

### 9.1 The original gate is RETIRED (falsified, not unmet)

The old gate — *"no H/L hierarchy until the single loop beats its control on Task A **and**
Task B"* — is **structurally unsatisfiable** and is withdrawn. M6a's `multi_parity` both-axes
probe (built to be the one task needing depth *and* width at a fixed budget, exactly where a
generalist *should* beat both single-axis controls) produced **zero** loop-beats-both cells and
**falsified** the weaker "never-worst" claim too. A weight-tied generalist judged against
single-axis *specialist* controls at a fixed parameter budget cannot dominate on both axes —
"beats both on A and B" was the wrong success criterion, not a bar the loop narrowly missed. Do
**not** re-run experiments to satisfy it (M5/M6a closed that), and do **not** treat it as the
trigger for Task C.

### 9.2 What the loop actually buys (the result Task C must build on)

The defensible positive finding, fully scoped:

> **Tied recurrence with a JOINT multi-output state buys whole-row COHERENCE on LOCAL-UPDATE
> (CA) HARD multi-output FIXED-POINT targets.**

It decomposes into two mechanisms plus one broad architectural fact (narratives: LOG.md M8–M15c):

- **Leg 1 — joint-state coherence mechanism (deep + local; CLEAN, within-task).** Refining all
  output cells in one shared latent beats refining them in independent latents (`trm` ≫
  `trm_decoupled`, budget/recurrence/supervision matched) — ΔEM up to +0.66, 10/0, *growing*
  with model size (M10/M11), across the whole hard-convergence ECA regime (M12), and it
  **transfers off-CA** to a non-uniform deep+local map (M15). Needs *deep + local* structure;
  null on a dense net (M13) and on a shallow one (M14).
- **Leg 2 — loop-beats-the-shallow-MLP EM edge (needs a UNIFORM local rule).** The loop beats a
  param-matched MLP on whole-row exact-match only where the per-cell update is a single shared
  rule; demonstrated depth-distribution-controlled on rule 13 (+0.21 EM, 10/0, hardness against
  the result), absent on a per-position-mixed rule at identical depth (M15c). EM-only, w≤24.
- **P1 — tying-positive (broadest leg).** The tied loop beats a *fair* (width-matched) untied
  stack on coherence — width-robust (M9), survives off-CA in 3/4 cells (M13/M14). The one
  regime-independent architectural pro-loop fact.

Equally load-bearing, **what the loop does NOT buy** (so Task C isn't designed to chase a ghost):
*not* depth-extrapolation or a transferable step operator (M1/M3b/M7 — over-unrolling decays even
on a convergent target, M8); *not* adaptive test-time compute (M8); *not* a token-accuracy edge at
large w (M9); *not* universal across operator families (ff dominates per-cell-*easy* targets, M11);
*not* a property of hard-convergence fixed points in general — it is CA/local-update specific
(M13/M14); *not* a capacity-independent "beats-both" (ff wins at small model size, M11).

### 9.3 Task C, re-imagined — "earn the hierarchy against the LOOP, not the baselines"

The original Task C was a generic `compositional` hierarchy probe gated on the loop beating the FF
*baselines*. Both halves of that framing are now wrong. The ARC autopsy plus our M0–M2 work
already showed the *loop*, not the H/L hierarchy, is the active ingredient; and §9.2 shows the
loop's value is **joint-state coherence on local fixed-point maps**, not depth or composition. So
Task C (substrate now built, M17) asks the only hierarchy question the evidence leaves genuinely open:

> **Does a two-timescale (H-slow / L-fast) loop buy whole-row coherence that the validated
> single-timescale joint-state loop CANNOT — on a target that is itself a hierarchy of local
> fixed points?**

Concretely, re-imagine Task C as **`nested_converge`** (a two-timescale fixed point), *not* a
generic `compositional` probe:

- **Target.** A **nested fixed point**: an outer local map whose every step is itself the
  converged fixed point of an inner local CA (e.g. inner = a verified converging rule run to
  s_inf over blocks; outer = a second local rule over the inner-converged blocks, run to *its*
  fixed point). Stay in the regime where the loop's mechanism lives — **local + deep + ff-hard**
  — and **rejection-filter to the convergent basin** exactly as `converge`/`mixed_converge` do.
  Difficulty dials: nesting levels, inner-vs-outer convergence depth, block size.
- **The control is now the SINGLE LOOP.** The §4 FF/untied controls still ship, but the
  *decisive* comparison is **H/L two-module loop vs the validated single-timescale `trm`** (does
  the second timescale add coherence?), with `trm_decoupled` (still the joint state?) and a
  depth-matched untied stack (two-timescale *tying* or just more depth?). This is the autopsy's
  "earn the hierarchy" done honestly: the hierarchy must beat the loop we already trust, not a
  baseline the loop already beats.
- **The gate to BUILD it (now satisfiable).** Build Task C only once there is a concrete
  `nested_converge` instance where the **single-timescale loop's coherence plateaus below the
  target** (one joint-state timescale is provably insufficient) **and** the structure is genuinely
  two-timescale. If the single loop already solves the nested target, the hierarchy is unearned —
  report that null and stop. Unlike the retired gate, this is a *within-loop ablation*, not a
  generalist-beats-specialists demand, so it can actually be met or cleanly falsified.
  **This gate FAILS the equal-compute control test (M18g) → the H/L build (M19) is NOT earned; Task C is
  re-DEFERRED.** `make_nested_converge` (inner_rule=13 / outer_rule=79 / block_w=8, w∈{24,32}) is built +
  screened + tested, and M17 *initially* reported the gate MET (single-loop EM 0.56 plateau). But two
  adversarial reviews dismantled that: (1) M17's 0.56 was **undertrained** (curriculum path; plain
  standard-train already reaches 0.689 — M18d); (2) decisively, the gate compared a 4×-compute loop
  against **1×-compute controls**. **M18g (hidden=64) + M18i (hidden=128) re-ran the gate with EVERY arm
  at equal compute (400 epochs, all train_acc≈1.0).** The unified picture: the single-timescale loop is
  **the best arm** (beats ff and untied, and the edge GROWS with capacity — Δ(trm−ff) EM +0.036 @ h64 →
  +0.064 @ h128, the M11 hard-convergence signature), **but it plateaus far below the target at EVERY
  capacity** — EM 0.75 (h64) / 0.79 (h128) ≪ 1.0 — and a param-matched FEEDFORWARD sits just behind it in
  the same ~0.7–0.8 band. The loop's edge is the ordinary **leg-2 coherence edge, ~0.04–0.06 — nowhere
  near the ~0.2 headroom to EM=1.0** the gate needs, and more capacity barely moves the plateau (+0.036 for
  2× hidden). So the loop's gap-to-target is **a capacity/data-bound ceiling the controls share, NOT a
  single-*timescale* deficit** (the §8 trap: the loop trivially degenerates into a deep net; a plain MLP
  hits the same wall). The gate needs a timescale-*specific* insufficiency — the loop *uniquely* stuck
  where a richer single-timescale model is not — and that is absent, so **M19 is not earned**. The
  constructive lever is **data**, not a hierarchy, and M18j makes this airtight: the data sweep EM
  4k→16k→64k is **0.75 → 0.93 → 0.99** for the single loop (ff 0.71 → 0.90 → 0.97) — **at 64k the
  single-timescale loop SOLVES the nested target (0.99 EM)**, a pure sample wall (M5 signature). This
  literally triggers §9.3's own null clause ("if the single loop already solves the nested target, the
  hierarchy is unearned — report that null and stop"). 2× capacity barely helps (M18i, +0.036); data is
  the lever.
  **What DOES survive the equal-compute test (all three §9.2 legs, at honest scope, and they GROW with
  capacity):** leg-1 joint-state (Δ(trm−decoupled) EM +0.110/+0.161, 8/0), P1 tying (Δ(trm−untied) +0.065
  @ w24, 8/0 — at hidden=64 untied is +2.5% over budget so conservative; **M18i confirms P1 budget-clean at
  hidden=128**), and leg-2 (loop>ff EM +0.036 @ w24 h64 → **+0.064 @ h128**, 8/0). All reproduce/strengthen
  with size on the two-timescale family (the M11 pattern) — but none is timescale-*sized* (the loop still
  plateaus at 0.79 ≪ 1.0 at h128). **M18i also KILLS the M17b "P1 reverses at hidden=128" confound: that
  reversal was a 1×-compute artifact — at EQUAL compute the loop beats untied at h128 (+0.056, budget-clean).**
  **Re-gate condition for any future M19:** find a nested instance where the single-timescale loop
  plateaus below the target **and the feedforward/untied controls do NOT share that ceiling at equal
  compute** — i.e. the loop is uniquely stuck where a richer single-timescale model is not. The current
  instance does not show that.

**Proposed reference generator (NOT built — a sketch for the next agent, in the §3 style).** The
two-timescale structure reuses the existing `ca_step` at both levels and the existing
`make_mixed_converge` rejection-filter/depth-tracking boilerplate verbatim; only the relaxation
operator is new. Screen the `(inner_rule, outer_rule)` pair for convergence + ff-hardness exactly
as M8/M12/M15 screened single rules, and add the §5 determinism test before any run.

```python
# --- Task C (PROPOSED, gated — see §9.3): nested / two-timescale fixed point -------------
# A hierarchy of local fixed points. A ROUND = one slow OUTER step then a full INNER relax:
#   - inner (FAST, "L"): each block is its OWN ring of width block_w; iterate inner_rule to a
#     per-block fixed point. ca_step touches only axis -1, so reshaping to (n, n_blocks, block_w)
#     relaxes every block independently as a ring.
#   - outer (SLOW, "H"): one outer_rule step on the FULL ring couples neighbouring blocks.
# Target = the JOINT fixed point of (inner_relax ∘ outer_step). Two timescales by construction;
# local + deep + (screen for) ff-hard; spatially uniform at each level so leg 2 (§9.2) can apply.
# Difficulty dials: n_blocks, block_w, inner depth (rule), #outer rounds to converge.
# WHY: one joint refinement timescale must discover it has to FULLY relax inner blocks between
# every outer coupling — the within-loop insufficiency §9.3's build-gate tests for.
# NOTE (M17, now BUILT): the placeholder rule defaults below (232/232) are NOT validated convergent —
# the screened/locked instance is inner_rule=13, outer_rule=79, block_w=8 (use those; 232/232 may hit
# the non-convergence raise). The real generator lives in src/looptab/data/generators.py.

def _inner_relax(s, n_blocks, block_w, inner_rule, max_inner):
    blk = s.reshape(s.shape[0], n_blocks, block_w)        # each block = its own ring (axis -1)
    for _ in range(max_inner):
        nxt = ca_step(blk, inner_rule)
        if np.array_equal(nxt, blk):
            break
        blk = nxt
    return blk.reshape(s.shape[0], n_blocks * block_w)

def make_nested_converge(n, n_blocks, block_w, task_seed, sample_seed,
                         inner_rule=232, outer_rule=232, distractors=0,
                         T=None, return_trajectory=False, max_rounds=None, max_inner=None):
    w = n_blocks * block_w
    max_rounds = max_rounds or 4 * n_blocks               # outer timescale ~ #blocks
    max_inner  = max_inner  or 4 * block_w                # inner timescale ~ block width

    def round_(s):                                        # one SLOW round: outer step, inner relax
        return _inner_relax(ca_step(s, outer_rule), n_blocks, block_w, inner_rule, max_inner)

    # Rejection-filter to the convergent basin EXACTLY as make_mixed_converge: draw blocks of 2n,
    # iterate round_ to a joint fixed point, keep convergent rows, depth = #rounds, raise loudly if
    # too few converge (a non-converging rule pair). s_inf = the joint fixed point; X = s0 (+ static
    # task_seed distractors). return_trajectory => frames AFTER EACH ROUND (loops ≈ outer rounds) for
    # step-aligned DS. row_rng = sample_seed (rows); fn_rng = task_seed (distractors / any rule pick).
    ...
    return X.astype(np.float32), s_inf.astype(np.int64)   # (n, w) float32 input, (n, w) int64 target
```

The new model piece is the **H/L two-module loop** (L = `_inner_relax`-shaped fast updates to a
fixed point, H = one outer update per L-convergence) would be registered alongside `trm`. That model
**does not exist and is NOT to be built yet (M19 is unearned)** — the equal-compute gate (M18g) shows the
single loop's nested ceiling is shared by a feedforward, so the within-loop insufficiency the build-gate
requires has NOT been demonstrated.

**Task C is RE-DEFERRED (M18g).** The within-loop insufficiency was *not* demonstrated: at equal compute
(all arms saturated, train_acc≈1.0) the single-timescale loop and a param-matched feedforward share the
nested ceiling (Δ(trm−ff) EM +0.036 @ w24, −0.003 @ w32) — a generic capacity/generalization wall, not a
single-timescale deficit. Building the H/L split now would repeat the exact HRM mistake the ARC autopsy
diagnosed (hierarchy without first showing the single loop *uniquely* fails). The re-gate condition: a
nested instance where the loop plateaus below target **and the feedforward/untied controls do NOT share
the ceiling at equal compute**. Until then, do NOT build M19. (The two legs that *did* reproduce on nested
at equal compute — joint-state coherence and tying — are §9.2 results extended to a new family, not a
hierarchy mandate.)

### 9.4 Still out of scope (rationale updated now that the synthetic story is clear)

- **Real / downloaded datasets** (ARC, Sudoku, OpenML, Grinsztajn). The synthetic story is clear,
  so the real-tabular bridge is the *other* legitimate frontier — a deliberate, separately scoped
  step (port the joint-state-coherence finding to a real multi-output tabular target, §4 control
  contract intact), **not** a casual download started mid-stream. **STARTED (M20) — and, properly
  evaluated, the loop does NOT transfer:** a `multilabel` task on real `emotions`/`yeast` (vendored
  network-free, §5). Under micro/macro-F1 + 10-fold CV, joint modeling beats binary-relevance (leg-1) but a
  plain joint MLP gets that too, and every loop-SPECIFIC edge is EM-only and ties under F1 — so the synthetic
  coherence value does not cross to real tabular as a *loop* property (§11(b)/LOG.md "M20 — PROPER
  EVALUATION"). Confirmed on TWO large datasets with opposite coupling (`yeast` co-occurrence, `scene`
  mutual-exclusion). The bridge machinery (F1 metric, K-fold CV) is built + reusable.
  Downloads are confined to a one-time out-of-band fetch script; the *task path* stays network-free +
  content-hash-guarded, so this does not relax §5's "no network in the task path."
- **No RL extension. No large models. No speculative architecture zoo.** Tiny-first, one variable
  at a time (§5) still governs everything, a re-imagined Task C included.

## 10. For agents working in this repo

1. Read this whole file before editing.
2. Validate any change against **Task 0 first, then Task A**.
3. **Always emit the control** alongside the recurrent run; report `Δ` and variance.
4. Write a determinism test for any new generator.
5. Keep diffs scoped to one milestone item.
6. Update §11 when you land something. That section is how the next (context-free) agent
   knows where the repo stands.

## 11. Project status / next milestone

This is the hot-context status page. Full evidence is in `results/LOG.md` (index) and
`results/log/*.md` (per-milestone narratives). Reference notes moved to
`docs/REFERENCES.md`. Keep this section short: only behaviour-changing conclusions and
open/closed work.

### 11.1 Current substrate

- **Tasks:** `linear`, `parity`, `iterated`, `multi_parity`, `converge`, `hopfield`,
  `mixed_converge`, `nested_converge`, `disruption`, real `multilabel`, and `sudoku`.
  Generators live in `src/looptab/data/generators.py`; real data loading lives in
  `src/looptab/data/real.py`; determinism and validity tests live under `tests/`.
- **Models/arms:** `trm`, `ff_matched`, `untied_stack` (ceiling only),
  `untied_matched` (clean tying control), `trm_decoupled`, `trm_mixer`,
  `untied_mixer`, and `untied_mixer_matched`. Registrations are in
  `src/looptab/registry.py`; implementations are in `src/looptab/models/`.
- **Training/eval:** standard training, curriculum/step-aligned DS, progressive loss,
  canonical detached deep supervision, EMA, ACT, paired Δ reports, sign tests,
  `coherence_excess`, multilabel F1/CV, and optional M21 introspection are implemented.
  Entry point: `python -m looptab.run --config <yaml>`.
- **Configs/results:** experiment configs are in `configs/experiments/`; tracked summaries
  are in `results/`; milestone narratives are indexed by `results/LOG.md`.

### 11.2 Behaviour-changing conclusions

1. **Report deltas, not lone scores.** Every recurrent claim must be paired against the
   relevant matched control(s), with seed variance/sign information where applicable.
2. **Use `untied_matched` as the clean §4b control.** `untied_stack` has multiplied
   parameters and is only a labelled ceiling; using it as the main control confounds
   recurrence with capacity.
3. **The original “beat both controls on Task A and Task B” hierarchy gate is retired.**
   M6a showed it is structurally unsatisfiable for the generalist loop against
   single-axis specialist controls. Do not rerun experiments to satisfy it.
4. **Task A parity:** depth helps; weight tying is neutral. The loop and the
   param-matched untied stack solve hard parity where the shallow MLP can fail.
5. **Task B iterated CA:** the loop has shallow-depth, parameter-efficiency wins over a
   fair untied stack, but no reliable depth-extrapolation; at larger T every arm hits a
   learnability wall. Extra test-time loops do not recover longer computations.
6. **Deep supervision is protocol-dependent.** Final-state per-step DS is usually neutral
   to negative; step-aligned DS helps at short horizons but flips negative at longer ones.
   Canonical detached-carry DS gains in M18 were mostly extra optimization, not a distinct
   mechanism.
7. **The strongest synthetic positive is joint-state / cross-output structure, not generic
   tabular recurrence.** Flat TRM wins were best explained by whole-row coherence on
   hard fixed-point regimes, and later mixer work showed cross-cell mixing architecture —
   not weight-tied recurrence alone — carries the hard-solving capability. Tying mainly
   buys parameter efficiency.
8. **Real multi-label classification is negative for the loop thesis.** Proper F1 +
   10-fold CV on yeast and scene finds robust joint-output modeling benefits, but they are
   not loop-specific: a joint MLP gets them too.
9. **Task C / H-L hierarchy remains unearned.** `nested_converge` is built, but equal-compute
   controls and larger data show the single-loop shortfall is a sample/generalization wall,
   not a timescale deficit. Do **not** build M19 until a new gate shows a timescale-specific
   single-loop insufficiency.
10. **Introspection reframes loop wins as non-contractive paths.** M21 found no latent fixed
    point even where the loop wins; over-unroll collapses. A future `trm_stable` arm would
    be a new hypothesis and must be judged against both extrapolation and coherence costs.
11. **Mixer arc:** `trm_mixer`/untied mixer results show cross-cell dependency is the useful
    axis (`converge`, `hopfield`, `disruption`), while exchangeable outputs (`multi_parity`)
    do not benefit. Mixing-matched controls attribute the capability to architecture, not
    recurrence.

### 11.3 Open work

- **Only build H/L (M19) after a new Task-C gate passes**: single-timescale loop plateaus
  below target; ff/untied controls do not share the ceiling at equal compute; and more data
  does not close the gap.
- **Possible real-tabular frontier:** coupled multi-target regression. It needs regression
  heads/losses/metrics and the same joint-vs-decoupled-vs-control discipline.
- **Optional mechanism frontier:** `trm_stable` (Jacobian/Lipschitz/path-independence
  regularization) if the goal is to test contractive dynamics explicitly.
- **Low-priority mixer cleanup:** `mixed_converge` mixer retest or a stricter single-pass
  feedforward-mixer control; neither is needed to interpret the current evidence.

### 11.4 Closed levers — do not redo casually

Depth-extrapolation via progressive loss/path-independence (M7/M8), adaptive compute as a
canonical hard-solving unlock (M23 ACT), M4 sample-wall lifting (M5), both-axes gate rescue
(M6a), decoupled-head mechanism isolation (M10), size axis for the synthetic coherence
regime (M11), ECA orbit generality (M11/M12), dense/local threshold-net transfer (M13/M14),
uniform-vs-depth controls (M15–M15c), and real multi-label classification (M20).

## 12. References

See `docs/REFERENCES.md`.
