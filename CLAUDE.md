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

**⭐ DEFAULT ARCHITECTURE = the CROSS-CELL MIXER (`trm_mixer`), NOT the flat `trm`.** The single most
load-bearing finding of the whole program (M23 Sudoku → M24 rings → M26 forecasting): a weight-tied loop
whose update is a *flat MLP over the concatenated grid* is **≈ a feedforward net** — it cannot let cells
communicate, so "does iterative refinement help?" was systematically confounded with "does the update
operator match the task structure?" for M0–M22. The thing that makes TRM *work* — constraint propagation,
the hard-solving win, the transfer to real forecasting — **is the cross-cell token-mixing update**, not the
loop per se. So **start every new recurrent experiment from `trm_mixer`; use the flat `trm` only as the
COMPARISON control** (the §4b "is it the mixing operator or just recurrence?" ablation — flat `trm` ≈ ff on
structured tasks, so `Δ(trm_mixer − trm_flat)` is the operator's contribution). Do NOT default to flat
`trm` as a "simplest starting point" — a null measured on it is a null about a feedforward-equivalent, not
about refinement (the §8 trap, one axis deeper). **Caveats that bound the default (don't misapply it):**
(1) `trm_mixer` is **multi-output only** and requires **`in_features % out_features == 0`** — so distractor
columns are unsupported (use `distractors: 0` or `pad_to_label_multiple`); single-output tasks (e.g.
scalar-`y` parity) have no mixer, use flat `trm` there. (2) The mixer helps only where outputs are
**cross-cell COUPLED with a shared input/output cell topology** (rings/grids/graphs, forecasting) — it is
inert on **exchangeable** features (`multi_parity`, M24d) and on the **naive multi-label reshape** (no
input↔output correspondence, M25), where a plain MLP is best; a mixer null there is expected, not a
refinement null. (3) 3-D matmul ⇒ pin `num_threads=1` for bit-repro. When in doubt, run `trm_mixer` AND
flat `trm` AND `ff_matched` together (the M24 lean triple) so the operator's contribution is always visible.

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
4. **Default the recurrent arm to `trm_mixer`, use flat `trm` only as the comparison control** (§4 ⭐):
   the flat update is feedforward-equivalent, so a finding measured on it is not a finding about
   refinement. Mixer needs multi-output + `in_features % out_features == 0` (distractors=0 / padding);
   it is inert on exchangeable-feature tasks — see the §4 caveats before applying it.
5. Write a determinism test for any new generator.
6. Keep diffs scoped to one milestone item.
7. Update §11 when you land something. That section is how the next (context-free) agent
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
  local + temporally-uniform but spatially non-uniform; `mixed_ca_step` is the per-position step),
  and `nested_converge` (**M17 Task C substrate**: a TWO-TIMESCALE fixed point — a ROUND = one SLOW
  outer full-ring `ca_step` then a full FAST inner relax (`_inner_relax` settles each block on its
  OWN ring); target = the JOINT fixed point of `round_ = inner_relax ∘ outer_step`, basin-rejection-
  filtered like `mixed_converge` — **and the acceptance check requires INNER-STATIONARITY
  (`_inner_stationary`), not just `round_(s)==s`** (a PR-review fix: a cycling inner rule whose period
  divides `max_inner` can make the round map periodic at a non-fixed state; the locked inner=13/outer=79
  is genuinely convergent so this is bit-identical — golden hash unchanged — and only hardens the filter
  for arbitrary rule pairs); locked instance inner_rule=13 / outer_rule=79 / block_w=8,
  n_blocks∈{3,4} ⇒ w∈{24,32}; built for the §9.3 single-loop-insufficiency gate).
  Generators in `src/looptab/data/generators.py`, determinism-tested in
  `tests/test_generators.py`; `make_trajectory_dataset` dispatches iterated/converge/hopfield/
  mixed_converge/nested_converge/disruption. **DISRUPTION (M22, user-requested synthetic from an
  airline ops spec):** `disruption` (one row = one disruption COMPONENT of `w` coupled flights;
  X = per-flight features known at t0 [severe_0 + fixed structural cols], y = the settled
  severe-outcome vector — a JOINT multi-output fixed point. A structured integer THRESHOLD net
  [`_build_disruption_weights`: symmetric `W = w_rot·rotation-chain adjacency [LOCAL] + w_bank·bank
  cliques [NON-LOCAL]`] iterated by `_threshold_step` with PSD self-coupling `gamma` ⇒ convergent,
  bit-exact; the graph is fixed by `task_seed`, only `severe_0` varies per row. Locked instance
  w=24/n_banks=4/w_rot=6/w_bank=3/γ=14 [the MINIMAL-PSD γ for task_seed=42] — screened genuinely
  ff-HARD: LINEAR-baseline EM ~0.23 ≪ ff EM ~0.34, copy-severe_0 frac 0.82. TWO design lessons
  [the 2nd from the M22 adversarial review]: a faithful MONOTONE cascade is ff-EASY/vacuous
  [OR-reachability is linearly separable], AND an over-damped γ [margin>0] drives the threshold net
  toward IDENTITY [near-affine; a linear model solves it] — so pin the MINIMAL-PSD γ and screen
  against a LINEAR baseline, not just cell-mean balance). Task C **substrate is BUILT (M17), but its build-gate FAILS the
  equal-compute control test (M18g — a feedforward shares the single-loop ceiling), so the H/L MODEL
  (M19) is NOT earned and Task C is re-DEFERRED** (§9.3). **REAL-TABULAR (M20, §9.4 bridge):**
  `multilabel` (a real downloaded multi-label classification task — binary-per-label outputs, so
  EM = subset accuracy = leg-1's whole-row coherence metric, and `trm_decoupled` = binary-relevance).
  Vendored as numpy `.npz` under `datasets/` (built once by `scratchpad/fetch_multilabel.py` from
  OpenML; the *task path is network-free + content-sha256-guarded*, §5); loader
  `src/looptab/data/real.py` (`make_multilabel_splits`: disjoint seed-keyed train/test from a finite
  pool, features z-scored on TRAIN stats only), dispatched via a `task=="multilabel"` branch in
  `make_splits`, determinism-tested in `tests/test_real.py`. Locked datasets: `emotions` (593×72, 6
  labels), `yeast` (2417×103, 14 labels), `scene` (2407×294, 6 labels, near-mutually-exclusive).
  **Eval (M20-review fix):** `multilabel_f1` (micro+macro, the honest co-headline to EM) and **K-fold CV**
  (`n_folds`/`cv_seed` in `make_multilabel_splits` — disjoint test folds, so the paired sign test is valid;
  the legacy random-split mode overlaps ~0.30 and suppresses the sign test). Configs use 10-fold CV.
  **M25 adds `pad_to_label_multiple`** (additive, off-by-default ⇒ M20 bit-identical): right-pads X with
  constant-zero columns so `d % L == 0`, needed only by the cell-mixing arm `trm_mixer`
  (`in_features % out_features == 0`) — e.g. yeast 103→112; scene 294/6 and emotions 72/6 need no pad.
  **FORECASTING (M26, the real-tabular REGRESSION form):** `etth1` (7 vars) / `weather` (21 vars),
  multivariate time-series forecasting = multi-target regression with COUPLED targets. Vendored
  `datasets/{etth1,weather}.npz` (content-sha256-guarded, network-free; built by
  `scratchpad/fetch_forecast.py`), loader `make_forecast_splits` (`real.py`): one window = M variables'
  `lookback`-step history → `horizon`-step future, arranged as VARIABLE-CELLS (cell i = variable i's
  lookback, so `in_features = M·lookback` is mixer-divisible and cell i is variable i on input AND output
  — the SHARED topology M25's multi-label lacked). Split = **EXPANDING-WINDOW (rolling-origin) backtest**
  (disjoint chronological test blocks, seed=block, train = its own past prefix minus a purge gap; the
  time-series analog of K-fold — but NESTED train sets, so sign-test p is *indicative*, Dietterich 1998),
  per-variable z-scored on train-only stats. `objective: regression` ⇒ MSE loss + MSE/MAE/R² eval +
  persistence baseline (all additive, off-by-default ⇒ M0–M25 bit-identical). Both `etth1`/`weather` task
  names dispatch to `make_forecast_splits` (the `dataset` param picks the series); determinism +
  no-leakage (window-purge AND standardization) tested for both.
  **SUDOKU (M23, the canonical-TRM positive-control TRIPWIRE):** `sudoku` (one row = one unique-solution
  puzzle; X = one-hot per cell over {blank,1..size}, y = the solved grid as `(size*size,)` classes
  `0..size-1` — a multi-output FIXED POINT, shape-compatible with converge/disruption). `make_sudoku`
  shuffles a valid grid (`_sudoku_full_grid`) and digs it to EXACTLY `n_givens` clues while a
  deterministic MRV bitmask backtracking solver (`_count_sudoku_solutions`) keeps the solution UNIQUE
  (`_dig_sudoku`), rejection-redrawing un-diggable grids; box geom 9→3×3, 6→2×3, 4→2×2; `task_seed`
  inert except `distractors`. **The ONE multi-class task** — it forced `run.py` to infer
  `num_classes = max(2, y.max()+1)` instead of the hardcoded 2 (exactly 2 for every binary M0–M22 task ⇒
  bit-identical). Determinism/validity/uniqueness-tested in `tests/test_generators.py` (golden hash pinned).
- **Models/arms:** (**⭐ DEFAULT recurrent arm = `trm_mixer`; flat `trm` is the COMPARISON control, not
  the starting point — §4 / §10.4.**) `trm` (weight-tied FLAT refinement loop, optional per-step readouts
  — feedforward-equivalent on structured tasks, so use it as the "is it the mixing operator?" contrast),
  `ff_matched` (§4a param-matched shallow MLP), `untied_stack` (§4b untied, ~`n_steps`×
  params — a confounded ceiling, NOT param-matched), `untied_matched` (§4b untied,
  width-shrunk to the loop's budget — the *clean* tying control), and `trm_decoupled` (M10
  mechanism ablation: the loop with **per-cell** refinement — each output cell has its own
  latent slice and sees only its own answer, severing the joint multi-output state; budget-
  matched, multi-output only), and `trm_mixer` (**M23 re-test — the loop with a CROSS-CELL MIXING
  update: keeps the grid as per-cell tokens and applies an MLP-mixer over the size² cells (token-mixing
  = the constraint-propagation operator the flat `trm` LACKS), then a per-cell channel MLP; multi-output
  only, budget-matchable, `token_hidden` knob**), and (**M24e — the B1 mixing-matched §4b controls**)
  `untied_mixer` (the `trm_mixer` block stacked `n_steps`× WITHOUT weight tying — ~7× params, a labelled
  ceiling) and `untied_mixer_matched` (that stack width-shrunk to the tied mixer's budget — the CLEAN tying
  control; both reuse a standalone `_MixerBlock` = one `TRMMixer` step, so `TRMMixer` is untouched and M24
  results stay bit-identical). In `src/looptab/models/{trm,controls,decoupled,mixer}.py`,
  registered in `src/looptab/registry.py`. **M18 TRM-faithful knobs (all OFF by default = bit-identical to
  pre-M18; `trm` only):** `use_rmsnorm` (RMSNorm on the latent each update), `n_latent` (z-updates
  per answer update; 1 = original 1:1), plus the training-side `n_sup` (detached deep-supervision
  passes) and `ema_decay` — a `trm_faithful` arm stacks all four. **Determinism exception (M11):** unlike every 2-D arm, `trm_decoupled`'s
  3-D batched matmul `(B,w,m)` has thread/BLAS-order-sensitive reductions — it is reproducible only at a
  fixed `num_threads` (committed runs use 1) and its EM does NOT reproduce bit-for-bit across environments
  (M10 vs M11 differ ~±0.015; the effect sizes dwarf this). The "bit-identical" guarantees below cover the
  2-D arms only.
- **Train/eval:** deep supervision is a **per-arm weight** (`src/looptab/train/loop.py`),
  not a global flag. **`train` takes `loss_type` ("ce" default / "mse" for M26 regression; "ce" ⇒
  bit-identical).** Six training routines: `train` (standard), `train_curriculum` (M3b
  depth-curriculum + step-aligned DS), `train_progressive` (M7 Deep Thinking progressive
  loss: detach `(T−k)` steps, gradient on `k`, modes `progressive_final`/`progressive_step`),
  `train_deep_supervision` (M18 — canonical TRM/HRM deep supervision: `n_sup` supervised passes
  carrying `(z,a)` across them with the carry **detached** between passes; the autopsy's active
  ingredient, distinct from the per-step-readout DS above; runs on `trm`/`trm_decoupled` AND — since
  M29 — `trm_mixer`, all of which expose the init_state/return_state API), and `train_act` (**M23 — ACT adaptive
  computation, the §4/§12 unbuilt TRM ingredient, now BUILT**: `train_deep_supervision` + a learned
  halt head trained by BCE to predict per-example exact-match; `evaluate_act`/`act_predict` do
  per-example adaptive-segment inference and report `avg_segments`. Enabled per-arm via `use_act`
  (TRM only; `n_sup` = max segments); OFF by default ⇒ bit-identical), and `train_stable` (**M27 —
  the M21 contractive-loop lever, now BUILT**: standard train + a penalty on the one-step latent map
  `F(z)=update(cat[X,z,readout(z)])` — a grad-enabled `torch.func.jvp` Hutchinson `jac_reg_weight·
  mean‖Jv‖²` [drives ρ↓, DEQ Bai 2021] + optional `fixed_point_weight·‖Δz‖/‖z‖` [Anil 2022]. TRM-only,
  param-identical to `trm`; per-arm `jac_reg_weight`/`fixed_point_weight`/`n_reg_steps`, all default 0
  ⇒ OFF ⇒ bit-identical. Same jvp float-reduction determinism caveat as `trm_decoupled` — bit-repro at
  pinned `num_threads=1` only). `train`/`train_deep_supervision`
  take an optional `ema_decay` (M18 ingredient 2) folded into the weights for eval.
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
  m15_{mixed_screen,mixed_converge,uniform_anchor}, m15b_uniform_matched{,_screen},
  m15b_depth_matched, m17_nested_converge_{smoke,gate}, m17b_nested_capacity,
  m18{a,b,c}_faithful_{depthwall,converge,ablation}, m18d_faithful_nested,
  m18e_compute_matched, m18f_epochs_matched, m18g_nested_equalcompute, m18h_nested_data16k,
  m18i_nested_equalcompute_h128, m18j_nested_data64k, m21_introspection_{converge,iterated},
  m22_disruption_base, m22_size_{small,base,large},
  m23_sudoku_{screen,base,scaleup_screen,scaleup_base,scaleup_sig,segments_pretest,act_sweep,
  mixer_sweep,mixer_lean}, m24_mixer_{converge,iterated}, m24b_converge_ablation,
  m24c_hopfield_mixer, m24d_multiparity_mixer, m24e_converge_mixer_untied,
  m24f_disruption_{mixer_w24,mixer_w32,ablation_w24,ablation_w32,untied_w24,untied_w32},
  m25_mixer_multilabel_{yeast,scene,emotions}, m26_{etth1,weather}_forecast, m26_etth1_smoke,
  m29a_ds_mixer_converge{,_4xep}, m29b_ds_mixer_nested{,_4xep}, m29c_ds_mixer_iterated{,_4xep}).
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
  `src/looptab/eval/metrics.py`; paired Δ with variance + sign test is `delta_report`. **CEILING-TIE
  GUARD (M29c):** `sign_test` takes an `eps` (default 0 ⇒ bit-identical classic test) that treats
  |Δ|≤eps as a practical tie; `delta_report` additionally emits `n_near_tie` + a `sign_test_robust`
  at `near_tie_eps=1e-3` (and a `sign_p_robust` CSV column + a `⚠ near-ties` console flag when a
  raw-significant sign test collapses under it) — so an 8/0 riding one-flipped-cell ties on
  near-saturated arms is caught (the M29c over-claim: raw 8/0 p=.008 → robust 3/0 p=.25). Per-seed
  EM/coherence are now persisted (`exact_match_per_seed` etc.) so those sign-counts are re-derivable
  from the record, like accuracy. **Regression
  (M26):** `evaluate_regression` (MSE/MAE/R² on standardized targets; no argmax) + `persistence_baseline_mse`
  (naive last-value forecast) — gated by `objective: regression`, and MSE the honest headline (lower is
  better ⇒ a NEGATIVE Δ favours the first arm, opposite of accuracy).
- **Introspection (M21) — measurement-only latent/weight diagnostics:** `src/looptab/eval/
  introspection.py` (`run_introspection` dispatcher) reads a TRAINED arm and emits, per arm:
  Jacobian spectral radius ρ + operator norm (autograd JVP/VJP power iteration), latent residual
  ‖Δz‖/‖z‖ trajectory + per-step readout acc/EM out to an over-unroll horizon, path-independence /
  asymptotic alignment from random z0 inits (Anil 2022), and effective-rank / participation-ratio +
  per-Linear spectral norms / Lipschitz product. Recurrent arms (`trm`/`trm_decoupled`) get all
  three families; controls get representation + weight only. Gated by an OFF-by-default
  `DiagnosticsConfig` on `ExperimentConfig`; when enabled the runner writes a side-car
  `*_diagnostics.csv`. Touches **no** model code (rides the M7 `init_state`/`return_state` API +
  forward hooks), so every committed result is bit-identical. Descriptors, NOT Δs — they generate
  refinement hypotheses (§8), they do not clear a gate. **Determinism caveat:** unlike argmax
  accuracy, the diagnostics involve float-reduction-order-sensitive ops (`jvp`/`vjp`, SVD,
  `matrix_norm`), so they reproduce bit-for-bit (incl. serial-vs-`parallel_workers`) only with CPU
  threads PINNED (`num_threads=1`, the committed default); a `num_threads: null` scale-up would make
  the diagnostic *numbers* thread-count-dependent (same class of caveat as `trm_decoupled`'s matmul).
- **Tests:** `tests/` — generator determinism, model shapes/param-ratios (incl. the M10
  decoupled no-cross-cell-leakage invariant), runner determinism/independence, coherence-metric
  math, (M13) `make_hopfield` determinism/fixed-point/balance, (M17) `make_nested_converge`
  determinism/golden-hash/joint-fixed-point/two-timescale/trajectory-by-round, and (M18) the
  TRM-faithful knobs (bit-identity-when-off, `train_deep_supervision` + EMA determinism), and (M21)
  the introspection layer (`test_introspection.py`: spectral-radius / effective-rank known-answer
  sanity, same-seed determinism incl. `trm_decoupled`, F(z) n_latent faithfulness, right families per
  arm), and (M27) the contractive `train_stable` arm (`test_stable.py`: penalty-lowers-ρ, determinism,
  accuracy-preserved), and (M29) the detached-carry DS routine on the mixer (`test_ds_mixer.py`:
  `train_deep_supervision` runs on `trm_mixer`, determinism at pinned threads, carry-flag wired,
  run_point dispatch, non-loop guard), and (M29c) the ceiling-tie sign-test guard + per-seed-EM
  persistence (`test_training.py`/`test_run.py`: eps reclassifies near-ties, robust sign test collapses
  the m29c 8/0, `exact_match_per_seed` persisted). Run `uv run --extra dev pytest -q` (278 tests); lint `uv run ruff check`.
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
- **The M8–M12 result decomposes into TWO legs (M15 — partially resolves the M14 confound; leg 1 CLEAN,
  leg 2 SUGGESTIVE-but-confounded).** Built `mixed_converge` (a per-position MIXED CA: each cell its own
  radius-1 rule from orbit1, iterated to a fixed point, rejection-filtered to the convergent basin) —
  **deep + local + ff-hard but spatially NON-uniform (not a CA)** — and contrasted it with the uniform
  `converge` rule-78 anchor. **Task-matching caveat (adversarial review): the two tasks are NOT
  single-variable** — beyond translation-invariance the mixed task is per-row *harder* (ff EM 0.255/0.042
  vs 0.311/0.121 @ w24/w32) and *shallower-tailed* (max depth 6, 0% rows depth>6, vs uniform max 16,
  ~10–13% depth>6; mixed target fully fixed at T=6). **(1) [CLEAN, within-task] The joint-state coherence
  mechanism (joint refinement ≫ per-cell, M10) is DEEP+LOCAL, NOT translation-invariance — it TRANSFERS
  to the non-uniform mixed-CA:** trainability-clean Δ(stepDS−decoupled_stepDS) EM **+0.206 (10/0, p=.002)
  @ w=24** (+0.053 @ w=32), decoupled BELOW ff (−0.237, 0/10) — the M10/ECA signature on a non-CA. Being a
  within-task arm contrast, it is immune to the hardness/depth mismatch. With M13 (dense ⇒ null) and M14
  (shallow ⇒ null), the mechanism needs **local + deep**, uniform or not. **(2) [SUGGESTIVE, cross-task,
  CONFOUNDED] Loop-beats-the-shallow-MLP (the loop-beats-both headline, M8/M9) does NOT reproduce without
  the uniform rule:** at w=24 the loop beats ff on EM for uniform-78 (**+0.133, 10/0, p=.002**) but does
  NOT beat ff on the mixed version (**−0.028, 3/6, p=.51 — point estimate favours ff**). Consistent with
  uniformity being required, but the mixed/uniform pair is confounded by hardness+depth-tail, so this
  single cross-task Δ does NOT isolate translation-invariance; it is w≤24 only (both tie at w=32).
  **[M15b — leg 2 NOW STRONGLY SUPPORTED.]** **(3) P1 survives on both** (conservative; untied over-budget).
  **Net: leg 1 (deep+local joint-state mechanism, transfers off-CA) is established; leg 2 confirmed by M15b.**
- **[M15b] Leg 2 STRONGLY SUPPORTED (not fully isolated) — loop-beats-the-MLP needs a UNIFORM local rule.**
  Added an `accept_max_depth` cap to `make_mixed_converge` (additive; `None` = bit-identical to before, golden-
  hash + additivity tested) and built a **max-depth-matched uniform control**: a uniform single-rule CA
  (`rule_set=[r]`) through the same rejection-filter pipeline, capped to depth ≤6 to match the mixed task's
  depth-*tail* (max 6, target fully fixed at T=6). **Result (w=24, EM, 10 seeds):** uniform rules 78 and 13
  → loop **beats ff +0.090 / +0.175 (both 9/1, p=.021)**; the non-uniform mixed task (same max-depth) → loop
  **ties (−0.028, ns)**. The mixed task is *harder* (ff EM 0.255 vs 0.36), which by M11 should *favour* the
  loop, yet the loop wins only on uniform — so hardness predicts the opposite and can't explain it. Edge
  vanishes at w=32 for all (M9 w≤24). Leg 1 stays large+significant on both uniform (+0.31) and mixed
  (+0.206) — orthogonal. **Why NOT "isolated" (2nd adversarial review):** (i) depth is only MAX-matched —
  uniform controls are ~1 step *deeper* on average (mean 3.9/3.6 vs mixed 2.9), a residual that is NOT
  conservative (deeper ⇒ wider light-cone, could favour the loop); (ii) "uniform vs non-uniform" is
  definitionally entangled with rule-cardinality (1 truth table vs 4) — a non-uniform local rule *must* use
  ≥2 tables, so this can never be separated; (iii) EM-only (loop beats ff on token-acc on none, all ns);
  (iv) two rules × one width at the 9/1 significance floor; the 3-seed screen is directional (all 4 rules
  positive) but none significant. So leg 2 = "the loop-beats-MLP EM edge tracks rule-uniformity at matched
  max-depth and against the hardness gradient," strongly supported, with the central-depth residual the one
  un-eliminated alternative. (Code/determinism verified clean both reviews.)
- **[M15c] Central-depth residual CLOSED — leg 2 survives depth-control on rule 13; rule 78 was partly
  depth.** Added a `depth_profile` stratified-subsample to `make_mixed_converge` and ran mixed/uniform-78/
  uniform-13 all subsampled to an IDENTICAL depth histogram (intersection profile, mean depth 3.40 for all,
  verified bit-for-bit). At depth held fixed: uniform rule 13 → loop beats ff **+0.210 EM (10/0, p=.002)**
  while the non-uniform mixed task **ties** (−0.005, ns) — and mixed is *harder* (ff EM 0.204 vs 0.353), so
  hardness runs against the result; depth can no longer explain it. **Leg 2 confirmed depth-controlled (the
  project's cleanest leg-2 cell).** But uniform rule 78's M15b edge (+0.090) **drops to +0.032 (ns)** once
  depth-distribution-matched (and rule 78 becomes ff-easy, EM 0.44 — no-room/inconclusive), so M15b's
  two-rule 9/1 overstated: under full depth control leg 2 rests on **rule 13 (robust)**, rule 78
  inconclusive. Leg 1 (joint-state) holds at matched depth in all three cells (+0.19/+0.29/+0.36, 10/0,
  decoupled < ff) and P1 survives all (10/0, conservative). Only the definitional uniformity↔rule-cardinality
  (1 vs 4 truth tables) caveat remains un-removable.
- Each leg still rests on few configs: Task A now multi-`d`/multi-`k` (M4) and the d≥40 wall has
  been swept over `n_train` (M5 — it is sample-bound and lifts to all-solve, except d=80,k=5 which
  is capacity-bound); Task B depth swept (M3a) but unlearnable past T=4 one-shot; M3b on one rule
  (30) / one width (9).
- **The §9.3 Task C build-gate FAILS the equal-compute control test — the H/L build (M19) is NOT earned
  (M17 built it, M18g killed the verdict).** M17 built `make_nested_converge` and reported the gate MET:
  the single-timescale `trm` plateaus at EM 0.56 (w24) / 0.37 (w32) ≪ 1.0, capacity-robust (hidden 64→128
  only +0.03, M17b). Two adversarial reviews then dismantled it. **(i)** 0.56 was *undertrained* (M17
  curriculum; plain standard-train reaches 0.689 — M18d) and "more optimization" (the faithful bundle = 4×
  compute, since the DS *mechanism* is inert, M18e/M18f) lifts it only to 0.82, still ≪ 1.0. **(ii)
  Decisively, M18g (h64) + M18i (h128) re-ran the gate with EVERY arm at equal compute (400 epochs, all
  train_acc≈1.0): the single loop is the BEST arm (beats ff, edge GROWING with capacity — Δ(trm−ff) EM
  +0.036 @ h64 → +0.064 @ h128) but STILL plateaus far below the target at every capacity (EM 0.75 → 0.79
  ≪ 1.0), with a feedforward just behind in the same band.** So the single-loop
  "insufficiency" is a **shared capacity/generalization ceiling, NOT a single-*timescale* deficit** (the
  §8 trap: a plain MLP hits the same wall). The §9.3 gate needs the insufficiency to be timescale-specific
  to motivate a second timescale; it is not → **M19 unearned, Task C re-DEFERRED**; the lever is
  data (M5-style — M18h/M18j: the data sweep 4k→16k→64k is 0.75→0.93→0.99, the single loop SOLVES the
  target at 64k, triggering §9.3's null clause; M18i: 2× capacity barely moves it), not H/L. (The loop's edge over ff is leg-2-sized ~0.04–0.06, not the ~0.2 timescale headroom the gate
  needs.) **What survives equal compute (all three §9.2
  legs, honest scope):** leg-1 Δ(trm−decoupled) EM +0.110/+0.161 (8/0); P1 Δ(trm−untied) +0.065 @ w24 (8/0,
  *conservative — untied over budget*); leg-2 loop>ff EM +0.036 @ w24 (8/0, w24-only). All reproduce on the
  two-timescale family (and GROW with capacity — leg-2 +0.036→+0.064 h64→h128, M18i); none is
  timescale-sized (the loop still plateaus ~0.79 ≪ 1.0 at h128). **M18i KILLS the M17b "P1 reverses at
  hidden=128" confound — that was a 1×-compute artifact; at equal compute the loop beats untied at h128,
  budget-clean.** Re-gate condition: a nested instance where the loop plateaus below target **and ff/untied
  do NOT share the ceiling at equal compute** — the current instance doesn't (the loop's edge is leg-2-sized,
  data-bound).
- **The N_sup "win" is MORE OPTIMIZATION, not the detached-carry deep-supervision MECHANISM — the
  adversarial review (B1) was right, and the autopsy's mechanism is essentially inert here (M18/M18e).**
  The repo's old "deep supervision" = per-step readout losses inside one back-propagated forward; the
  TRM/HRM mechanism the ARC autopsy credits = an OUTER loop of `n_sup` passes carrying `(z,a)` **detached**
  between them (`train_deep_supervision`). On `converge` rule 78 w=24 (8 seeds), `n_sup=4` lifts EM
  0.584→0.879 (**+0.295, 8/0**) and the four-ingredient `trm_faithful` bundle hits 0.944 (**+0.359 EM**).
  **BUT `n_sup=4` is 4× the optimizer steps, and two compute-matched controls settle it: (M18e) a NO-CARRY
  arm (4 passes/batch, fresh init each — same compute, no carry) gets +0.282 EM, so the detached CARRY adds
  only +0.012 EM, ns; and (M18f) a PLAIN loop trained 4× the EPOCHS reaches EM 0.873 ≈ trm_nsup's 0.879.**
  So neither the carry nor the N_sup pass-structure buys anything a longer plain run doesn't — the entire
  win is just **more optimization** (the convergent target is undertrained at 100 epochs). The autopsy's
  deep-supervision *mechanism* is ~inert on this anchor; this **upholds** the project's prior "DS is inert"
  verdict — what was missing was training budget, not the carry. Consequently the "faithful loop beats both
  at w=32" is a **compute-unfair** comparison (the loop trained 8× longer than ff; equal-compute ff
  untested), and the practical takeaway is mundane: **train the loop longer on convergent targets** — the
  four faithful ingredients add no mechanism (per-ingredient, EMA-alone is *catastrophic* −0.411 EM,
  RMSNorm/n_latent-alone mildly negative; only N_sup's extra compute helps, and a plain 4×-epoch run
  matches it). **Even "more optimization helps" is regime-specific:** the SAME bundle is INERT on the
  non-convergent rule-30 depth wall (M18a — all arms at test-chance T=8/16, EM=0), so extra training only
  helps where there is a fixed point to converge to (§9.2), and it does NOT grant depth-extrapolation.
  Going forward: **train loops longer on local-update fixed-point tasks; do NOT reach for the faithful
  bundle as a "mechanism."** All M18 knobs are additive/off-by-default → every committed M0–M15c result is
  bit-identical and intact. (This bullet is a walk-back of the pre-review M18 headline; the adversarial
  PR review's B1 caught the compute confound — see LOG.md M18e/M18f.) **This "DS mechanism inert /
  just more optimization" verdict HOLDS on the cross-cell mixer too (M29, after two adversarial reviews): the
  detached-carry MECHANISM is inert on `trm_mixer` (harmful −0.112 on the convergent regime where it is motivated;
  a non-robust 2-seed ceiling rescue [p=0.125] in the one under-fit regime), while the COMPUTE carries any gain —
  exactly M18. An intermediate "m29c reversal" was a ceiling-saturation artifact and is retracted. See the M29
  bullet + LOG.md M29 (both review blocks).**
- **The §9.2 loop finding does NOT transfer to real multi-label tabular under honest evaluation — M20 is a
  NEGATIVE for the loop thesis, with one non-loop positive; CONFIRMED on TWO large datasets (M20, §9.4 bridge;
  verdict after 2 review passes + a proper-eval redo + a replication).** Built a `multilabel` task (vendored,
  network-free, sha-guarded) where EM = subset accuracy and `trm_decoupled` = binary-relevance; ran the M10
  arm set on `emotions` (6 labels) + `yeast` (14) + `scene` (6, near-mutually-exclusive). Implementation
  verified clean (disjoint splits, train-only standardize, bit-identical re-run, budget ±2%). The first runs
  (overlapping random splits, EM-only) *looked* positive, but two adversarial reviews + a **proper evaluation
  (micro/macro-F1 + 10-fold CV, DISJOINT test folds)** dismantled the loop-specific reading, and `scene`
  replicated the corrected verdict. **Verdict (yeast / scene 10-fold CV):** **(1) leg-1 (joint >
  binary-relevance) is the ONLY robust finding** — Δ(trm−decoupled) EM **+0.060 / +0.104**, micro-F1
  **+0.043 / +0.049**, macro-F1 **+0.051 / +0.048**, all **10/0, p=.002**; decoupled worst on every metric —
  **but NOT loop-specific** (the non-recurrent `ff_matched` joint MLP also beats decoupled everywhere; it is
  "joint output modeling > per-label," not recurrence). **(2) Every loop-SPECIFIC edge is EM-ONLY and TIES
  under F1, on both datasets:** Δ(trm−ff) EM +0.019/+0.037 (8-9/1) but **micro-F1 +0.000/+0.001, both 5/5,
  p=1.0**; Δ(trm−untied) likewise EM-only. The loop's EM "win" was a subset-accuracy / **modal-label-combo
  artifact** (probe: 94% of its extra-correct rows were frequent label-sets); under the metric multi-label
  work uses, the loop **ties** a shallow joint MLP. **(3) emotions' CV null was SAMPLE SIZE, not label count**
  — `scene` (6 labels like emotions but large n) fires leg-1 at 10/0 (emotions' old "9/0, p=.004" was pure
  overlapping-split inflation). **(4) The hidden=128 capacity probe (both datasets) FALSIFIES the synthetic
  M11 "grows-with-size" lever:** at 2× width Δ(trm−ff) on F1 stays a tie/slightly favours ff (yeast micro
  −0.009 3/7; scene micro −0.008 3/7; ff ahead per-arm) — NO loop-specific edge emerges with capacity; and
  leg-1 itself does NOT grow (on yeast it *weakens* at h128, decoupled catching up — opposite of synthetic
  M11). **Net: joint multi-label modeling beats binary-relevance (real, robust on 2 large datasets with
  opposite coupling, but a plain MLP gets it — no loop needed); the iterative loop buys NOTHING robust over a
  shallow joint MLP on real tabular, at any capacity tested.** So the synthetic "tied-recurrence coherence"
  value does not cross to real multi-label data as a *loop* property. Substrate additive (F1 gated by
  `want_f1`; K-fold opt-in) → all M0–M18 results bit-identical. Canonical summaries:
  `m20_multilabel_{emotions_smoke_20260626T120659,yeast_20260626T123047,scene_20260626T133446}_*`.
- **The trained loop does NOT settle a latent fixed point — even where it WINS; "dressed-up depth"
  is now MEASURED, corroborating that the loop's value is STATIC, not dynamical (M21, introspection;
  the static verdict itself rests on the M9–M15c leg Δs, M21 adds the dynamical-level evidence).**
  Built a measurement-only latent/weight introspection layer (`eval/introspection.py`: Jacobian
  spectral radius, latent residual + over-unroll readout, path-independence, effective rank,
  Lipschitz product — all reading a trained arm, touching no model code, off-by-default ⇒ M0–M20
  bit-identical) and ran the same suite on the loop's WIN regime (`converge` rule 78) and FAIL
  regime (`iterated` rule 30). The pre-registered hypothesis ("contractive on converge,
  non-contractive on iterated") is **FALSIFIED, and the falsification is the result:** the latent
  residual ‖z_{t+1}−z_t‖/‖z_t‖ ≈ **1.2–1.3 in BOTH** (the latent moves by more than its own norm
  every step — no convergence), ρ>1 with **frac_expanding=1.0 in both**, and over-unrolling
  collapses the readout everywhere (EM 0.58→0.01 converge, 0.82→0.003 iterated). So §8's "the loop
  trivially degenerates into a deep net" is literally true even on the fixed-point target where the
  loop beats its controls — it runs a **fixed-depth feedforward circuit with a depth-tuned readout**,
  not iteration toward an attractor (a stronger, mechanistic M1/M8 null). **The strongest WIN-vs-FAIL
  contrast is PATH INDEPENDENCE, not contraction (but scope it — `trm` only, n=5):** `za_alignment`
  (cosine of deep-unrolled z across random z0 inits) = **0.97 converge vs 0.43 iterated** for the
  loop (the convergent target funnels inits into a shared Geiping-style *orbit*). NOT a clean
  cross-arm discriminator: `trm_decoupled` (also a leg-1 winner) shows za 0.55/0.32, bands
  overlapping — so za tracks the convergent-vs-chaotic *target* for the loop, not *winning*. The
  loop's value is **static joint-state coherence** (§9.2) — but that verdict rests on the M9–M15c
  leg Δs, not on M21; here the legs are only *anchored* at **5 seeds** (Δ(trm−decoupled) EM +0.333,
  but 5/5 → p=0.0625, cannot clear §5's significance floor — indicative, not a fresh demonstration).
  M21 *corroborates* statically: the joint latent compresses to effective rank ~18/64 vs the per-cell
  decoupled ~66/1848 (far more collapsed relative to dims; the 64-vs-1848 mismatch makes this
  suggestive only). NB `readout_agreement` (0.32/0.001) measures whether random inits agree with
  EACH OTHER, not with the trained answer — so it does not by itself prove the loop "can't decode its
  attractor." Tying gives a far smaller Lipschitz product (~6.5 vs the untied stack's ~10⁵).
  **Refinement
  lever handed forward (NOT acted on — §8):** if the goal is to make the loop actually use its
  recurrence (extrapolation / test-time compute, repeatedly found ABSENT), the diagnostics give a
  concrete target — Jacobian-spectral / Lipschitz regularization (DEQ Jacobian-reg arXiv 2106.14342;
  Rethinking Deep Thinking 2410.23451) to push ρ<1, path-independence training (Anil 2211.09961) to
  raise za, a fixed-point loss to lift readout_agreement. **Risk flagged:** the loop wins WHILE non-contractive, so forcing
  contraction may trade away the coherence win — judge any `trm_stable` arm against BOTH the
  extrapolation metric AND the coherence Δ it costs. Tracked:
  `m21_introspection_{converge_20260627T083814,iterated_20260627T083602}_*`.
- **On a domain-motivated airline DISRUPTION operator (M22, user-requested synthetic from an ops
  spec), the §9.2 WITHIN-LOOP structure transfers (joint-state coherence + tying) but the loop has
  NO coherence edge over a shallow feedforward MLP — the spec's "more coherent than feedforward?"
  criterion is answered NO (the M20 real-data pattern: joint modeling helps, but it is not a *loop*
  property).** Built `make_disruption` (one row = one disruption component of `w=24` coupled flights;
  X = per-flight features at t0; y = the settled severe-outcome vector — a JOINT fixed point), an
  integer THRESHOLD net on `W = w_rot·rotation-chain adjacency [LOCAL] + w_bank·bank cliques
  [NON-LOCAL]`, PSD-convergent. **Two design lessons (the 2nd from an ADVERSARIAL REVIEW that caught
  a BLOCKER):** (a) a faithful MONOTONE cascade is ff-EASY/VACUOUS (OR-reachability is linearly
  separable); (b) an over-damped γ drives the threshold net to near-IDENTITY (a linear model solves
  it). The FIRST locked instance was near-affine (linear EM 0.55 > loop) and its dramatic "−0.263
  clean negative" was an ARTIFACT; **the committed instance is re-pinned ff-HARD** (w_rot=6/w_bank=3,
  MINIMAL-PSD γ=14: linear EM 0.23 ≪ ff EM 0.34, copy-frac 0.82). **Corrected run (w=24, 8 seeds, full
  m12 arm set, exact sign test, ✓ all arms within ±2% budget):** per-arm EM **ff 0.429 > trm_stepDS
  0.371 > trm_nods 0.300 > trm_dec_stepDS 0.245 ≫ trm_dec_nods 0.078 ≈ untied 0.056.** (1) **leg-1
  (joint>per-cell) REPRODUCES, trainability-clean:** nods Δ +0.223 (8/0) is partly the decoupled arm
  underfitting (train 0.771), but the trainability-clean step-aligned Δ(trm_stepDS−trm_dec_stepDS) is
  **EM +0.127 (8/0, p=.008)** with both arms training fine — a genuine joint-state edge (transfers to
  this deep+local family, like M15). (2) **leg-2 (loop>ff) = the headline NULL/NEG:** Δ(trm_nods−ff)
  **acc −0.025 (0/8, p=.008, sig)** and **EM −0.128 (1/7, p=.070, NS)** — ff is sig more accurate and
  ≥ as coherent; the loop produces NO more coherent whole-component states than the MLP → spec
  criterion **NO**. (3) **P1 (tying) +0.244 EM (8/0), NOW BUDGET-CLEAN** (untied within ±2%; still
  underfits train 0.801, so partly a width-split optimization gap). **Net: the loop beats its INTERNAL
  ablations (per-cell, untied) but not the external param-matched MLP — joint-state + tying transfer,
  a loop-vs-MLP coherence advantage does not (the M20 verdict on a synthetic ops coupling).**
  Caveats: this sparse-PSD family is ~92% per-cell LINEAR (its hardness is EM-coherence only, NOT the
  §9.2 per-cell-hard regime, so NOT the M13/M14 negative where leg-1 also failed); additive code, all
  M0–M21 bit-identical (226 tests, 11 new). **ROBUST across a hidden∈{32,64,128} × w∈{24,32} sweep
  (m22_size_{small,base,large}):** leg-2 (loop−ff) is NEGATIVE in ALL 6 cells — ff beats the loop on
  token-acc 0/8 everywhere, on EM sig at w24 — and the deficit does NOT close with capacity (plateaus
  ~−0.02 acc / −0.16 EM at w24), the EXACT OPPOSITE of the ECA M11 lever (where 2× hidden flipped leg-2
  positive) and a repeat of the M20 real-data capacity null. leg-1 (joint, 8/0 nods all cells) + P1
  (tying, budget-clean all sizes) reproduce across the grid. So the loop beats its INTERNAL ablations
  but never the external MLP, robustly across scale+width — the M20 verdict on a synthetic ops coupling.
  Canonical: `m22_disruption_base_20260627T160259_*`, `m22_size_{small_20260627T193419,
  base_20260627T203248,large_20260627T222418}_*`. (Minor non-blocking determinism note: grid-vs-standalone
  base agree in sign but differ ~0.02 per-arm — curriculum-RNG / trm_decoupled-matmul; audit pending.)
  **[⚠ UPDATE — this leg-2 NULL is OVERTURNED as a FLAT-ARCHITECTURE artifact by M24f (verdict post-adversarial-review):
  re-run with the cross-cell `trm_mixer`, the mixer significantly out-accuracies ff — Δ(mixer−ff) ACC +0.041/+0.034
  @ w24/w32, 8/0, p=.008 (EM +0.49/+0.48 is the SAME edge amplified by whole-row EM; `coherence_excess` Δ(mixer−ff)
  −0.044/+0.004 [≤0 / ≈0-ns], so it is a RAW per-cell-accuracy win on a ~93%-linear task, NOT a coherence edge) —
  while the flat loop ties/trails ff (acc −0.006/−0.009), so the null was flat-update-specific. It is PRIMARILY the
  mixer ARCHITECTURE not recurrence (M24e-style control, both widths: budget-matched untied mixer already beats ff
  acc +0.034/+0.026 8/0) — with a width-dependent tying refinement: tying is ns at w24 but SIGNIFICANT at w32 (tied
  beats even the 7.5× untied ceiling acc +0.005 / EM +0.088 8/0 — consistent with the untied stack overfitting,
  train 1.0→test 0.755), so recurrence is not entirely irrelevant at higher width, unlike M24e's near-solve
  converge (one width transition, an evidenced observation not a proven law). The mixing operator is
  a distinct capability (untied mixer > untied non-mixer +0.098 acc) but does NOT "dwarf" the §9.2 legs on accuracy
  (mixing +0.047 vs leg-1 +0.065 / P1 +0.058 — the EM "dwarfs" was an EM-floor artifact). Read the M24 cluster
  bullet + LOG.md M24f before citing this null.]**
- **[✅ RESOLVED by the M23 MIXER re-test — the negative was an ARCHITECTURE ARTIFACT, now OVERTURNED: a
  weight-tied loop with a cross-cell MIXING update (`trm_mixer`) SOLVES hard Sudoku a param-matched
  feedforward cannot — Δ(mixer−ff) EM +0.89…+0.96, 6/0, p=0.031, advantage GROWING with difficulty (TRM's
  real signature). The flat `trm` used in the M23 body below could not mix cells and so tied ff; read the
  "M23 MIXER re-test" bullet + LOG.md before using the M23 body's "coherence-only / raw-scale" framing.]**
  **The canonical-TRM positive-control TRIPWIRE (Sudoku) is MET in a SCOPED sense: scaled up, the loop
  SIGNIFICANTLY beats a param-matched MLP on whole-grid EM (+0.092, 15/16 seeds, p=0.0005) — but it is the
  repo's COHERENCE edge in the EASY regime, NOT TRM's hard-puzzle-solving win (M23).** Built `sudoku`
  (synthetic, network-free, deterministic unique-solution puzzles; multi-output fixed point) to test whether
  the repo's `trm` reproduces TRM's HEADLINE win (loop ≫ ff) on its HOME regime — constraint-propagation
  grid reasoning — and thereby validate the implementation behind every prior null. **Stage 1, minimal smoke
  (6×6, 2.5k data, hidden 64): INCONCLUSIVE** — whole-grid EM FLOORED at ~0 for every arm at every difficulty
  (per-cell capped ~0.86 ≪ the ~0.99 needed over 36 cells), loop slightly BEHIND ff per-cell. **Stage 2,
  scale-up (16k data, hidden 128, n_steps 16, deep-supervision, EMA OFF): EM lifts off the floor and the loop
  WINS** — at n_givens=30 (easy, 6 blanks) Δ(trm−ff) EM **+0.0923, 15/16, p=0.00052** (trm EM 0.611 vs ff
  0.519; per-cell +0.0043, 15/16) — the first clean, significant loop-beats-ff result on a canonical TRM task
  in this repo, budget-clean ±5%. **Two load-bearing scope limits:** (1) the EM gain (+0.092) far exceeds the
  per-cell gain (+0.004) ⇒ it is the established whole-grid **COHERENCE** mechanism (leg-2 / M9), NOT a new
  capability; (2) it appears ONLY at the easy, near-saturated end and does **not** grow with difficulty (ff
  ties-or-wins at medium/hard, EM re-floors below n_givens≈24) — the OPPOSITE of TRM's "advantage widens as
  propagation deepens," so the **canonical hard-solving win is NOT reproduced** at the CPU scale reachable here
  (likely needs ACT/PonderNet halting — the repo LACKS it — + far more compute / 9×9). **Confounds for any
  rerun:** EMA 0.999 is CATASTROPHIC at short training (~1k updates ≪ warmup) — collapsed `trm_decoupled` to
  baseline 0.167 (reproduces M18); leave EMA off. `trm_decoupled` is ~12× slower at hidden=128/n_steps=16
  (951s/8 ep) and collapses — the leg-1 attribution needs a cheaper config. `untied_matched` collapses to a
  non-learning baseline (~0.33) at this width, so P1 is not cleanly testable on Sudoku. **Net: the
  implementation is SOUND (the loop genuinely beats the MLP once EM has signal), and the result CORROBORATES
  the central thesis (loop value = whole-grid coherence on fixed-point targets, §9.2, NOT algorithmic depth) —
  so the program's tabular negatives are TRUSTWORTHY, not a missed hidden capability.** Infrastructure
  (generator, multi-class `num_classes` runner support, 10 tests, configs) built + validated + reusable.
  Canonical: `m23_sudoku_{screen,base}_20260628T*` (smoke), `m23_sudoku_scaleup_{screen,base,sig}_20260628T*`
  (scale-up); full narrative LOG.md M23.
- **ACT (the §4/§12 unbuilt TRM ingredient) is now BUILT, faithful, and demonstrably ADAPTIVE — but it
  does NOT unlock the canonical hard-solving win (M23-ACT).** [⚠ The stronger reading — "so the negatives are
  NOT an artifact of missing ACT" — is UNDERCUT by the M23 correction: ACT was tested on the flat-MLP TRM that
  lacks the cross-cell mixing operator, so its null on hard puzzles may reflect that architecture gap, not a
  ceiling of adaptive computation. ACT works; whether it helps a *cell-mixing* loop is untested.] Added a halt head + `train_act` (segmented detached-carry deep supervision + BCE halt
  target = per-example exact-match) + adaptive `evaluate_act` (OFF by default ⇒ bit-identical; 241 tests).
  Decisive sweep (6×6, 16k, hidden 128; `trm_act` [ACT, max 8 seg] vs `trm_seg` [fixed 8 seg] vs `ff`; 6
  seeds): **(1) ACT WORKS** — `avg_segments` scales monotonically with difficulty **1.00 → 7.15 → 8.00**
  (n_givens 30/24/18): solves easy puzzles in one segment, spends all 8 on hard ones (the repo's first
  genuine adaptive test-time compute). **(2) Adaptivity buys NOTHING** — Δ(trm_act−trm_seg) EM is a tie
  everywhere, and slightly NEGATIVE on easy (−0.057: halting at segment 1 discards the refinement that
  8 segments use to build coherence — trm_seg EM 0.543 vs trm_act 0.487). **(3) The loop's edge does NOT
  grow with difficulty — it REVERSES (the opposite of TRM's signature):** Δ(trm_seg−ff) EM **+0.073 (easy,
  6/0, p=0.031)** → −0.024 (med) → −0.002 (hard); ff is sig. more accurate at med/hard (Δacc 0/6). The
  fixed-segment loop replicates the easy-regime coherence win (consistent with the +0.092 scale-up) but it
  erodes and flips as puzzles harden. **Verdict: the FULL TRM recipe (segmented deep supervision + working
  ACT) still shows only the coherence edge, never algorithmic hard-solving.** The remaining gap to
  Sudoku-Extreme is NOT an ingredient we lacked but raw SCALE (far beyond a 4-core CPU) and, per M21, the
  fact that this loop doesn't learn genuine iterative search (fixed-depth pattern-matching — why more
  depth/segments/adaptive-compute all come back null on the hard regime). Pre-test confirmed fixed 8-seg
  depth alone also ties ff at hard. Canonical: `m23_sudoku_segments_pretest_*`, `m23_sudoku_act_sweep_20260702T*`.
- **★ M23 MIXER re-test — the M23 NEGATIVE is OVERTURNED: the loop DOES do algorithmic hard-solving once
  its update can MIX cells; the "coherence-only, no hard-solving" verdict was a FLAT-ARCHITECTURE artifact,
  exactly as the adversarial review predicted.** Built `trm_mixer` (weight-tied loop; update = MLP-mixer
  over the size² cells [token-mixing = constraint propagation] + per-cell channel MLP) — ONE knob vs the
  flat `trm` (the mixing operator), everything else matched (tying, deep supervision, RMSNorm, n_steps,
  data; params ±0.3%). On 6×6 at the difficulties where the flat trm/ff FLOOR (EM≈0 even at 80 epochs/16k in
  M23): mixer EM **0.970 / 0.955 / 0.893** at n_givens 24/18/14 vs flat/ff ≈ 0 — **Δ(mixer−ff) EM +0.96/+0.96/
  +0.89, all 6/0, p=0.031**, in ~15 epochs on 8k data. Three facts: **(1)** the advantage is HUGE and GROWS
  as puzzles harden — TRM's canonical signature, the OPPOSITE of the flat loop's M23 reversal; **(2)** it is
  the OPERATOR, not loop-vs-ff and not scale — `trm_flat ≈ ff` everywhere (Δ≈0, ns: a weight-tied loop with
  a FLAT update is no better than a feedforward), the win needs the cell-mixing update, and it appears with
  LESS compute than the flat arms had in M23; **(3)** discrepancies B (1-step gradient) and C (depth) were NOT
  needed — full BPTT at n_steps=8 suffices once the operator is right. **Broader implication (flagged, NOT yet
  tested — the key open question): every M0–M22 negative used the FLAT `trm`, so they are negatives about a
  loop with a FLAT update.** Pure-tabular / exchangeable-feature targets (parity) have no structure to mix and
  are likely robust, BUT the **CA / converge / hopfield family has ring/graph structure a mixer could exploit**
  and should be RE-RUN with `trm_mixer` before their negatives are trusted as architecture-independent. The
  synthetic thesis must be re-read as "loop-with-a-FLAT-update ≈ feedforward"; whether iterative refinement
  helps is confounded with whether the update operator matches the task structure (the §8 trap, one axis
  deeper than controlled). Caveats: 3-D matmul ⇒ bit-repro only at fixed num_threads (=1); baselines floor at
  15 AND 80 epochs so not merely undertrained; easy cells (n_givens≥30) trivially solved (EM→1, omitted).
  Canonical: `m23_sudoku_mixer_lean_20260702T*`; full narrative LOG.md "M23 MIXER re-test".
- **[⚠ ADVERSARIAL-REVIEW CORRECTION + M24e RESOLUTION apply to ALL M24 bullets below — read first. The Δs
  REPRODUCE, but the headline ATTRIBUTION is walked back. (B1, BLOCKER — now RESOLVED by M24e toward
  ARCHITECTURE): "the LOOP does algorithmic hard-solving" is FALSE. The mixing-matched §4b controls
  (`untied_mixer`/`untied_mixer_matched`) show a NON-recurrent untied mixer stack MATCHES the tied loop given
  equal capacity — Δ(tied − 7×-untied-ceiling) EM is ns at every w (p≥.07), while even a BUDGET-matched untied
  mixer beats ff by EM +0.31/+0.60/+0.65 (8/0). So the cross-cell MIXING ARCHITECTURE does the hard-solving;
  weight-tied recurrence adds only the ordinary parameter-EFFICIENCY edge (P1: Δ(tied − untied_matched) EM
  +0.03/+0.07/+0.29, 8/0, growing with w) but NO capability an untied mixer of equal capacity lacks — mirroring
  M21 ("the loop is dressed-up depth"). Corrected headline: *a mixing architecture does the hard-solving; tying =
  efficiency, not recurrence-qua-capability.* The "it's the operator not recurrence, because flat≈ff" logic was
  also FALSE on converge (Δ(flat−ff) EM +0.14/+0.14/+0.08, 8/0 — the flat loop beats ff). (M1) M24b's "opposite
  width-scaling ⇒ distinct mechanisms" is an EM-FLOOR artifact: on token-acc tying Δ(flat−untied) GROWS
  (+0.087→+0.120), leg-1 is stable (~0.07); keep only "the mixer is width-robust while flat-update arms degrade."
  (M3) the converge win is RAW per-cell accuracy (coh_excess vs ff is NEGATIVE −0.061), NOT the coherence
  mechanism. (M4) the multi_parity mixer is ~broken on parity, so that control shows DIRECTION (mixer<ff 0/8) not
  a clean marginal≈0. Full blocks: LOG.md "M24 — ADVERSARIAL-REVIEW CORRECTION" + "M24e".]**
- **★ M24 — the MIXER re-test carried to the ORIGINAL ring tasks (`converge` + `iterated`): the verdict SPLITS
  along the CONVERGENT / NON-CONVERGENT axis. On the CONVERGENT fixed-point ring the mixer SOLVES the task and
  the win GROWS with difficulty (the CA `converge` negatives WERE a flat-architecture artifact); on the
  NON-CONVERGENT moving CA the mixer improves the per-step operator but does NOT crack the depth wall (the
  depth-extrapolation negatives SURVIVE the mixer).** Lean M23 isolation — `trm_mixer` vs `trm_flat` vs
  `ff_matched`, same training (standard train + deep supervision + RMSNorm + full BPTT), the mixing operator the
  ONLY changed variable; `distractors: 0` for all arms (mixer needs `in_features % out_features == 0`); 8 seeds;
  num_threads=1. **`converge` (rule 78, sweep w∈{24,32,48}):** the mixer solves it at every width (acc ~0.998 ⇒
  EM ≈ 1.0) while flat/ff degrade with w (acc 0.982→0.934 flat, 0.976→0.889 ff ⇒ EM collapses at ~0.93^48).
  **Δ(mixer−ff) EM +0.345 / +0.676 / +0.942 and Δ(mixer−flat) EM +0.204 / +0.538 / +0.859 — all 8/0, p=.008,
  GROWING monotonically with w** (the M23 signature on a ring CA). The flat loop's leg-2 edge over ff reproduces
  (Δ(flat−ff) EM +0.14/+0.14/+0.08, 8/0) but is DWARFED — so the "small/marginal CA loop edge" the whole M8–M15
  program characterized was the flat architecture leaving the win on the table (conservative at w48: mixer wins
  with ~13% FEWER params, budget ratio 0.866). **`iterated` (rule 110, non-convergent moving CA, sweep T∈{4,8,12},
  couple_n_steps_to_param=T):** a LARGE per-step operator win at the learnable depth — T=4 mixer acc 0.903 vs
  flat 0.661 ≈ ff 0.665, **Δ(mixer−ff) acc +0.238 / EM +0.109, both 8/0, p=.008** (flat≈ff, 4/4, p=1.0 — the
  M3a "flat loop = feedforward" replicated) — but the mixer does NOT crack the depth wall: at T≥8 the M3a
  moving-target wall reasserts (all arms ~baseline 0.53, mixer edge +0.016 ns at T=8, gone at T=12), so the
  advantage is LARGEST at T=4 and vanishes (the OPPOSITE of grows-with-difficulty). **Net: the mixer amplifies
  the loop's value exactly where the program located it — fixed-point / convergent targets (§9.2) — without
  unlocking depth-extrapolation (M1/M3a/M7/M8/M21).** So of the M0–M22 negatives, the COHERENCE-on-convergent-ring
  ones are now shown to be flat-architecture artifacts (mixer solves them); the DEPTH-extrapolation negatives
  survive. **M24b — the MECHANISM ablation (adding `trm_decoupled`/`untied_matched` to the converge setup) —
  ANSWERS the attribution: the mixer's win is a DISTINCT capability, NOT leg-1 done better.** Using the flat loop
  as the pivot (it already carries the joint state ⇒ Δ(mixer−flat) isolates mixing): the win decomposes into TWO
  significant mechanisms with OPPOSITE width-scaling — leg-1 joint-state Δ(flat−decoupled) and P1 tying
  Δ(flat−untied) are large but SHRINK with w (EM +0.58→+0.09, +0.60→+0.10), while the MIXING marginal
  Δ(mixer−flat) GROWS (EM +0.20→+0.54→+0.86); all 8/0, p=.008, decoupled trainability-clean (acc 0.86–0.92). So
  ordinary joint-state/tying coherence (the established §9.2 value) dominates at small w and plateaus; the explicit
  cross-cell propagation operator is width-robust (mixer acc 0.999→0.998 vs flat 0.982→0.934) and carries the win
  at w≥32 — exactly where M9 found the flat loop's coherence edge over ff vanished. P1 is BUDGET-CLEAN here (untied
  within ±2%). Updated picture of the loop's value on local-update fixed-point rings: **joint-state coherence
  (leg-1/P1, established) + a cell-mixing propagation capability (M24/M24b, NEW) — the latter is what the flat
  architecture left on the table.**
- **★ M24c + M24d — the two mixer CONTROLS bound the capability and REFRAME the dividing line: the mixer helps
  wherever outputs are CROSS-CELL COUPLED (local `converge` AND — OVERTURNING the pre-registered null — DENSE
  `hopfield`), and NOT where they are EXCHANGEABLE (`multi_parity`). "Cross-cell dependency," not "local rule," is
  the axis.** Same lean 3-arm isolation (mixer vs flat vs ff, M24 recipe, distractors=0 for all — mixer divisibility
  constraint; 8 seeds, num_threads=1). **M24c `hopfield` (dense threshold-net attractor, w∈{24,32}) — NOT the
  expected null: the mixer WINS 8/0** — Δ(mixer−ff) acc +0.046/+0.044, EM +0.454/+0.362; Δ(mixer−flat) acc
  +0.063/+0.059, EM +0.496/+0.377; all p=.008, budget-clean. The flat loop REPLICATES M13 (Δ(flat−ff) acc
  −0.017/−0.015, 0/8 — flat < ff on a dense net) but the mixer beats both: the token-mix is a learned ALL-TO-ALL
  cross-cell map, so it fits DENSE coupling as naturally as local — **M13's "loop edge is CA/local-specific, dies on
  a dense net" was a FLAT-loop property, not a property of refinement** (same lesson as M23/M24). Caveat: distractors=0
  ⇒ easier than M13's noise regime (ff acc 0.93 vs M13 EM ~0.26), and the mixer's divisibility constraint makes the
  distractor=8 instance untestable — so the qualitative reframe stands, the exact-M13-regime question is out of reach.
  **M24d `multi_parity` (w independent k=4 parities, w∈{4,8}) — the NEGATIVE control WORKS:** Δ(mixer−flat) ≤ 0 and ns
  (acc −0.050 p=.29 / −0.087 p=.07; EM −0.033/−0.008) — mixing adds NO positive marginal on exchangeable outputs (vs
  converge's +0.20→+0.86), and the mixer NEVER beats ff (Δ(mixer−ff) acc −0.32/−0.23, 0/8, p=.008 — a plain MLP is
  best where outputs are uncoupled). So the M24 win is STRUCTURE-specific, not generic capacity. Caveat: loop arms
  train unstably on parity under this recipe (some seeds at chance), so the airtight statement is the 0/8 mixer<ff.
  **Net reframe (branch-closing): the mixer's cross-cell propagation capability is gated by cross-cell DEPENDENCY,
  not locality — fires on local (`converge`) AND dense (`hopfield`) coupling, inert on independent outputs
  (`multi_parity`); this further localizes the M13 negative as a flat-architecture artifact.** Canonical:
  `m24c_hopfield_mixer_20260703T150202_*`, `m24d_multiparity_mixer_20260703T150912_*`; full narrative LOG.md M24c+M24d.
  `disruption` (MED structure) since tested and is a mixer WIN too (M24f — the M22 leg-2 null was a
  flat-architecture artifact; Δ(mixer−ff) ACC +0.041/+0.034, 8/0 — EM +0.49/+0.48 is that edge amplified, NOT a
  coherence win [coh_excess Δ ≤ 0]; architecture-not-recurrence per the M24e-style control). Earlier bullets'
  claim that these controls were "expected NULL" is superseded by M24c (hopfield is a mixer WIN). **`mixed_converge`
  is now DONE too (M28): a mixer WIN** — the mixer (near-)SOLVES the per-position NON-uniform CA (EM 0.95/0.88 vs
  ff 0.51/0.11, Δ(mixer−ff) EM +0.44/+0.77 8/0, growing with w — but on a small acc gain +0.03/+0.07, coh_excess
  Δ≤0, an EM-floor amplification not a coherence win); architecture-not-recurrence (untied_mixer_matched beats ff,
  ceiling ties the tied loop). **NB (adversarial-review walk-back):** M15's leg-2 "loop-beats-MLP needs a UNIFORM
  rule" does NOT survive here — but at the mixer-required `distractors: 0` the FLAT loop also beats ff (EM
  +0.16/+0.30, 8/0), so the requirement's dissolution is CONFOUNDED with the distractors=8→0 change (uncontrolled
  vs M15) and is NOT cleanly a flat-architecture property; the clean result is the mixing marginal Δ(mixer−flat).
  So EVERY cross-cell-coupled synthetic task is now mixer-tested.
- **★ M28 — the mixer re-test carried to the last two untested tasks: `mixed_converge` is a mixer WIN,
  `nested_converge` (Task C) is the informative EXCEPTION — the mixer improves but does NOT re-open the
  hierarchy gate.** Audited the repo against M23–M26 and found exactly two cross-cell-coupled synthetic tasks
  never run against `trm_mixer`: `mixed_converge` (M15) and `nested_converge` (M17, the Task C substrate).
  Same M24 recipe (lean triple + M24b mechanism + M24e tying attribution; `distractors:0`, num_threads=1, 8
  seeds; additive ⇒ all M0–M27 bit-identical; budget-clean ±5%). **(1) `mixed_converge` (w∈{24,32}) — the
  mixer (near-)SOLVES the per-position NON-uniform CA:** EM 0.949/0.880 vs ff 0.510/0.105, Δ(mixer−ff) EM
  +0.44/+0.77 (8/0, growing with w — small acc gain +0.03/+0.07, coh_excess Δ≤0 ⇒ EM-floor amplification, not a
  coherence win); mixing marginal Δ(mixer−flat) EM +0.28/+0.48 (8/0). Architecture-not-recurrence
  (untied_mixer_matched beats ff EM +0.39/+0.69; 7× ceiling ties/beats the tied loop; tying = small efficiency
  EM +0.05/+0.09). **On M15's leg-2 (adversarial-review walk-back):** "loop-beats-MLP needs a UNIFORM rule" does
  NOT survive here — BUT at the mixer-required `distractors:0` the FLAT loop also beats ff (EM +0.16/+0.30, 8/0),
  so the dissolution is CONFOUNDED with the distractors=8→0 change (uncontrolled vs M15) and is NOT cleanly a
  flat-architecture artifact (and no uniform arm was run at distractors=0 to re-test the contrast); the clean
  architecture stat is the mixing marginal Δ(mixer−flat). **(2) `nested_converge` @4k (n_blocks 3/4) — the
  mixer IMPROVES but does NOT solve, and the ceiling is SHARED by the non-recurrent mixer ⇒ Task C stays
  CLOSED:** Δ(mixer−ff) EM +0.076/+0.130 (8/0) but trm_mixer EM 0.79/0.70 ≪ 1.0 (unlike converge/mixed where
  the mixer solved). The §9.3 re-gate needs the loop to plateau below target AND the non-recurrent control to
  NOT share the ceiling — it is NOT met: Δ(untied_mixer_matched−ff) captures the whole edge (EM +0.098/+0.095,
  8/0), tying is NS (Δ(mixer−untied_mixer_matched) EM −0.022 1/7 @w24 / +0.035 7/1-ns @w32), and the 7× ceiling
  matches/beats the tied mixer. So the nested 4k plateau is the same shared capacity/data wall M18g found for
  the flat loop, now shown to hold for the MIXING architecture too — and the **data sweep 4k→16k→64k CONFIRMS
  it: trm_mixer EM 0.79/0.70 → 0.93/0.90 → 0.99/0.985 (w24/w32), the mixer SOLVES the nested target at 64k**
  (the M18j signature; data, not a second timescale, is the lever, triggering §9.3's null clause). **Do NOT
  build M19.**
  §9.2 legs (leg-1 joint-state, P1 tying) reproduce on both tasks. Canonical:
  `m28a_mixed_converge_{mixer,untied}_20260706-07T*`, `m28b_nested_converge_{mixer,untied}_20260707T*`; full
  narrative LOG.md M28.
- **★ M29 — the DEEP-SUPERVISION MECHANISM re-tested on the mixer. FINAL verdict (after TWO adversarial reviews;
  an intermediate "m29c reversal" was itself over-claimed and RETRACTED): M18's "the detached-carry deep-supervision
  MECHANISM is inert; the N_sup gain is just more optimization" verdict HOLDS on `trm_mixer` — and M29c, correctly
  analyzed, STRENGTHENS it (the carry mechanism is inert even given optimization headroom), it does NOT reverse it.**
  Trail: m29a/m29b found the carry inert-to-harmful on the mixer but with NO optimization headroom (review-1 MAJOR 1:
  inconclusive — can't separate "mechanism inert" from "nothing helps a model that already fits train"). M29c found
  the one headroom regime (`iterated` rule 110 T=6, non-convergent, mixer train 0.80) and I FIRST read it as a
  reversal ("carry +0.054, 8/0"). Review-2 DISMANTLED that: the +0.054 is a non-robust **2-seed ceiling rescue**
  (seeds 1,3 = +0.21; the other 6 within one test-cell of a tie at ceiling; median ≈ 0.0004; drop ceiling-ties →
  4/4 → **p=0.125, BELOW §5's floor**), i.e. the SAME saturation trap review-1 flagged (its MAJOR 3), now reversed.
  The robust gain is the compute-matched **no-carry** arm (+0.185) = 4 consecutive steps/batch = *batch-repetition
  COMPUTE* — which by M18's own "nocarry = compute, carry = mechanism" decomposition UPHOLDS M18. On the convergent
  target where the carry is motivated (m29a converge) it is −0.112 (HARMFUL). The only non-M18 wrinkle is narrow and
  not the mechanism: on that one non-convergent operator the compute is not deliverable as reshuffled 4× epochs (a
  batch-ORDERING effect, no isolating control, bimodal training — a solved seed *regresses* under 4× epochs). Every DS finding (M18 included) had been measured on the flat `trm` (≈ ff); since the mixer
  landed DS was only ever HELD FIXED in the M24 recipe, never ablated — so by the M23/M24 logic it needed a mixer
  re-test. Ran the M18 decomposition on `trm_mixer` (+ flat `trm` replicated IN-RUN) on two regimes (the M27
  saturation trap: a DS null is vacuous without headroom) — undertrained `converge` (w24, distractors=0,
  epochs=25, `mixer_nods` EM 0.86 ≪ the 0.97 ceiling but token-acc 0.99) and hard `nested_converge` @4k (mixer
  EM 0.78 at full training — but **train_acc 0.994, so this is DATA/EM headroom, NOT optimization headroom**). 8
  seeds, exact sign test, budget-clean, num_threads=1. **One additive code change:** widened run.py's `n_sup>1`
  gate to admit `trm_mixer` (its `forward` already exposes the init_state/return_state API; the per-cell `(z,a)`
  detaches element-wise) — off-by-default ⇒ M0–M28 bit-identical; 5 tests (`tests/test_ds_mixer.py`). **(1) The
  detached-carry N_sup MECHANISM is INERT on the mixer, both regimes:** pivot `mixer_nods`, raw 4× compute helps
  (Δ(nocarry−nods) EM +0.107 converge / +0.0415 nested, 8/0) but the CARRY adds nothing (Δ(nsup−nocarry) EM
  −0.112 converge [1/7 ns, DESTABILIZING] / −0.009 nested [3/5 ns]); and plain 4×-EPOCH matches-or-beats N_sup
  (mixer_nods@100 EM 0.974 ≥ no-carry@25 0.965 ≫ carry@25 0.853 on converge; @400 EM 0.801 ≈ N_sup@100 0.81–0.82
  on the shared ~0.80 nested plateau). So the N_sup gain that exists is raw compute (matched by plain epochs), and
  the carry is null-to-harmful — CONSISTENT with M18 but NOT decisive (MAJOR 1: raw compute barely helps the mixer
  either — +0.0415 nested — because the mixer has no optimization headroom, so a carry-null there is nearly
  by-construction, not a clean "DS mechanism inert"). **(2) Per-step-READOUT DS is NOT robustly load-bearing either:** Δ(mixer_ds−
  mixer_nods) EM +0.075 8/0 on converge but at the token-acc ceiling (coh_excess≤0, a raw near-ceiling effect)
  and +0.010 5/3 NS on nested (genuine headroom) — inert where it counts. **(3) The one NEW finding — an
  ARCHITECTURE CONTRAST:** the carry is NOT universally inert — on the weak FLAT loop (≈ ff) it HELPS
  (Δ(flat_nsup−flat_nsup_nocarry) EM +0.129 converge 8/0 / +0.038 nested 7/0, sig) and BEATS plain 4×-epoch
  (flat carry-N_sup 0.624 vs 0.485 converge; 0.805 vs 0.734 nested) — a real effect that DEPARTS from M18's own
  flat null (+0.012 ns, which was distractors=8 / 100-epoch base = LESS undertrained). The carry adds effective
  depth the weak flat operator benefits from, but the mixer is nearer its ceiling in both regimes — **this split
  is CONFOUNDED with operating point (MAJOR 2): "carry helps whoever is further from ceiling" is not excluded, and
  the "mixer converges in one pass" reading is itself an operating-point argument, not a clean architecture
  dissociation** (it is *consistent with* the mixer's IN-OPERATOR work, M23/M24 vs M21, not proof of it).
  **Net (FINAL, post-review-2):** M29c does NOT reverse M18/m29a-b — it STRENGTHENS them. Even in the one mixer
  optimization-headroom regime, the detached-carry MECHANISM is inert (2-seed ceiling rescue, p=0.125 once
  ceiling-ties removed), while the COMPUTE (no-carry batch-repetition, +0.185) carries the gain — exactly M18. So
  on the mixer: the DS *mechanism* is inert (now shown WITH headroom, not just where the mixer fits); the gain from
  N_sup is more optimization. **Meta-fact:** the mixer FITS TRAIN on every convergent cross-cell task screened
  (converge/mixed/nested at 8k–16k), so optimization-headroom regimes for it are rare (only the non-convergent M3a
  depth-wall edge) — the mechanism has essentially no task on which to help the mixer. NO BLOCKER anywhere (code
  faithful, bit-identity TRUE, all numbers re-derived from the JSON/CSVs, 5 tests + 275 total pass, ruff clean);
  both the reversal AND this retraction are interpretation corrections, not code bugs. Caveats: m29c is ONE regime,
  NON-CONVERGENT (moving CA, not the fixed-point regime the carry is motivated on), EM-floored (token-acc),
  high-variance/bimodal pivot; the only non-M18 wrinkle (compute ≠ reshuffled epochs) is an uncontrolled
  batch-ordering effect. Per-step DS not robustly load-bearing for the mixer. Residual m29a/m29b caveats: the flat
  "carry helps" result contradicts M18's flat null but is not one-knob-isolated (distractors 8→0 + epochs + n_train
  all changed); the "just more epochs" comparison is cross-run/unpaired (indicative, M18f precedent);
  per-seed EM not persisted (audit gap — only `accuracy_per_seed` is); budget tol ±5% (mixer +4.1%); single-width
  (w24), one rule pair. Canonical: `m29a_ds_mixer_converge_20260708T095050_*` (+ `_4xep_20260708T162954`),
  `m29b_ds_mixer_nested_20260708T155324_*` (+ `_4xep_20260708T190109`), `m29c_ds_mixer_iterated_20260709T133451_*`
  (+ `_4xep_20260709T184421`) — the headroom test (its "reversal" retracted by review-2; it UPHOLDS M18); LOG.md
  M29 + its two ADVERSARIAL REVIEW blocks + M29c.

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
and contrasted it with uniform rule 78, **partially resolving the M14 confound into a two-leg
decomposition**: **(leg 1, CLEAN)** the **joint-state coherence mechanism** (joint ≫ per-cell, M10) is
**deep + local**, NOT translation-invariance — it **TRANSFERS** to the non-uniform mixed-CA (within-task
trainability-clean ΔEM +0.206, 10/0, p=.002 @ w=24; decoupled < ff); **(leg 2, SUGGESTIVE/CONFOUNDED)**
**loop-beats-the-MLP** does not reproduce without the uniform rule (loop +0.133 EM on uniform-78 vs −0.028
ns on mixed) — and **M15b STRONGLY SUPPORTS this with a max-depth-matched uniform control** (uniform CAs
through the same filter, capped to depth ≤6: loop beats ff +0.090/+0.175 EM, 9/1, p=.021, while mixed
ties; mixed is *harder* so the hardness gradient runs against the result). P1 survives both. **M15c then
CLOSED the central-depth residual** with a `depth_profile` stratified-subsample (all tasks at an identical
depth histogram, mean 3.40): at fixed depth, uniform rule 13 → loop beats ff +0.210 EM (10/0) vs mixed ties
(−0.005), hardness against the result — **leg 2 confirmed depth-controlled on rule 13**; rule 78's edge was
partly depth (drops to +0.032 ns, becomes ff-easy). Leg 1 + P1 hold at matched depth. **Leg 2 stands
depth-controlled (rule 13); only the definitional uniformity↔rule-cardinality entanglement remains.**

**M20 (the §9.4 real-tabular bridge, step 1) is DONE — and, properly evaluated, a NEGATIVE for the loop:**
under micro/macro-F1 + 10-fold CV (disjoint folds), joint multi-label modeling beats binary-relevance (leg-1,
yeast, 10/0 on EM AND F1) **but a plain joint MLP gets that too** (decoupled is worst on every metric, ff also
beats it), and **every loop-SPECIFIC edge is EM-only and ties under F1** (Δ(trm−ff) F1 = 5/5). So the loop
buys nothing robust over a shallow joint MLP on real tabular; the apparent EM win was a modal-label-combo
artifact (§11(b) M20 bullet; LOG.md "M20 — PROPER EVALUATION"). **M19 (the H/L build) is still NOT earned —
the Task C gate FAILED its equal-compute control test (M18g), Task C re-DEFERRED.** Open threads, in rough
priority:
- **★ The canonical-TRM Sudoku TRIPWIRE is DONE and the loop PASSES it — but only with the right ARCHITECTURE
  (M23 → correction → MIXER re-test, resolved).** The flat `trm` tied ff and its edge reversed with difficulty;
  ACT (built, faithful, adaptive) didn't change that. An adversarial review found the confound — our TRM
  FLATTENED the grid, lacking the cross-cell MIXING operator the TRM paper credits for Sudoku (74.7%→87.4%) —
  and the **MIXER re-test OVERTURNED the negative:** `trm_mixer` (MLP-mixer over the cells; one knob vs flat
  `trm`, budget-matched ±0.3%) SOLVES hard Sudoku ff cannot — **Δ(mixer−ff) EM +0.89…+0.96, 6/0, p=0.031,
  GROWING with difficulty** (TRM's real signature), in ~15 epochs on 8k data where the flat arms floored at
  80 epochs/16k. `trm_flat ≈ ff` (a flat-update loop = feedforward), so it is the OPERATOR, not scale or
  loop-vs-ff. **So the loop DOES do algorithmic hard-solving; the "coherence-only" verdict was a flat-architecture
  artifact.** The open thread this opened — re-run the CA ring family with `trm_mixer` — is **PARTLY DONE (M24,
  below).** Reusable: `trm_mixer` + generator + ACT machinery; `trm_mixer` pins num_threads=1 (3-D matmul). Full
  narrative: LOG.md "M23 MIXER re-test".
- **★ M24 (the M23-mixer re-test carried to `converge` + `iterated`) is DONE — the verdict SPLITS on the
  convergent/non-convergent axis.** On the CONVERGENT fixed-point ring (`converge`, rule 78, w∈{24,32,48}) the
  mixer SOLVES the task (EM ≈ 1.0 at every width) while flat/ff degrade, **Δ(mixer−ff) EM +0.35/+0.68/+0.94 and
  Δ(mixer−flat) +0.20/+0.54/+0.86, all 8/0 p=.008, GROWING with w** — so the marginal CA `converge` loop edge the
  M8–M15 program characterized WAS a flat-architecture artifact (conservative at w48, mixer wins with ~13% fewer
  params). On the NON-CONVERGENT moving CA (`iterated`, rule 110) the mixer gives a large per-step operator win at
  the learnable depth (T=4: Δ(mixer−ff) acc +0.238 / EM +0.109, 8/0; flat≈ff) but does NOT crack the M3a depth
  wall (T≥8 all arms ~baseline) — so the depth-extrapolation negatives SURVIVE the mixer. Net: the mixer amplifies
  the loop's value on fixed-point/convergent targets (§9.2) without unlocking depth (M1/M8/M21). **The mechanism
  ablation (M24b) is DONE:** adding `trm_decoupled`/`untied_matched` to the converge setup shows the mixer's win is
  a DISTINCT capability, not leg-1 done better — joint-state (leg-1) + tying (P1) dominate at small w and FADE (EM
  +0.58→+0.09), the MIXING marginal Δ(mixer−flat) GROWS (+0.20→+0.86) and carries the win at w≥32 where the flat
  loop's coherence runs out; all 8/0, decoupled trainability-clean, P1 budget-clean. **The two CONTROLS are DONE
  (M24c/M24d) and reframe the dividing line:** the mixer helps wherever outputs are CROSS-CELL COUPLED — local
  `converge` AND (overturning the pre-registered M13-based null) DENSE `hopfield` (mixer beats flat+ff 8/0, EM
  +0.36…+0.50; the flat loop still fails M13-style) — and NOT where outputs are EXCHANGEABLE (`multi_parity`:
  Δ(mixer−flat)≤0 ns, mixer<ff 0/8). So "cross-cell dependency," not "local rule," is the axis, and the M13 negative
  is further localized as a flat-architecture artifact. **[⚠ see the M24 ADVERSARIAL-REVIEW CORRECTION above: the
  M24b "opposite width-scaling ⇒ distinct mechanisms" claim in this bullet is an EM-floor ARTIFACT — on token-acc
  tying GROWS, not fades; keep only "mixer is width-robust while flat-update arms degrade." And the whole M24 arc
  lacks the §4 mixing-matched controls, so "loop does the hard-solving" is unearned.]** **B1 RESOLVED (M24e)
  toward ARCHITECTURE:** the §4b mixing-matched controls (`untied_mixer` ceiling + `untied_mixer_matched`) show a
  NON-recurrent untied mixer stack MATCHES the tied loop at equal capacity (Δ(tied−7×-ceiling) EM ns at every w,
  p≥.07) and a budget-matched untied mixer already beats ff by EM +0.31/+0.60/+0.65 (8/0) — so the cross-cell
  MIXING architecture does the hard-solving; weight-tied recurrence adds only the parameter-EFFICIENCY edge (P1:
  Δ(tied−untied_matched) EM +0.03/+0.07/+0.29, 8/0, growing with w), NOT a capability an untied mixer of equal
  capacity lacks (mirrors M21's "dressed-up depth"). "The loop does hard-solving" → "a mixer does; tying =
  efficiency." **`disruption` DONE (M24f, verdict post-adversarial-review):** carried the mixer re-test (+ M24b
  mechanism + M24e tying controls) to the airline `disruption` operator — the M22 leg-2 NULL was a FLAT-ARCHITECTURE
  artifact: the mixer significantly out-accuracies ff (Δ(mixer−ff) ACC +0.041/+0.034 @ w24/w32, 8/0, p=.008; EM
  +0.49/+0.48 is the SAME edge amplified by whole-row EM — `coherence_excess` Δ(mixer−ff) −0.044/+0.004 [≤0/≈0-ns],
  a raw per-cell-accuracy win on a ~93%-linear task, NOT coherence) where the flat loop ties/trails ff, and it is
  PRIMARILY the ARCHITECTURE not recurrence (budget-matched untied mixer beats ff acc +0.034/+0.026 8/0 at w24/w32).
  Tying is width-dependent: ns at w24 (M24e efficiency edge) but SIGNIFICANT at w32 (tied beats the 7.5× untied
  ceiling acc +0.005 / EM +0.088 8/0 — consistent with the untied stack overfitting; one width transition, an
  evidenced observation not a proven law). The mixing operator is a distinct capability (untied mixer > untied
  non-mixer +0.098 acc) but does NOT "dwarf" the §9.2 legs on accuracy (an EM-floor artifact) — so `disruption`
  (MED structure: LOCAL rotation chains + NON-LOCAL bank cliques) joins the CROSS-CELL-COUPLED side alongside
  `converge`/`hopfield`.
  **STILL OPEN (low priority):** a stricter single-pass feedforward-mixer §4a (the untied stack already shows
  non-recurrent mixing suffices, so it would only reinforce M24e). (`mixed_converge` is now DONE — M28, a mixer
  WIN — so the mixer re-test covers every cross-cell-coupled synthetic task.)
  Canonical: `m24_mixer_{converge,iterated}_20260703T*`, `m24b_converge_ablation_20260703T102323_*`,
  `m24{c_hopfield,d_multiparity}_mixer_20260703T*`, `m24e_converge_mixer_untied_20260703T235935_*`,
  `m24f_disruption_{mixer_w24,mixer_w32,ablation_w24,ablation_w32,untied_w24,untied_w32}_20260704T*`; full narrative
  LOG.md M24 / M24b / M24c+M24d / M24e + correction.
- **★ M25 — the M24 cross-cell MIXER re-test carried to REAL multi-label tabular (`yeast`/`scene`/`emotions`):
  a CLEAN NEGATIVE (verdict SHARPENED by an adversarial review) — the mixer does NOT transfer; the M20 result
  HOLDS for the mixer architecture too.** M24 showed the mixer solves synthetic tasks wherever outputs are
  cross-cell COUPLED; real multi-label labels are coupled (yeast co-occurrence, scene mutual-exclusion), so this
  tests whether mixing crosses over where the flat loop tied (M20). 6-arm design (trm_mixer / trm_flat /
  ff_matched / untied_mixer_matched / untied_mixer ceiling / trm_decoupled), 10-fold CV, judged on
  **micro/macro-F1** (not EM — the M20 modal-combo caveat), M24e attribution controls, budget-clean. **Headline
  Δ(trm_mixer−ff_matched): the mixer NEVER beats the shallow MLP — it TIES on the PRIMARY metrics (micro-F1,
  accuracy) on BOTH large datasets: yeast miF1 −0.005 (5/5, p=1.0), scene −0.005 (4/6, ns).** The ONLY
  significant mixer deficit is the TIED mixer on yeast macro-F1/EM (maF1 −0.032, 1/9, p=.021) — and the
  distributed-padding rerun re-attributes it to **TYING**, not an artifact and not the architecture:
  Δ(trm_mixer−untied_mixer_matched) maF1 −0.028 (p=.021), while Δ(untied_mixer_matched−ff) is **ns on every
  metric** (the mixing architecture ties the MLP). Δ(mixer−flat) ns-to-neg (the mixing OPERATOR buys nothing;
  OPPOSITE of synthetic's +0.2…+0.9 EM). **NEW `Δ(trm_flat−ff)` M20-replication delta (sign-tested):** the flat
  loop TIES ff on F1 all three (yeast all ns; scene miF1/maF1 ns, EM-only +0.054 10/0; emotions p=1.0) — "the
  loop doesn't beat the MLP" reproduces recipe-independently. **Leg-1 (Δ(flat−decoupled) joint>binary-relevance
  on F1: yeast maF1 +0.034 10/0; scene +0.056/+0.055 10/0) is the one robust positive — but a plain joint MLP
  gets it too (ff>decoupled everywhere), so joint-MODELING not a loop/mixer property (exactly M20).**
  **★ ADVERSARIAL-REVIEW FIX (MAJOR, sharpened the negative):** the first yeast run's "significantly worse macro-F1
  −0.033, p=.002" was a PADDING ARTIFACT — contiguous 103→112 padding put all 9 zeros into mixer cell 13 = the
  RAREST label (base 0.014), a dead-input cell hitting macro-F1 only (verified; appeared in both mixer arms). Fix:
  DISTRIBUTE the pad one-per-cell (no dead cell) + re-ran yeast → micro-F1 −0.012→−0.005 (**clean p=1.0 tie**) and
  the architecture macro deficit VANISHED (untied_mixer_matched−ff maF1 −0.013 [p=.002] → −0.005 [ns]); the residual
  is the TYING cost above. **Mechanistic why it doesn't transfer:** synthetic mixer wins because input cells
  correspond POSITIONALLY to output cells (shared ring/grid/graph topology) so token-mixing propagates real
  constraints; the real-tabular reshape groups ARBITRARY feature-blocks into label-cells with NO input↔output
  correspondence, so the token-mix is just a differently-parameterized MLP. M24's "cross-cell dependency" is
  sharper than "outputs correlated" — it needs a SHARED input/output cell topology real tabular lacks. **Scope
  limit (review framing point):** the naive reshape handicaps the mixer by construction, so M25 shows "the NAIVE
  reshape doesn't transfer"; a learned input→cell projection is the untried steelman (future work). Closes the
  mixer arc. Caveats: emotions underpowered (593 rows, all p≥.11); mixer mildly under-fits train where it loses
  (yeast 0.993 / emotions 0.980 vs ff 1.0; scene both ~0.9998 and mixer still no win, so not merely optimization);
  multi-target coupled REGRESSION the one untested real-tabular FORM (§11c). Additive code
  (`pad_to_label_multiple`, DISTRIBUTED, off-by-default ⇒ all M0–M24 bit-identical; 5 padding tests, 256 total).
  Canonical: `m25_mixer_multilabel_{yeast_20260705T164426,scene_20260705T190349,emotions_20260705T201321}_*`
  (superseded contiguous-padding yeast `_20260705T120725` removed); full narrative LOG.md M25.
- **★ M26 — the mixer DOES transfer to real data: a clean POSITIVE on multivariate TIME-SERIES
  forecasting (`etth1` 7 vars, `weather` 21 vars), the §9.4 bridge's last untested FORM (multi-target
  REGRESSION with coupled targets) — the positive bookend to M25's multi-label null.** M25 failed
  because real multi-label's feature→label reshape had NO input↔output cell correspondence; forecasting
  has the SHARED cell topology (cell i = variable i on both the L-step-lookback input AND the H-step
  horizon), so the mixer's cross-variable token-mixing has real structure. **Δ(trm_mixer − ff_matched)
  MSE −0.089 (ETTh1, 9/1) / −0.188 (weather, 0/10, p=.002) — the mixer beats the shallow MLP on both,
  edge GROWING with variable count.** Honestly attributed (M24e/M24f controls): **primarily the mixing
  ARCHITECTURE not recurrence** — a non-recurrent budget-matched untied mixer captures the bulk
  (Δ(untied_mixer_matched−ff) −0.087 / −0.163), and the tied loop ≈ the 7.9× ceiling — with a
  WIDTH-DEPENDENT tying edge (P1 Δ(mixer−untied_mixer_matched) a tie −0.002 at 7 vars but SIG −0.025,
  0/10 at 21 vars — the M24f regularization-at-scale pattern). Plain recurrence does NOT do it (flat loop
  ties/only-modestly-beats ff; the win needs the MIXING operator, mixer−flat −0.097/−0.120 both sig).
  Leg-1 CD>CI: cross-variable coupling helps, SIG at 21 vars (Δ(flat−decoupled) −0.087, 9/1; channel-
  independent `trm_decoupled` is the WORST arm) — a partial counterpoint to DLinear/PatchTST "CI wins" at
  this tiny scale. It is a **GENERALIZATION win** (adversarial-review-verified): the mixer fits TRAIN
  worse but TESTS better while ff/flat OVERFIT (weather ff train 0.069 → test 0.390). Net (M23–M26 arc):
  **the mixer helps wherever there is cross-cell coupling AND a shared input/output cell topology —
  forecasting has it (M26), the naive multi-label reshape lacked it (M25).** NEW repo capability: the
  first REGRESSION path (`objective: regression` → MSE loss / MSE-MAE-R² eval / persistence baseline /
  expanding-window rolling-origin backtest; additive, off-by-default ⇒ all M0–M25 bit-identical). Honest
  scope: 2 datasets, ONE lookback/horizon (96→24), 25 epochs, standardized MSE, tiny CPU models; the
  sign-test p is INDICATIVE (disjoint test blocks but NESTED train sets ⇒ correlated Δs, Dietterich 1998).
  Canonical: `m26_etth1_forecast_20260705T212226_*`, `m26_weather_forecast_20260706T005434_*`; full
  narrative LOG.md M26.
- **M21 (latent/weight introspection) is DONE — and reframes the whole question: the trained loop does NOT
  settle a latent fixed point even where it WINS** (residual ~1.2, ρ>1, frac_expanding=1.0 on BOTH the
  `converge` win regime and the `iterated` fail regime; over-unroll readout collapses everywhere). This
  corroborates the (M9–M15c-established) STATIC joint-state-coherence reading of the loop's value; the
  strongest dynamical contrast is the loop's path-independence (za 0.97 vs 0.43, `trm` only — `trm_decoupled`,
  also a winner, shows a weak 0.55/0.32, so za tracks the target not winning) (§11(b) M21 bullet; LOG.md M21). The
  **evidence-gated follow-up** this licensed is now DONE (M27, below) — a `trm_stable` arm
  (Jacobian-spectral reg, DEQ 2106.14342 + a fixed-point/path-independence residual, Anil 2211.09961)
  that pushes ρ<1 — and it largely CLOSES the lever: contraction is achievable and does exactly what
  theory predicts (over-unrolling no longer decays) but buys ZERO test-time-compute GAIN and does not
  crack OOD depth — architecture-independently (flat AND mixer); its EM cost is a saturation effect, free
  when the model already solves the task (M27). The
  introspection suite is reusable anywhere (one line on real-tabular M20, or on a step-aligned-DS/progressive
  loop whose dynamics M21 did not probe).
- **M27 (the `trm_stable` contractive arm — the M21 lever) is DONE — a constructive NULL, and (after 2
  adversarial-review passes + a disentangling test) ARCHITECTURE-INDEPENDENT: contraction is achievable and
  FREE-when-solved, but buys no capability, on both the flat loop AND the mixer.**
  Built `train_stable` (standard train + a grad-enabled `torch.func.jvp` Jacobian penalty on the one-step
  latent map `F(z)=update(cat[X,z,readout(z)])`, + an optional fixed-point residual; all off-by-default ⇒
  M0–M26 bit-identical; `trm`/`trm_mixer`, param-identical to the plain arm). **(1) The manipulation is real, clean,
  monotone:** on `converge` rule 78 w24, `jac_reg_weight` drives ρ 1.264→0.654 (1e-2, the elbow)→0.114
  (1e-1), za→1.000, over-unroll drop 0.204→0.002 — the FIRST stable loop in the repo. **(2) But stability
  is BY CONSTRUCTION and buys nothing:** under over-unroll R'6→24 plain `trm` DECAYS (EM 0.632→0.059,
  the M8/M21 signature) while `trm_stable` HOLDS FLAT (EM ~0.52) — so Δ(trm_stable−trm) grows to +0.200
  acc @R'24 (8/0) but that is ENTIRELY trm decaying; trm_stable never RISES, so the "test-time compute
  benefit" hypothesis is **falsified even in the WIN regime**, and it loses to `ff_matched` at every R'.
  **(3) Contraction COSTS EM:** Δ(trm_stable−trm) EM −0.097 (8/0, p=.008); the cost appears at the
  contraction onset (1e-3/ρ=1.16 ~zero cost → 1e-2/ρ=0.65 cost). **(4) OOD depth null is CLEAN** (after
  an adversarial-review fix — see below): with a genuinely contractive weight on `iterated` (elbow 3e-2 ⇒
  ρ=0.610), OOD T=12/16 STILL collapse to baseline for every arm, and on the moving target contraction
  merely caps the fit (acc 0.883→0.736). **(5) MIXER re-test + disentangling test — the cost is SATURATION,
  not architecture.** Extended `train_stable` + the ρ diagnostics to the cross-cell `trm_mixer` and repeated
  converge: the WINNING architecture is EVEN MORE non-contractive (ρ=2.307 > flat 1.264, decays EM
  0.938→0.031) — the dynamical null is architecture-INDEPENDENT. The first-draft "contraction is FREE on the
  mixer → architecture-specific" headline was OVERSTATED (2nd review): the split (mixer Δ EM +0.007 ns vs
  flat −0.097) lives ONLY in EM — on ACCURACY the cost is ns for BOTH arms (flat Δacc −0.006 p=.07; mixer
  +0.00002 p=.29) — and EM tracks accuracy LEVEL (the M9 confound: flat base EM 0.63 vs mixer 0.94
  near-solved). A **disentangling sweep** (flat `trm`, w∈{8,12,16,24}, matched ρ≈0.65) SETTLES it: the flat
  loop's EM cost is ~0 at every saturating width (w≤16: +0.02/−0.01/+0.05) and the −0.097 appears ONLY at
  w=24 where it is not saturated — so the flat loop is *equally free when solved*, exactly as the mixer is.
  **Verdict: architecture-INDEPENDENT — contraction is achievable at ~zero accuracy cost, its EM cost is a
  saturation/operating-point effect (free when the model already solves the task, flat OR mixer), and it buys
  NO capability either way (flat under over-unroll = no test-time-compute gain; no OOD crack even when
  contractive ρ=0.610). The M21 lever is CLOSED architecture-independently; contraction buys ROBUSTNESS
  (step-count invariance), not extrapolation.** Upholds every depth-extrapolation null (M1/M3a/M7/M8).
  **Adversarial review (TWO passes) corrected FOUR overstatements:** (i) "coherence win to trade" FALSE —
  Δ(trm−ff) EM a tie (−0.011, p=.73), cost is vs trm's OWN baseline (not M9's +0.133); (ii) first iterated
  run left ρ=1.026 (still expanding) — re-screened to 3e-2/ρ=0.610, re-run (superseded run removed); (iii)
  "did we check the mixer?" — extended to `trm_mixer`; (iv) the mixer "free = architecture" headline was an
  EM/saturation confound (accuracy cost ns for both), resolved by the disentangling sweep (+ an acc-mislabeled-
  as-EM number fixed). Code review (both passes): jac map faithful, penalty grad-enabled, off-by-default
  bit-identical, over-unroll valid for the mixer; loud guard added (contraction + `use_act`/`n_sup` raise).
  Scope: converge + iterated, jac penalty only; single-knob. Canonical: `m27_stable_{screen,converge,
  iterated_screen,iterated_extrap,mixer_converge,flat_saturation}_20260706T*`; full narrative LOG.md M27.
- **The real-tabular bridge gave the loop a fair shot and it did not transfer (M20) — now CONFIRMED on a
  3rd dataset.** `scene` (6 labels, mutual-exclusion) replicated the corrected verdict exactly: leg-1 robust
  10/0 on EM+F1 but ff also beats decoupled (not loop-specific); Δ(trm−ff) EM-only, a 5/5 F1 tie; and it
  resolved emotions' null as sample-size (scene = 6 labels + large n fires leg-1). So the bridge is a **clean
  negative on TWO large datasets with opposite coupling** — the loop's synthetic coherence value does not
  cross to real multi-label tabular. **The `hidden=128` capacity probe is DONE and NEGATIVE on both datasets**
  (Δ(trm−ff) on F1 stays a tie; the M11 grows-with-size lever fails on real data), so the multi-label-
  classification verdict is now FINAL across data, coupling regime, AND capacity. **And the M25 cross-cell
  MIXER re-test (the M23/M24 architecture that overturned the synthetic negatives) is DONE and ALSO NEGATIVE
  on real multi-label** — `trm_mixer` TIES `ff` on the primary metrics (micro-F1/accuracy) on BOTH yeast and
  scene; no mixing-operator or mixing-architecture edge, and tying is a small rare-label macro-F1 DEFICIT on
  yeast (an adversarial review caught+fixed a padding artifact that had inflated it) (§11(b) M25 bullet; LOG.md
  M25). So the multi-label negative spans the FLAT loop AND the mixer. **The other real-tabular FORM —
  multi-target REGRESSION with coupled targets — is now DONE (M26) and is a POSITIVE for the MIXER (not
  the flat loop):** on multivariate time-series forecasting (`etth1`/`weather`) the cross-variable mixer
  significantly beats the shallow MLP on MSE (−0.089/−0.188), because forecasting supplies the SHARED
  input/output cell topology multi-label lacked (§11(b) M26 bullet; LOG.md M26). So "loop-on-tabular" is
  NOT a blanket negative: a mixing ARCHITECTURE transfers where the task has cross-cell coupling + shared
  topology; the flat loop and plain multi-label do not. Eval machinery reusable: classification
  (`multilabel_f1`, K-fold CV, `pad_to_label_multiple`) and regression (`objective: regression`, MSE/MAE/R²,
  expanding-window backtest). **Open forecasting threads (M26 scope limits):** a horizon sweep (96→
  {96,192,336,720}), more datasets (ETTh2/ETTm/electricity/traffic), and benchmark-scale models where
  channel-independence is known to win (the CD>CI finding may not survive) — none run yet.
- **DO NOT build M19 (H/L) yet — the gate is unmet (M18g).** The §9.3 build-gate required the single-timescale
  loop to be insufficient on the nested target *in a timescale-specific way*. At equal compute (M18g, 400
  epochs, all train_acc≈1.0) the single loop is the best arm but stays far below target at every capacity
  (EM 0.75 @ h64 → 0.79 @ h128), with a feedforward just behind — a generic capacity/generalization wall,
  not a timescale deficit. And **M18j settles it: at 64k data the single loop SOLVES the target (0.99 EM)**
  — a pure sample wall, triggering §9.3's null clause. Building H/L now would repeat the HRM mistake.
  **Re-gate first:** find a nested instance (vary inner/outer rule, nesting depth, block size, n_train,
- **DO NOT build M19 (H/L) yet — the gate is unmet (M18g).** The §9.3 build-gate required the single-timescale
  loop to be insufficient on the nested target *in a timescale-specific way*. At equal compute (M18g, 400
  epochs, all train_acc≈1.0) the single loop is the best arm but stays far below target at every capacity
  (EM 0.75 @ h64 → 0.79 @ h128), with a feedforward just behind — a generic capacity/generalization wall,
  not a timescale deficit. And **M18j settles it: at 64k data the single loop SOLVES the target (0.99 EM)**
  — a pure sample wall, triggering §9.3's null clause. Building H/L now would repeat the HRM mistake.
  **Re-gate first:** find a nested instance (vary inner/outer rule, nesting depth, block size, n_train,
  model size) where the loop plateaus below target **and ff/untied do NOT share the ceiling at equal
  compute AND more data does NOT close it**; only then is M19 earned. On the current instance the lever is
  simply data (M18h/M18j). **M28b now also rules out the mixer ARCHITECTURE as a re-gate:** the cross-cell
  `trm_mixer` improves the nested operator (Δ(mixer−ff) EM +0.076/+0.130 @4k, 8/0) but does NOT solve
  (EM 0.79/0.70 ≪ 1.0), and its NON-recurrent control shares the ceiling (untied_mixer_matched gives the
  whole ff edge; the 7× untied_mixer ceiling matches/beats the tied mixer; tying NS) — so even the mixing
  architecture shows a shared capacity/data wall, not a recurrence/timescale-specific insufficiency. The
  mixer does NOT re-open Task C.
- **DONE — M17 built the Task C substrate + ran the gate (parallel branch, integrated here); the gate verdict
  was OVERTURNED by M18g.** `make_nested_converge` (two-timescale fixed point) is built, screened, tested —
  solid, reusable infrastructure. M17's "gate MET → M19 earned" conclusion does NOT hold: it compared a
  4×-compute loop to 1×-compute controls; at equal compute a feedforward shares the loop's ceiling (M18g). The
  durable nested findings are leg-1 (joint-state) and P1 (tying), which reproduce at equal compute. Full
  narrative in LOG.md M17 + M18g.
- **DONE — TRM-faithful ingredients added + tested, then the headline WALKED BACK by the adversarial
  review (M18, this branch).** A 2024–26 literature scan found the repo's `TRM` lacked four ingredients the
  looped-model work flags (TRM ablations / HRM autopsy); all four added additively (off-by-default,
  bit-identical; `train_deep_supervision`, `use_rmsnorm`, `n_latent`, `ema_decay`, plus the `n_sup_carry`
  compute-matched control). The pre-review headline ("canonical detached deep supervision is a large win")
  was **a compute confound**: M18e (no-carry control) + M18f (plain 4×-epoch loop) show the apparent
  +0.295 EM N_sup win is **just more optimization** on an undertrained convergent target — the detached
  carry mechanism adds +0.012 EM (ns), and a plain 4×-epoch loop matches it (EM 0.873 ≈ 0.879). So the
  autopsy's DS *mechanism* is ~inert (upholds the prior "DS is inert" verdict); recommendation is the
  mundane **train loops longer on convergent targets**, NOT "adopt the faithful bundle." The bundle is
  INERT on the non-convergent depth wall (M18a). **Cross-checks on the Task C gate (M18d→M18g):** M18d
  applied the more-optimization lever to M17's nested gate (single loop → 0.82, still ≪ 1.0); the 2nd
  adversarial review then asked the decisive question, and **M18g answered it — at EQUAL compute a
  param-matched feedforward SHARES the single loop's nested ceiling (Δ(trm−ff) EM +0.036/−0.003), so the
  "single-timescale insufficiency" is a generic capacity wall, NOT a timescale deficit → the M17 gate FAILS
  and M19 is NOT earned** (§11(b), §9.3). The faithful machinery + the `n_sup_carry` control remain as
  infrastructure and a cautionary §8 case study. Lowest-value leftovers: M18h (does the shared ceiling lift
  with data?); the §9.4 real-tabular bridge.
- **DONE — the §9 gate has been rewritten (M16, this branch).** The unsatisfiable "beats both on A and B"
  gate is retired/falsified (M6a) and §9 is reframed around the actual finding (joint-state coherence on
  local-update hard fixed-point targets; legs 1/2 + P1, precisely scoped) with Task C re-imagined as
  `nested_converge` and gated on a *satisfiable* within-loop criterion (§9.3). The substrate was built (M17)
  but the gate FAILED its equal-compute control test (M18g — M19 unearned, above); the remaining frontier is **(b)** the separately-scoped
  real-tabular bridge (§9.4).
- **The experimental program is COMPLETE — all leg-2 confounds now controlled.** M14 closed locality;
  M15 established leg 1 (joint-state mechanism = deep+local, transfers off-CA, clean); M15b max-depth-matched
  leg 2; **M15c closed the central-depth residual** (depth-distribution-matched: leg 2 confirmed
  depth-controlled on rule 13, +0.21 EM 10/0 with depth held identical to the non-uniform mixed task and
  hardness against it; rule 78 shown depth-inflated/inconclusive). The only un-removable leg-2 caveat is the
  DEFINITIONAL uniformity↔rule-cardinality entanglement (a non-uniform local rule must use ≥2 truth tables).
  Lowest-value leftovers only: more uniform rules at matched depth (rule 78 went ff-easy — try a harder
  matched rule); finer/larger size sweep; radius-2 mix; the operator-sharing *why*. **The §9-gate rewrite is
  DONE (M16); the Task C substrate is built (M17) but its build-gate FAILED the equal-compute control test
  (M18g — M19 NOT earned, Task C re-deferred);** the next genuine frontier is the **§9.4 real-tabular bridge**
  (the synthetic Task-C thread is parked pending a re-gate, above).
- **Closed levers (do not redo):** depth-extrapolation via progressive loss / path-independence (M7/M8 —
  decay is intrinsic, not convergence-related); adaptive compute on a fixed-point target (M8 — decays);
  "lift the M4 sample wall" (M5); "re-judge via a both-axes task" (M6a); the decoupled-head mechanism
  question (M10 — coherence is the joint state); **the model-size axis (M11 — generalizes & strengthens);
  the rule/operator generality within ECAs (M11/M12 — family-specific to hard-convergence = two symmetry
  orbits, all 8 members confirmed, no other balanced+deep ECA exists); LEAVING the ECA family (M13 — the
  joint-state result is CA/local-update specific, does not transfer to a dense threshold net at any size);
  the LOCALITY hypothesis (M14 — a local-but-non-CA threshold net is ff-easy and the loop loses; the
  M8–M12 edge is NOT explained by coupling locality); the UNIFORM-RULE-vs-DEPTH question (M15 — a
  per-position MIXED CA, deep+local+ff-hard but non-uniform: leg 1 [joint-state mechanism = deep+local]
  TRANSFERS to the mixed task and is CLEAN/within-task; leg 2 [loop-beats-the-MLP needs the uniform rule]
  STRONGLY SUPPORTED by **M15b** — a max-depth-matched uniform control [`accept_max_depth` cap, uniform CA
  through the mixed pipeline]: loop beats ff +0.090/+0.175 EM 9/1 p=.021 on uniform rules 78/13, ties on
  the non-uniform mixed task at matched max-depth, with the hardness gradient running against the result.
  M15c CLOSED the central-depth residual via depth-distribution matching: leg 2 confirmed depth-controlled
  on rule 13 (+0.21 EM 10/0, depth identical to mixed, hardness against), rule 78 inconclusive at matched
  depth [ff-easy]. Only the definitional uniformity↔rule-cardinality entanglement is un-removable. Do not
  re-conflate the two legs).**
  A bigger-model probe of d=80,k=5 must scale the budget for *all* arms (M5).
  **The DEEP-SUPERVISION MECHANISM on the mixer (M29 — SETTLED after two adversarial reviews: M18's "the
  detached-carry MECHANISM is inert; the gain is just more optimization" HOLDS on the mixer. Inert where the mixer
  FITS train [all convergent cross-cell tasks]; and where it UNDER-FITS [`iterated` T6, the only such regime], the
  carry is STILL inert [a 2-seed ceiling rescue, p=0.125] while the compute [no-carry batch-repetition] carries the
  gain — exactly M18. The intermediate "m29c reversal" was a ceiling-saturation artifact, retracted by review-2.]
  Do not re-run the m29a/m29b/m29c decomposition. Genuinely OPEN [low priority]: a CONVERGENT fixed-point task the
  mixer under-fits [none found — the mixer fits them all], to test the carry in its motivated regime rather than
  the non-convergent T6 proxy.**

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
  Also the source of the **latent "orbits"** observation our M21 introspection echoes (a shared
  init-independent limit set that never settles to a fixed point).

### Introspection / stability diagnostics (the M21 toolkit + the architectural-refinement levers)

These ground the M21 measurement-only suite and the (unbuilt, evidence-gated) `trm_stable` follow-up.

- **Stabilizing Equilibrium Models by Jacobian Regularization** — Bai, Koltun, Kolter 2021,
  arXiv 2106.14342. Jacobian spectral radius ρ(J) is the stability condition for a fixed-point
  iteration (ρ<1 ⇒ contraction); regularizing it (Hutchinson trace estimator) stabilizes DEQ
  training. The headline M21 metric and the first refinement lever.
- **Path Independent Equilibrium Models** — Anil, Pokle et al. 2022, arXiv 2211.09961. Already cited
  above for the diagnostic; *also* the intervention lever — training that promotes path independence
  improves upward generalization, training that penalizes it hurts. The M21 `za_alignment` is its
  Asymptotic-Alignment score.
- The Jacobian-spectral / Lipschitz refinement argument rests ENTIRELY on the three verifiable refs
  above (2106.14342, 2410.23451, 2211.09961) — do not let it depend on any single citation.
  A newer looped-LM template, **"STARS / Stabilizing Recurrent Dynamics…" (arXiv 2605.26733)**, was
  found via web search (Jacobian-Spectral-Radius Regularization + random-loop sampling; pre-norm
  "grow without bound" vs post-norm "settle into poor states"); it is an OPTIONAL pointer whose ID is
  a 2026 arXiv number this repo cannot verify offline — cite it as supplementary, not load-bearing.
