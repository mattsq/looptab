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
scale with difficulty) but did NOT unlock the canonical hard-solving win — §11.2 / `results/log/m23.md`.**

**⭐ DEFAULT ARCHITECTURE = the CROSS-CELL MIXER (`trm_mixer`), NOT the flat `trm`.** The single most
load-bearing finding of the whole program (M23 Sudoku → M24 rings → M26 forecasting): a weight-tied loop
whose update is a *flat MLP over the concatenated grid* is **≈ a feedforward net** — it cannot let cells
communicate, so "does iterative refinement help?" was systematically confounded with "does the update
operator match the task structure?" for M0–M22. The thing that makes TRM *work* — constraint propagation,
the hard-solving win — **is the cross-cell token-mixing update**, not the
loop per se. **(EXCEPTION — real forecasting: M31/M32 showed the M26/M30 `etth1`/`weather` mixer win is NOT the
mixing update; mixing is net HARMFUL there. The win is CHANNEL-INDEPENDENCE first (per-cell own-variable
processing, dominant when channels are many) + a modest shared-readout second. The constraint-coupled
SYNTHETIC wins below stand; forecasting is on the channel-INDEPENDENT side — §11.2 #13–#14.)** So **start every new recurrent experiment from `trm_mixer`; use the flat `trm` only as the
COMPARISON control** (the §4b "is it the mixing operator or just recurrence?" ablation — flat `trm` ≈ ff on
structured tasks, so `Δ(trm_mixer − trm_flat)` is the operator's contribution). Do NOT default to flat
`trm` as a "simplest starting point" — a null measured on it is a null about a feedforward-equivalent, not
about refinement (the §8 trap, one axis deeper). **Caveats that bound the default (don't misapply it):**
(1) `trm_mixer` is **multi-output only** and requires **`in_features % out_features == 0`** — so distractor
columns are unsupported (use `distractors: 0` or `pad_to_label_multiple`); single-output tasks (e.g.
scalar-`y` parity) have no mixer, use flat `trm` there. (2) The mixer helps only where outputs are
**cross-cell COUPLED with a shared input/output cell topology** (rings/grids/graphs) — it is
inert on **exchangeable** features (`multi_parity`, M24d), and on real **forecasting** the mixer's *apparent*
win is channel-independence + shared readout, not coupling — mixing there is net HARMFUL (M31/M32, §11.2
#13–#14), and on the **naive multi-label reshape** (no
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
  coherence value does not cross to real tabular as a *loop* property (§11.2 / `results/log/m20.md`
  "M20 — PROPER EVALUATION"). Confirmed on TWO large datasets with opposite coupling (`yeast` co-occurrence, `scene`
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

This is the hot-context status page. Full per-milestone evidence (tables, readings,
caveats) now lives in **`results/LOG.md`** (a skimmable index) and **`results/log/*.md`**
(one narrative file per milestone). Reference notes are in **`docs/REFERENCES.md`**. Keep
this section terse: current substrate, the behaviour-changing conclusions with their
load-bearing caveats, and what's open vs closed. Append new detail to a `results/log/*.md`
file and one index row, not here.

### 11.1 Current substrate — what exists, where

- **Tasks** (generators in `src/looptab/data/generators.py`; real data in
  `src/looptab/data/real.py`; determinism/validity tests under `tests/`):
  `linear` (smoke), `parity` (Task A), `iterated` (Task B CA), `multi_parity` (both-axes
  probe), `converge` (fixed-point CA), `hopfield` (dense/`bandwidth`-local threshold-net
  attractor), `mixed_converge` (per-position non-uniform CA), `nested_converge` (Task C
  two-timescale fixed point), `disruption` (airline ops threshold net), `sudoku` (the one
  multi-CLASS task — forced `num_classes` inference in `run.py`), real `multilabel`
  (`emotions`/`yeast`/`scene`, EM = subset accuracy), and real forecasting `etth1`/`weather`
  (multivariate time series = coupled multi-target REGRESSION).
- **Models/arms** (`src/looptab/models/`, registered in `registry.py`): **⭐ default recurrent
  arm = `trm_mixer`** (cross-cell token-mixing loop — see §4); **flat `trm`** is the
  comparison control (feedforward-equivalent on structured tasks); `ff_matched` (§4a),
  `untied_stack` (labelled ceiling), `untied_matched` (clean §4b tying control),
  `trm_decoupled` (per-cell / joint-state ablation), and the mixing-matched controls
  `untied_mixer` (ceiling) + `untied_mixer_matched` (clean tying control), and `trm_mixer_nomix`
  (M31 shared-readout control: `TRMMixer` with `disable_token_mix=True` — shared readout, cross-cell
  mixing REMOVED; off-by-default flag ⇒ `trm_mixer` byte-identical), and the M32 ingredient-
  decomposition controls `trm_mixer_nomix_unsharedro` / `trm_mixer_unsharedro` (`shared_readout=False`
  block-diagonal readout) + `trm_mixer_nomix_distinctw` (`distinct_cell_weights=True` per-cell channel
  MLPs) — all off-by-default `TRMMixer` flags ⇒ byte-identical at defaults. M18 TRM-faithful
  knobs (`use_rmsnorm`, `n_latent`, `ema_decay`, `n_sup`) + `trm_faithful` arm; all
  off-by-default ⇒ bit-identical.
- **Training/eval** (`src/looptab/train/`, `src/looptab/eval/`): routines `train`,
  `train_curriculum`, `train_progressive`, `train_deep_supervision` (detached-carry DS —
  runs on `trm`/`trm_decoupled` and, since M29, `trm_mixer`), `train_act` (ACT halting,
  M23), and `train_stable` (M27 contractive: `jac_reg_weight`/`fixed_point_weight` Jacobian
  penalty). Deep supervision is a per-arm weight, not a global flag. Metrics: `accuracy`,
  `exact_match`, `coherence_excess`, `majority_baseline`, `multilabel_f1` (micro+macro) +
  K-fold CV, and regression `evaluate_regression` (MSE/MAE/R²) + `persistence_baseline_mse`
  gated by `objective: regression` (M26). `delta_report` = paired Δ + variance + sign test,
  with a ceiling-tie guard (`eps`/`near_tie_eps` + `sign_test_robust`, M29c). Optional M21
  introspection side-car (`eval/introspection.py`, off by default). Runner
  `python -m looptab.run --config <yaml>` (sweep / grid / extrapolation; pins `num_threads=1`
  — mandatory for the 3-D matmul arms `trm_decoupled`/`trm_mixer`/`train_stable`).
- **Configs/results:** experiment configs in `configs/experiments/`; tracked summaries in
  `results/`; milestone narratives indexed by `results/LOG.md`. `pad_to_label_multiple`
  (M25, off by default) right-pads X so `d % L == 0` for the mixer on real data.

### 11.2 Behaviour-changing conclusions (read before re-running anything)

1. **Report Δ, not lone scores** (§2). Every recurrent claim ships against its matched
   control(s), ≥5 seeds, with variance/sign info. **Use `untied_matched` (not `untied_stack`,
   which has ~4× params) as the clean §4b control** — a Δ against the stack confounds tying
   with capacity (the §8 trap that flipped M2's first verdict).
2. **The original "beat both controls on A and B" gate is RETIRED** (M6a): structurally
   unsatisfiable for a weight-tied generalist judged against single-axis specialists at fixed
   budget. Do not re-run to satisfy it. Task C is now gated on a *within-loop* criterion (§9.3).
3. **Task A parity: depth helps, tying is neutral** (M4). The loop and the untied stack solve
   hard parity where a shallow MLP fails; the edge grows with k but is DEPTH, not tying (ties
   `untied_matched`, loop-beats-both in 0 cells). The d≥40 wall is SAMPLE-bound and lifts to
   all-arms-solve (M5), except d=80,k=5 which is capacity-bound.
4. **Task B iterated CA: no transferable step operator; no depth-extrapolation** (M1/M3a/M3b/
   M7). Over-unrolling decays and OOD depth T>T_train collapses to baseline for every arm, even
   on a convergent target (M8) — the decay is intrinsic to learned depth, not task
   non-convergence. Deep CA is unlearnable one-shot at T≥8 for all arms (a learnability wall).
   The loop's only Task-B edge is a shallow-depth parameter-efficiency win over a fair untied
   stack, gone by T≥8.
5. **Deep supervision is protocol-dependent, not inert** (M3b): final-state per-step DS is
   neutral-to-negative; step-aligned DS wins at SHORT horizon (+0.162 EM @ T=4) but flips
   negative by T≥8. Progressive loss is inert (M7/M7b). The canonical detached-carry DS
   "mechanism" (M18) is **just more optimization** — the +0.295 EM N_sup gain is matched by a
   plain 4×-epoch run and a no-carry control; the carry adds ~0 (upholds "DS mechanism inert").
   Lesson: **train loops longer on convergent fixed-point targets**, don't reach for the bundle.
6. **The loop's one durable synthetic positive is JOINT-STATE COHERENCE on local-update hard
   fixed-point targets** (§9.2), decomposed and scoped M8–M15c:
   - **Leg 1 (joint-state, CLEAN):** refining all cells in one shared latent ≫ independent
     latents (`trm` ≫ `trm_decoupled`, ΔEM up to +0.66, 10/0, GROWS with size M11). Needs
     *deep + local*; null on a dense net (M13) and a shallow one (M14); transfers off-CA to a
     non-uniform local map (M15).
   - **Leg 2 (loop-beats-the-MLP, EM-only, w≤24):** holds only for a UNIFORM local rule,
     depth-controlled on rule 13 (+0.21 EM, 10/0, M15c); absent on a per-position-mixed rule.
   - **P1 (tying-positive, broadest):** tied loop beats a fair width-matched untied stack on
     coherence, width-robust (M9), survives off-CA (M13/M14). The one regime-independent
     pro-loop fact.
   Bounded to **CA/local-update hard-convergence** (= exactly two ECA symmetry orbits, M12) —
   NOT hard-convergence fixed points in general (M13/M14), NOT depth-extrapolation, NOT
   adaptive compute, NOT capacity-independent.
7. **★ The single most load-bearing finding: the flat weight-tied loop ≈ feedforward; the
   thing that WORKS is the cross-cell MIXING update** (M23→M29). A flat `trm` update over the
   concatenated grid cannot let cells communicate, so M0–M22 systematically confounded "does
   refinement help?" with "does the update operator match the task?". `trm_mixer` (token-mixing
   = constraint propagation) SOLVES hard Sudoku a param-matched MLP cannot (Δ EM +0.89…+0.96,
   growing with difficulty — TRM's real signature) where `trm_flat ≈ ff`. **But the win is the
   ARCHITECTURE, not recurrence:** a budget-matched *untied* mixer matches the tied loop
   (M24e/M24f/M26) — weight-tying adds only parameter EFFICIENCY (P1), no capability an untied
   mixer of equal capacity lacks (mirrors M21's "dressed-up depth").
8. **The mixer helps iff outputs are CROSS-CELL COUPLED *and* there is a SHARED input/output
   cell topology** (M24–M28) — the sharp version of the dividing line:
   - Fires on synthetic coupling: `converge` (win grows with w, SOLVES it — M24), dense
     `hopfield` (overturns the M13 flat-loop null — M24c), `disruption` (overturns the M22
     leg-2 null — M24f), `mixed_converge` (M28). Inert on exchangeable `multi_parity` (M24d).
   - Does NOT crack depth: on non-convergent `iterated` the mixer improves the per-step
     operator (T=4) but the M3a depth wall reasserts at T≥8 (M24).
   - Cross-cell coupling alone is not enough — it needs positional input↔output correspondence
     (why M26 forecasting *appeared* to transfer but M25 multi-label did not; see 9/11 below).
   - **Correction (M31, §11.2 #13): the M26/M30 forecasting "transfer" is NOT the mixing operator —
     it is the mixer's shared per-cell readout; at a held shared readout, mixing is inert-to-harmful on
     real forecasting (channel-INDEPENDENT regime). The synthetic coupled wins in this list stand.**
9. **Real multi-label classification is NEGATIVE for the loop AND the mixer** (M20/M25). Under
   micro/macro-F1 + 10-fold CV on `yeast`/`scene`: joint output modeling beats binary-relevance
   (leg-1, robust) **but a plain joint MLP gets it too** — not a loop property. Every
   loop/mixer-SPECIFIC edge is EM-only and TIES under F1 (the EM "win" was a modal-label-combo
   artifact). `trm_mixer` ties `ff` on micro-F1/accuracy on both datasets; the naive
   feature→label reshape has no shared input/output cell topology, so token-mixing is just a
   reparameterized MLP. Capacity (hidden=128) does not rescue it (falsifies the M11 lever on
   real data).
10. **Real coupled multi-target REGRESSION (forecasting) is POSITIVE for the mixer, and it is
    HORIZON-ROBUST** (M26 + M30; M30 adversarial-review-hardened). On `etth1`/`weather` the
    cross-variable mixer beats a param-matched MLP on MSE in the mean at every horizon, because
    forecasting supplies the shared cell topology (cell i = variable i on input AND output). **M30
    swept horizon {192,336,720} (budget-clean, mixer re-widened per horizon since the flat M×H
    readout balloons with H but the mixer's shared readout doesn't).** The win does NOT decay; on
    **weather it also grows** with horizon (direction-robust by median, Δ(mixer−ff) median
    −0.19→−0.31→−0.47), **but read the growth carefully:** the raw-mean progression (−0.99 at h720)
    is ~2× inflated by ONE hard backtest block, and on **etth1 the mean is non-monotonic** — so
    "amplifies" is weather-only, not both-dataset. Primarily ARCHITECTURE not recurrence (untied
    mixer carries it; tying is a variance-swamped TIE, **not** a significant reversal — the etth1-h720
    +0.074 is 7/3 ns) — **BUT this is CONFOUNDED with shared-readout parameter-efficiency: both mixer
    arms also have a shared `Linear(latent,H)` readout while the flat/ff arms carry an unshared M×H
    readout that horizon inflates; no M30 arm separates mixing from shared-readout** (a shared-readout
    non-mixing MLP is the needed control — **RUN in M31; it MIS-ATTRIBUTES this whole finding, see #13**).
    Most robust finding = GENERALIZATION: the shallow joint MLP
    overfits worse than persistence on 10/10 weather-h720 blocks, while every cross-variable arm holds.
    **CD>CI persists in DIRECTION but LOSES M26's significance** (weather 9/1 p=.021 at h24 → 2/8 ns);
    the one CI "divergence" (weather-h720 train mse ~91000 on 1 block) is a plausible optimization/seed
    artifact, NOT demonstrated CI instability — so DLinear/PatchTST "CI-wins-at-long-horizon" is
    neither reproduced nor refuted here (tiny CPU models; the scale where CI wins is larger). Scope: 2
    datasets, one lookback (96), tiny CPU models; sign-test p indicative (nested train sets) — the
    story is sign-consistency + median/trimmed growth, not a single p or a single hard-block mean.
    **Still open: more datasets + benchmark-scale (§11.3); the shared-readout control is DONE (#13).**
11. **The trained loop does NOT settle a latent fixed point even where it WINS** (M21): latent
    residual ~1.2, ρ>1, over-unroll collapses — "dressed-up depth" MEASURED. The win-vs-fail
    contrast is path-independence (za 0.97 vs 0.43, `trm` only), not contraction. **Making it
    contractive (M27 `trm_stable`) is achievable but buys NOTHING** — over-unroll no longer
    decays, but no accuracy / no test-time-compute gain / no OOD crack; architecture-independent
    (flat AND mixer), and its EM cost is a saturation effect (free when the model already
    solves the task). Contraction buys ROBUSTNESS (step-count invariance), not extrapolation.
12. **Task C / H-L hierarchy (M19) remains UNEARNED** (M17/M18g/M28). `nested_converge` is
    built, but at equal compute the single-timescale loop's shortfall is a SAMPLE/generalization
    wall shared by a feedforward AND by the untied mixer — not a timescale deficit (the loop
    SOLVES it at 64k data, 0.99 EM). Building H/L now would repeat the HRM mistake the ARC
    autopsy diagnosed. Do NOT build M19 until a gate shows a timescale-*specific* single-loop
    insufficiency (§9.3 re-gate condition).
13. **★ The forecasting mixer win (#10) is NOT cross-variable MIXING — it's the shared-readout /
    channel-independent parameterization** (M31 — runs M30's named control). `trm_mixer_nomix` =
    `TRMMixer` with token-mixing REMOVED but the per-cell SHARED `Linear(latent,H)` readout KEPT,
    re-widened to match `trm_flat`'s budget. It decomposes the M30 headline Δ(mixer−ff) =
    Δ(mixer−nomix)[mixing] + Δ(nomix−ff)[shared-readout], and the FIRM (negative) verdict is decisive
    across all 6 cells (etth1/weather × {192,336,720}, all arms within ±5% budget tol): **(a) token-
    mixing at a held shared readout is NOT the mechanism — it adds NOTHING and is at most weakly HARMFUL
    (weather only)**: Δ(mixer−nomix) MSE is mean-positive everywhere (+0.03→+0.16), sign-significant
    *against* mixing on all 3 weather cells (10/0, 9/1, 10/0, p≤.021) but NULL on etth1 (8/2, 5/5, 8/2);
    "worsens with horizon" is a mean artifact — by MEDIAN it is non-monotone (weather 0.06→0.14→0.12);
    **(b) the ENTIRE win is the shared-readout/channel-independent parameterization** — Δ(nomix−ff)
    negative in every cell (all 0/10–1/9, p≤.021), growing with horizon on weather (MEDIAN
    −0.24→−0.43→−0.65; the raw-mean −0.44→−0.59→−1.13 is ~1.7× inflated at h720 by ONE `ff` outlier block
    at MSE 6.60 — the same single-block inflation M30 flagged about itself; read the median); **(c)
    `trm_mixer_nomix` (channel-INDEPENDENT) is the BEST arm in every cell** (mean), beating
    mixer/flat/ff/persistence — mild evidence FOR the DLinear/PatchTST CI direction #10 could not settle.
    Mechanically obvious: the unshared M×H readout is ~47% of `trm_flat` at weather-h720 and balloons
    with M·H, starving the flat/ff arms' width at fixed budget, while the shared readout costs ~139k with
    no M factor. **Scope of the correction: FORECASTING ONLY.** The synthetic constraint-coupled mixing
    wins (#7–8: Sudoku ΔEM +0.89, `converge`/`hopfield`/`disruption`) are UNTOUCHED — this sharpens #8's
    dividing line (mixing helps only where outputs are genuinely constraint-coupled; real forecasting is
    on the channel-independent side). Caveats (do not overclaim): the mixing leg is budget-matched to
    `trm_flat` NOT `trm_mixer` (channel width differs ≤1.6%; a `nomix`-matched-to-`trm_mixer` follow-up
    would make it byte-clean), and Δ(nomix−ff) BUNDLES shared-readout + channel-independence + per-
    variable weight-sharing — "it's the shared readout" means that whole cheap-parameterization package,
    not the readout matrix alone. Reproduction is to ROUNDING not bit-identical: `trm_flat`/`ff` match
    M30 per-seed, but `trm_mixer` weather-h720 drifts ~2.6e-3 (3-D-matmul non-determinism at max width).
    **M32 (#14) SPLITS that bundle and CORRECTS the emphasis: it is channel-INDEPENDENCE first, the
    shared readout a modest second — "it's the shared readout" (part b, readout named first) is too
    specific; and mixing is not merely "weakly harmful" but net HARMFUL in all 6 cells.**
14. **★ Decomposing #13 (adversarial-review-hardened): on the MANY-CHANNEL dataset channel-INDEPENDENCE
    is the larger slice, the shared readout a modest second; token-mixing is NOT the mechanism (net
    harmful on weather); the split is DATASET-dependent, not universal** (M32 — three off-by-default
    `TRMMixer` controls split the #13 bundle; `ff`/`nomix` reproduce M31 bit-identically, `trm_flat` in
    5/6 cells). Isolating each of the three things `nomix` has and `ff` lacks, plus mixing, each flipping
    one named axis (+ a forced re-width to hold `trm_flat`'s budget — not a single-*parameter* change):
    **(a) token-mixing does NOT help; on weather it is net HARMFUL** — Δ(mixer−nomix) MSE positive in all
    6 cells (+0.03→+0.16), **decisive on weather** (10/0, 9/1, 10/0; p≤.021) but **NOT significant on
    etth1** (8/2, 6/4, 8/2; the h336 6/4 is a coin-flip), and NOT monotone in horizon (weather median
    non-monotone, per M31). Firm claim is the NEGATIVE one ("not the mechanism"); "harmful" is weather-
    only. **(b) the shared readout is a REAL, sign-consistent, MODEST helper** — Δ(nomix−nomix_unsharedro)
    negative in all 6 (−0.03→−0.10), decisive on weather; the more consistent axis on etth1 — NOT the
    majority ingredient #13's naming implied. **(c) channel-INDEPENDENCE is the larger structural slice
    when channels are MANY:** vs the fair `trm_flat` baseline it carries **76–84% on weather (M=21)** but
    only **~9–44% on etth1 (M=7), where it is small and noise-dominated** (a ~0.009 `trm_flat` baseline
    wobble alone swings the etth1-336 share 15%→9%). **(d) weight-sharing a modest third** (Δ(nomix−
    distinctw) −0.03→−0.09, all 6). **(e) most of #13's raw `nomix−ff` number is ff-OVERFIT, not an
    ingredient** — `trm_flat−ff` is 81% of the weather-h720 headline; the ingredients partition only the
    minority RESIDUAL after removing ff's pathology, so "CI leads" means it leads that residual on
    weather, not that it explains the win over ff. Net: #13's compound name is a real COMPOUND (both parts
    contribute, same sign in every cell), its EMPHASIS (readout first) corrected toward channel-
    independence ON THE MANY-CHANNEL SIDE — not a universal reordering. FORECASTING ONLY; synthetic mixing
    wins (#7–8) untouched — the channel-independent side of #8's line. Caveats: the CI-structure contrast
    (`nomix_unsharedro − trm_flat`) is NOT a single-knob flip — it bundles CI + readout-STRUCTURE + width
    (the clean CD/CI flip is the mixing leg, which only agrees in direction); recurrence not separated
    (read inert per M24e/M30, not re-proven); one reproduction wrinkle — `trm_flat` etth1-336 drifts
    +0.0086 despite being a 2-D arm (isolated CPU non-determinism, NOT the 3-D-matmul cause); the split is
    dataset-dependent
    (channel count), reported as two regimes not one ratio.
15. **★ On the SYNTHETIC constraint-coupled tasks the mixer's TOKEN-MIXING contribution FLIPS sign vs
    forecasting: here mixing is ~the ENTIRE win (the clean single-flag leg) — #8's dividing line, on the
    mixing axis, is now MEASURED not asserted** (M33 — runs the M31/M32 7-arm decomposition on all five
    mixer-win tasks: Sudoku, `converge`, `hopfield`, `disruption`, `mixed_converge`; 10 configs; no code
    changes, control arms pre-existed). Classification sign (POSITIVE Δ favours the first arm — OPPOSITE of
    forecasting's MSE). **(a) THE FIRM, CLEANLY-ISOLATED RESULT — token-mixing IS the mechanism:**
    Δ(trm_mixer − trm_mixer_nomix) EM **+0.51…+0.99, 8/0 (Sudoku 6/0), p≤.031** in every cell — the only
    single-flag flip (`disable_token_mix`, all else held); removing cross-cell mixing collapses the win.
    Every headline Δ(mixer−ff) reproduces its shipped M23/M24/M24c/M24f/M28a baseline (C3 reseed ⇒ the added
    arms don't perturb mixer/flat/ff; disruption-w32 reproduces only after a gamma-bug fix, see below).
    **(b) WEAKER, BUNDLED read — a channel-independent looped arm underperforms a joint MLP:** Δ(nomix − ff)
    EM negative, 0/8 (−0.02…−0.64), BUT `nomix−ff` bundles CI + loop + tokenization + shared channel MLP
    (the exact confound M31/M32 flagged), so this is a DIRECTION + plausible mechanism (a cell on a
    ring/grid/clique needs its neighbours; CI cuts that path), NOT a clean CI isolation. **(c) readout /
    weight-share axes are UNTESTABLE here (degenerate saturation, NOT a measured null):** Δ(nomix −
    nomix_unsharedro) = Δ(nomix − distinctw) = 0.000 EM only because the three CI arms converge to the
    IDENTICAL per-cell decision (per-seed metrics bit-identical to 15 sig figs; the CI optimum is a trivial
    per-cell lookup with the same argmax under any parameterization) — `sign_test_robust` flags all-ties
    (n_near_tie=8, p=1.0). On Sudoku the accuracy delta is actually nonzero-but-negligible (+0.0004/−0.0003);
    only EM floors to 0. Consistent with the forecasting shared-readout being a horizon-M×H efficiency
    artifact, but M33 does NOT independently measure it. The additive identity (Δ(mixer−ff)=Δ(mixer−nomix)+
    Δ(nomix−ff), residual ≈1e-16, 12 cells) is an arithmetic tautology, a bookkeeping check not evidence.
    **Net: the MIXING operator's contribution is opposite-signed between regimes — forecasting mixing net
    harmful (CI+readout win, #13–#14), synthetic constraint-coupled mixing ~all of it — a strong
    confirmation of #7/#8 on the mixing axis; the CI/readout legs are NOT cleanly measured on the synthetic
    side (bundled / degenerate).** Caveats: `trm_mixer_nomix_distinctw` breaches budget (~0.905–0.936,
    CONSERVATIVE/under) on binary w24/w32 cells (coarse per-cell-weight quantization) but Δ(nomix−distinctw)=0
    there so moot; mixer arms inherit baseline widths (up to 1.041, within ±5% spec); **disruption_w32 had a
    generator bug (gamma=14 copied from w24; the M24f w32 baseline + minimal-PSD margin require gamma=15) —
    fixed and re-run at gamma=15 so it reproduces M24f; the mixing-leg conclusion held at gamma=14 too.**

### 11.3 Open work

- **Only build H/L (M19) after a new Task-C gate passes** (§9.3): a `nested_converge` instance
  where the single-timescale loop plateaus below target **and** ff/untied/untied-mixer do NOT
  share the ceiling at equal compute **and** more data does not close it. The current instance
  fails all three (data is the lever, M18j/M28).
- **Forecasting frontier (M26 scope limits):** the horizon sweep (M30), the shared-readout control
  (M31), and the ingredient decomposition (M32) are **DONE — the mixer win is channel-INDEPENDENCE
  first, the shared readout a modest second, mixing net HARMFUL (§11.2 #13–#14)**. Still open, in
  priority order: (1) **more datasets** (ETTh2/ETTm/electricity/traffic — needs a one-time out-of-band
  fetch + hash + determinism test per §9.4, then replicate the M32 recipe; would test whether the
  CI-leads / shared-readout / CI-beats-CD pattern generalizes). (2) **benchmark-scale models** where
  channel-independence is known to win (M31/M32's `nomix` already points that way at tiny scale; M30's
  CD>CI held only at tiny CPU scale and lost M26's significance — the CD>CI finding may not survive
  scale, and M32 further weakens it: mixing is net harmful). (3) OPTIONAL, to name the single active
  ingredient *within* the channel-independent parameterization even more finely (own-variable input vs
  per-cell tokenization vs recurrence) — M32 already settles the readout/CI/weight-share/mixing split;
  this is only for a recurrence-clean CI-vs-CD flip (a feedforward CI arm), low priority.
- **Low-priority:** a stricter single-pass feedforward-mixer §4a control (the untied mixer
  already shows non-recurrent mixing suffices); a convergent fixed-point task the mixer
  under-fits, to test the DS carry in its motivated regime (none found — the mixer fits them
  all). Neither is needed to interpret current evidence.

### 11.4 Closed levers — do not redo casually

Depth-extrapolation via progressive loss / path-independence (M7/M8); adaptive compute /
ACT as a hard-solving unlock (M8/M23 — ACT works and is adaptive but buys nothing);
contraction / `trm_stable` (M27 — closed, architecture-independent); the M4 sample-wall lift
(M5); the both-axes gate rescue (M6a); decoupled-head mechanism isolation (M10); the model-size
axis for synthetic coherence (M11 — generalizes/strengthens); ECA orbit generality (M11/M12 —
exactly two orbits, no other balanced+deep ECA); leaving the ECA family (M13); the locality
hypothesis (M14); uniform-rule-vs-depth (M15–M15c — leg 2 depth-controlled on rule 13); real
multi-label classification for the loop AND the mixer (M20/M25); the deep-supervision
MECHANISM on the mixer (M29 — M18 holds, do not re-run the m29a/b/c decomposition); and the
forecasting mixing-vs-shared-readout attribution (M31/M32 — channel-independence leads, the shared
readout is a modest second, mixing is net HARMFUL; do not re-run the horizon×dataset decomposition
or the readout/CI/weight-share ingredient split, and do not credit forecasting to the mixing
operator again — §11.2 #13–#14); and the SAME decomposition on the SYNTHETIC mixer-win tasks (M33 —
Sudoku/`converge`/`hopfield`/`disruption`/`mixed_converge`: mixing is ~the ENTIRE win, channel-
independence is HARMFUL, shared readout/weight-share exactly 0.000; do not re-run — the attribution
flips vs forecasting exactly as #8's coupled-vs-CI line predicts — §11.2 #15).

## 12. References

See **`docs/REFERENCES.md`** (TRM / HRM / ARC autopsy / DEQ / Universal Transformers / ACT /
PonderNet / Deep Thinking / Path Independence / Jacobian regularization / the tabular and
extrapolation grounding papers).
