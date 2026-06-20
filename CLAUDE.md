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

**M0 — DONE.** Harness landed, real run executed, result recorded below. The
end-to-end machinery is in place and tested (26 tests, ruff check + format clean):
- Generators for Task 0 + Task A — spec-faithful and determinism-tested
  (`src/looptab/data/generators.py`, `tests/test_generators.py`).
- TRM-style recurrent refinement model with optional per-step readouts
  (`src/looptab/models/trm.py`).
- Param-matched feedforward control (§4a), param count matched analytically to ~0.6%
  (`src/looptab/models/controls.py`).
- Training loop with deep supervision as a **per-arm** weight, not a global flag
  (`src/looptab/train/loop.py`).
- Config-driven runner with **named arms** + a **single-config sweep** over a task
  parameter, emitting `Δ` between any pair of arms with variance bands, plus a curve
  CSV (and a PNG if matplotlib is installed) (`src/looptab/run.py`,
  `configs/experiments/m0_parity_sweep.yaml`).

**Key design choice (avoids the §4/§8 confound):** deep supervision is its own arm.
The canonical M0 experiment runs three arms — `trm_ds` (loop + DS), `trm_nods`
(loop, no DS), `ff_matched` (control) — so we report `Δ(trm_nods − ff_matched)`
(the loop alone) and `Δ(trm_ds − trm_nods)` (deep supervision alone) separately.
Each outer seed also draws a **new `task_seed`** (train/test still share it within a
seed, per §3) so the variance band reflects function-level variation, not just init+rows.

**Definition of done for M0:** produce the `k`-vs-accuracy curve for both models,
with variance bands, from a single config — done via `m0_parity_sweep.yaml`. Tracked
summary: `results/m0_parity_sweep_20260620T012344_curve.{csv,png}`.

**M0 result (parity, d=20, n_steps=4, 5 seeds, 100 epochs; ~9.9k params per arm).**

| k | trm_ds (loop+DS) | trm_nods (loop) | ff_matched (control) | Δ(loop − control) | Δ(DS − loop) |
|---|------------------|-----------------|----------------------|-------------------|--------------|
| 2 | 1.000 ± .000 | 1.000 ± .000 | 1.000 ± .000 | +0.000 | +0.000 |
| 3 | 0.978 ± .050 | 1.000 ± .000 | 1.000 ± .000 | +0.000 | −0.022 |
| 4 | 1.000 ± .000 | 1.000 ± .000 | 0.763 ± .246 | **+0.237 ± .246** | −0.000 |

**Reading (reported plainly per §8).** The weight-tied recurrent loop is the active
ingredient: at the hardest rung (k=4) `trm_nods` solves parity on every seed while the
param-matched feedforward control collapses to ~chance on 2 of 5 seeds (high-variance
failure — the seed-sensitivity §5.2 warns about; the loop's edge is *robustness*, not a
new capacity ceiling). **Deep supervision is NOT the active ingredient here:** Δ(DS − loop)
is ≈0 at k=2/4 and slightly *negative* at k=3 — so the loop's win is not silently credited
to DS. k≤3 is too easy to separate the arms (all ≈1.0). Caveat: a single run on one task;
the untied-stack control (§4b, M2) is needed before concluding "tied recurrence" beats
"mere depth."

**M1 — DONE.** Task B wired, per-cell output head landed, and depth-extrapolation harness fully implemented (30 tests, all passing):
- Multi-output support in `TRM` (`src/looptab/models/trm.py`) and `FFMatched` (`src/looptab/models/controls.py`) via the `out_features` parameter representing CA cell width $w$.
- Evaluation metrics (`accuracy` and `exact_match` in `src/looptab/eval/metrics.py`) updated to support unroll step override parameter `n_steps` passing to the forward pass.
- Config-driven depth-extrapolation runner (`src/looptab/run.py` and `configs/experiments/m1_iterated_extrapolation.yaml`) executing sweeps over test CA steps $T_{test} \in [4, 6, 8, 10]$ and test unrolling steps $R_{test} \in [4, 6, 8, 10, 12]$, writing JSON and CSV outputs.

**M1 result (iterated CA rule 90, w=8, distractors=4, n_steps=4, 5 seeds, 100 epochs).**
Because width 8 rule-90 (XOR neighbor updates) is a linear cellular automaton, all models (including the param-matched MLP control) converged to 100% accuracy and generalised perfectly up to $T_{test}=10$ steps. No divergence in accuracy was observed between the recurrent arms and control.

_Then:_
- **M2** — add the depth/compute-matched untied control (§4b).
- **M3** — revisit the hierarchy (Task C) *iff* M0–M2 justify it.

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
