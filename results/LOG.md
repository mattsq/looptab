# Milestone log (chronological)

This file holds the full per-milestone narratives — tables, readings, and caveats — that
used to live in `CLAUDE.md` §11. `CLAUDE.md` §11 now keeps only the terse current state,
the behaviour-changing conclusions, and the next-milestone pointer. Append new milestones
here in order; keep §11 short.

---

## M0 — DONE. Harness + parity (Task A).

Harness landed, real run executed, result recorded below. The end-to-end machinery is in
place and tested (26 tests, ruff check + format clean):
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

---

## M1 — DONE. Task B (iterated CA) + depth-extrapolation harness.

Task B wired, per-cell output head landed, majority baseline integrated, and
depth-extrapolation harness fully implemented and verified (31 tests, all passing):
- Multi-output support in `TRM` (`src/looptab/models/trm.py`) and `FFMatched`
  (`src/looptab/models/controls.py`) via the `out_features` parameter representing CA cell
  width $w$.
- Evaluation metrics (`accuracy` and `exact_match` in `src/looptab/eval/metrics.py`)
  updated to support unroll step override parameter `n_steps` passing to the forward pass.
- Majority baseline metric (`majority_baseline` in `src/looptab/eval/metrics.py`)
  implemented to capture the frequency of the most common class to detect task degeneracy
  early.
- Config-driven depth-extrapolation runner (`src/looptab/run.py` and
  `configs/experiments/m1_iterated_extrapolation.yaml`) executing sweeps over test CA steps
  $T_{test} \in [4, 6, 8, 10]$ and test unrolling steps $R_{test} \in [4, 6, 8, 10, 12]$,
  writing JSON and CSV outputs.

**M1 result (iterated CA rule 30, w=9, distractors=4, n_steps=4, 5 seeds, 100 epochs; ~11.5k
params per arm).** Non-degenerate config (chaotic `rule: 30`, odd width `w: 9`); majority
baseline 0.503 ± 0.004, so the target is balanced. Tracked summary:
`results/m1_iterated_extrapolation_20260620T023255_curve.csv` (per-arm) and
`..._extrapolation.csv` (the depth sweep).

Per-arm accuracy at the training config ($T_{test}=4, R'=4$):

| arm | accuracy | exact-match |
|-----|----------|-------------|
| trm_nods (loop) | 0.972 ± .009 | 0.828 ± .060 |
| ff_matched (control) | 0.971 ± .008 | 0.793 ± .057 |
| trm_ds (loop+DS) | 0.959 ± .007 | 0.760 ± .028 |

**The Δ (per §2 — this is the result, not the per-arm numbers):**

| Δ (paired, 5 seeds) | accuracy | exact-match |
|---|---|---|
| Δ(loop − control) = trm_nods − ff_matched | **+0.001 ± 0.014** | +0.036 |
| Δ(DS − loop) = trm_ds − trm_nods | −0.013 ± 0.010 | −0.068 |
| Δ(loop+DS − control) = trm_ds − ff_matched | −0.012 ± 0.010 | −0.032 |

**Reading (per §8).** On Task B the weight-tied loop gives **no token-accuracy advantage**
over the param-matched control: Δ(loop − control) = +0.001 ± 0.014 — a clean null, the
opposite of M0/parity where the loop's edge was robustness. Deep supervision is mildly
*negative* here (−0.013). The one non-null hint is exact-match: trm_nods leads the control
by ~+0.036 whole-row (the loop may help compose the per-cell outputs), but seed variance
(±0.05–0.06 per arm) swamps it — not a claim, a thing to watch when M2 adds the untied stack.

Extrapolation behaviour:
- **Over-unrolling ($R' > 4$) at $T_{test}=4$:** unrolling the recurrent arms beyond their
  trained depth degrades them back toward baseline (e.g. `trm_nods` → 0.525 at $R'=8$): the
  loop does not settle on a stable step operator / fixed point.
- **OOD depth ($T_{test} > 4$):** every arm — recurrent *and* feedforward — collapses to the
  majority baseline (~0.50) at $T_{test} \in \{6,8,10\}$ for all $R'$.

**Caveat — scope of the negative result.** This says the loop *as trained here* did not learn
a transferable step operator; it does **not** settle the §3 "loops ≈ algorithm steps" thesis.
Two protocol choices stack the deck against extrapolation and are the obvious next levers:
(i) training at a single fixed depth (`T=4`, `R=4`) rather than across a $T$-curriculum, and
(ii) deep supervision pinning *every* loop step to the *final* $T{=}4$ state rather than
supervising step $i$ against the intermediate CA state $s_i$. `trm_nods` (final-step loss only)
also fails, so the null is not purely a DS artifact — but a step-aligned curriculum is the
cleaner test and remains unrun. (This lever is taken up in M3b.)

---

## M2 — DONE. Untied-stack control (§4b), two forms, Task A + Task B.

The untied-stack control (§4b) landed in *two* forms and was run on Task A and Task B. This
is the control M0/M1 flagged as *the* missing piece before crediting anything to "tied
recurrence." **Both rounds of the result are recorded below because the first round was
confounded** — a worked example of the §8 trap (a clean Δ on a dirty axis), caught in review.
- `UntiedStack` (`src/looptab/models/controls.py`): the TRM block stacked `n_steps`× with a
  **separate** `update_net` + `readout` per step — identical per-step compute/depth to TRM,
  the only code difference is `ModuleList` vs a shared module. Supports deep supervision and
  multi-output. It is **not** param-matched: untying a tied loop necessarily multiplies block
  params by ~`n_steps` (measured **3.98×**), so Δ(loop − untied_stack) co-varies tying *with*
  capacity. Kept for completeness but **it cannot isolate tying.**
- `UntiedStackMatched`: the same untied stack **width-shrunk** (`hidden = latent = w`, via the
  same nearest-match search `FFMatched` uses) so total params ≈ the loop's. This holds capacity
  *and* depth fixed and varies **only** weight tying — it is the clean control. Param ratios to
  the loop: 0.99× (parity 9781 vs 9922; CA 11439 vs 11538).
- Registered `untied_stack` + `untied_matched`; `_build_model` passes `deep_supervision` to
  both; the extrapolation harness routes both as **fixed-depth** arms (like `ff_matched`:
  evaluated once, flat across `R'`) since an untied stack cannot unroll past `n_steps`. Configs
  `m2_parity_sweep.yaml` / `m2_iterated_extrapolation.yaml` run the 5-arm factorial. 44 tests
  (shapes, untied-ness, over-unroll clamp, param ratios, determinism, routing), ruff clean.

**The clean M2 Δ is Δ(loop − untied_matched)** — weight tying at a *fixed parameter budget and
fixed depth*. `Δ(untied_matched − ff_matched)` isolates **depth at fixed capacity** (deep untied
vs shallow MLP, same budget). `Δ(loop − untied_stack)` is the confounded version (tying + ~4×
capacity), retained only to expose the confound.

**M2 result — Task A (parity, d=20, n_steps=4, 5 seeds, 100 epochs).** Tracked summary:
`results/m2_parity_sweep_20260620T035036_curve.{csv,png}`.

| k | trm_nods (loop, 9.9k) | untied_matched (deep, 9.8k) | ff_matched (shallow, 9.9k) | untied_stack (deep, 39.5k) |
|---|------|------|------|------|
| 2 | 1.000 ± .000 | 1.000 ± .000 | 1.000 ± .000 | 1.000 ± .000 |
| 3 | 1.000 ± .000 | 1.000 ± .000 | 1.000 ± .000 | 1.000 ± .000 |
| 4 | 1.000 ± .000 | 1.000 ± .000 | **0.763 ± .246** | 1.000 ± .000 |

At k=4: Δ(loop − untied_matched) = **+0.000**, Δ(untied_matched − ff_matched) = **+0.237 ± .246**,
Δ(loop − ff_matched) = **+0.237 ± .246**.

**M2 result — Task B (iterated CA rule 30, w=9, distractors=4, n_steps=4, 5 seeds, 100 epochs).**
Majority baseline 0.503 ± .004. Tracked summary:
`results/m2_iterated_extrapolation_20260620T035435_curve.csv` (+ `..._extrapolation.csv`).

| arm (params) | accuracy | exact-match |
|-----|----------|-------------|
| trm_nods (loop, 11.5k) | 0.972 ± .009 | 0.828 ± .060 |
| untied_matched (deep, 11.4k) | **0.821 ± .021** | 0.197 ± .039 |
| ff_matched (shallow, 11.5k) | 0.971 ± .008 | 0.793 ± .057 |
| untied_stack (deep, 46k) | 0.999 ± .001 | 0.994 ± .008 |

Paired Δ on accuracy (5 seeds; exact-match deltas are point estimates, no variance computed):
Δ(loop − untied_matched) = **+0.151 ± 0.027** (EM +0.631); Δ(untied_matched − ff_matched) =
**−0.149 ± 0.015** (EM −0.596); Δ(loop − ff_matched) = +0.001 ± 0.014 (EM +0.036).

**Reading (per §2/§8 — this is the result, and it CORRECTS the first round).** Once capacity is
held fixed by `untied_matched`, two clean facts emerge, one per task:
- **Task A — the active ingredient is depth, and tying is neutral.** At a fixed budget, the
  *deep* arms (loop and `untied_matched`) both solve k=4 perfectly while the *shallow* same-budget
  MLP collapses on 2/5 seeds (0.763 ± .246). Loop = untied_matched exactly (Δ = 0), so weight
  tying buys nothing on parity; depth does (Δ(deep − shallow) = +0.237). This now *licenses* the
  "M0 edge = depth" claim that the confounded round could not.
- **Task B — weight tying HELPS at a fixed budget; the first round's conclusion was a capacity
  artifact.** The fat `untied_stack` still scores 0.999, but it has **4× the params**; the
  *param-matched* untied stack scores only **0.821**, below even the shallow MLP. So the loop
  *beats* the fair untied control by **+0.151 ± 0.027** (EM 0.828 vs 0.197). Round 1 reported
  "tying costs accuracy on B" — that was the §8 trap: the apparent untied win was bought with 4×
  capacity, not earned by untying. Removed, the sign flips.

**Synthesis (the real M2 finding).** Among the three *param-matched* architectures, the
weight-tied loop is the **only one robust on both tasks**: `ff_matched` (shallow) solves CA but
fails parity-k4; `untied_matched` (deep, narrow blocks) solves parity but fails CA; the loop
solves both. Mechanistically, tied recurrence is the parameter-efficient way to get **both depth
and width** from one budget — the untied stack must split the budget into narrow blocks (loses
the width CA needs), the shallow MLP has no depth (loses what parity needs), the loop reuses one
*wide* block at depth and gets both. Extrapolation is unchanged from M1 (over-unrolling the loop
past `R=4` decays toward baseline; all arms collapse at OOD depth `T>4`).

**Consequence for M3 (§9).** Less negative than the confounded round implied, but not a clean
pass either. Against each *fair* (param-matched) control the loop wins on one task and ties on
the other — it is never beaten by a capacity-matched control, and is uniquely robust across both
— but it does not strictly dominate any single control on *both* tasks, and each task rests on
one config (5 seeds). So the §9 gate is **not yet cleanly cleared**: confirm the cross-task
robustness on more Task B rungs / rules (and the M1 curriculum levers) before building the H/L
hierarchy. The signal now points toward the loop having genuine value, which it did not after
round 1.

---

## M2-confirm — DONE. Replicate the Task B tying result across a rule × width grid.

M2 named one blocker before the §9 hierarchy: the Task B finding rested on a *single* config
(rule 30, w=9). This milestone re-ran the **same 5-arm factorial** across a **grid of CA rule
{30, 90, 110} × width {9, 13}** (6 cells × 5 seeds, 100 epochs) to check the cross-task
robustness isn't a one-config fluke. A multi-param `grid` axis was added to the substrate
(`GridConfig` + `ExperimentConfig.axis_points`, `configs/experiments/m2_confirm_iterated_grid.yaml`),
generalising the 1-D `sweep`; the runner now also emits a **per-config Δ table CSV** with paired
exact-match Δs (variance), and reports EM deltas alongside accuracy. `grid` and `extrapolation`
are mutually exclusive (the harness keeps one (T,R) result set) — depth-extrapolation is M1's
separate question, left untouched. 50 tests, ruff clean. Tracked summary:
`results/m2_confirm_iterated_grid_20260620T070204_{curve,deltas}.csv`. The `rule=30, w=9` cell
reproduces the M2 numbers bit-consistently (loop 0.972, untied_matched 0.821, ff 0.971).

**The clean tying Δ(loop − untied_matched), all 6 cells (accuracy; EM in brackets):**

| rule | w | trm_nods | untied_matched | ff_matched | Δ(loop − untied_matched) | Δ(loop − ff_matched) |
|------|---|----------|----------------|------------|--------------------------|----------------------|
| 30  | 9  | 0.972 | 0.821 | 0.971 | **+0.150 ± .027** [EM +0.63] | +0.001 ± .014 |
| 30  | 13 | 0.752 | 0.689 | 0.813 | **+0.062 ± .010** [EM +0.02] | −0.062 ± .013 |
| 90  | 9  | 0.997 | 0.886 | 1.000 | **+0.111 ± .075** [EM +0.62] | −0.003 ± .007 |
| 90  | 13 | 0.973 | 0.830 | 1.000 | **+0.143 ± .064** [EM +0.64] | −0.027 ± .031 |
| 110 | 9  | 0.979 | 0.865 | 0.986 | **+0.114 ± .028** [EM +0.55] | −0.007 ± .017 |
| 110 | 13 | 0.800 | 0.723 | 0.831 | **+0.077 ± .007** [EM +0.06] | −0.031 ± .008 |

**Reading (per §2/§8).** The central M2 fact **replicates cleanly and consistently:**
- **Weight tying helps at a fixed budget on CA in *every* cell.** Δ(loop − untied_matched) is
  **positive in all 6 cells** (+0.062 → +0.150 token-acc), variance bands never crossing zero;
  `untied_matched` (deep, narrow blocks) is the **weakest param-matched arm in all 6 cells**.
  Δ(untied_matched − ff_matched) is negative everywhere (−0.11 → −0.17): splitting one budget
  into narrow untied blocks consistently loses the width CA needs. (EM deltas are large at w=9,
  ~+0.6, and small at w=13 where every arm's whole-row score is low — but token-acc tying Δ stays
  clearly positive.) **This is the requested confirmation: the loop's CA advantage over the fair
  untied control is not a one-config fluke.**
- **Refinement the grid surfaces (reported plainly — this is the inconvenient half):** the loop
  does **not** beat the *shallow* param-matched MLP (`ff_matched`, the §4a control) on CA.
  Δ(loop − ff_matched) is positive in only **1/6** cells (rule30/w9, +0.001 ± .014 — noise) and
  ≤ 0 in the other five; it is **clearly negative at w=13** for rule 30 (−0.062 ± .013) and rule
  110 (−0.031 ± .008) (the rule90/w13 −0.027 ± .031 band still crosses zero). The wide shallow MLP
  is the strongest param-matched arm on wide CA. Starkest case — **rule90, w13: `ff_matched`
  reaches EM 1.000 / acc 1.000 (perfectly solves it) while the loop gets EM 0.71 / acc 0.97.** So
  the loop's CA value is specifically *"tying beats a fair *untied* stack,"* **not** *"the loop
  beats its §4a control."* (EM tying-Δs are large at w=9 ~+0.6 but shrink to ~+0.02–0.06 at w=13.)
- **Deep supervision stays neutral:** Δ(trm_ds − trm_nods) ∈ [−0.013, +0.010] across all cells —
  consistent with M0/M1/M2, the loop's effect is not silently DS.

**Cross-task synthesis.** The precise, defensible robustness claim: among the four *param-matched*
arms, the loop (`trm_nods`) is **never the worst on either task** — `ff_matched` is worst on Task A
(parity-k4 collapse to 0.763, M2), `untied_matched` is worst on Task B (every one of the 6 cells).
That "never-the-worst" property is unique to the loop and now holds across 3 rules × 2 widths on
Task B. **But this is robustness-as-not-failing, NOT dominance:** the loop is top-2 *among all five
arms* in only 1/6 CA cells (the fat `untied_stack` and `ff_matched` usually beat it), and top-2
*among the four param-matched arms* in 2/6. The earlier "top-2 on CA in all 6 cells" was wrong;
corrected here.

**§9 gate — still NOT cleared; M3 stays gated.** §9's bar is literal: *no hierarchy until the loop
"beats its control on Task A and Task B."* The loop beats its §4b control (`untied_matched`) on B
robustly (6/6) — but it does **not** beat its mandatory §4a control (`ff_matched`) on B (wins 1/6,
by noise; loses on wide CA). On Task A it's the mirror image: it beats `ff_matched` (+0.237, M2)
but only ties `untied_matched`. So on **neither** task does the loop beat *both* its controls, and
on Task B it beats only the untied one. What M2-confirm *did* establish — and it's a real result —
is that the **tying-at-fixed-budget advantage over the untied stack replicates cleanly across
rules/widths** (Δ(loop − untied_matched) > 0, lower band > 0, in all 6 cells). What it did **not**
establish is the §9 condition. Two further gaps remain: the **Task A leg is still single-config**
(one `d`), and the **M1 step-aligned curriculum** lever is unrun.

---

## M3a — DONE. Depth-at-fixed-budget sweep (Task B). Prediction FALSIFIED.

Tested the M2-synthesis prediction: *if tied recurrence buys depth AND width from one
budget, the loop's advantage should GROW with required depth* (untying only becomes
unaffordable when many steps are needed). Swept depth **T ∈ {4, 8, 12, 16}** at a single
fixed parameter budget, **rules {30, 110}** (rule 90 skipped — linear over GF(2), not a
fair depth stress), **w ∈ {16, 20}**, with the loop's `n_steps` coupled to T. Four arms:
`trm_nods` (loop, final-step loss), `ff_matched` (§4a shallow), `untied_matched` (§4b
deep-narrow), `untied_stack` (~T× params — **non-param-matched ceiling reference only**).
**10 seeds**, 100 epochs. New substrate: `couple_n_steps_to_param`, `budget_audit`,
per-arm `train_accuracy` diagnostic, paired sign test on every Δ, depth-budget plots.
56 tests, ruff clean. Tracked summary:
`results/m3a_depth_budget_sweep_20260620T110725_{curve,deltas,params}.csv` +
`..._depth_{curve,deltas}.png`.

**Budget parity (the confound guard) — HELD for the depth-attribution arms.** The loop
(reference) and `ff_matched` are weight-tied / T-independent, so their param counts are
**exactly constant across all T** (loop 13792 @w16, 15080 @w20; ff within 0.7%). The audit
flagged **8 cells** where `untied_matched` drifts past ±2% (ratios 0.93–1.04) — this is the
**expected high-T width-quantization finding**, not a blocker: with T blocks sharing one
width, integer-width steps get coarse and the matched stack is forced into narrow/degenerate
blocks (w′→7–8 at T=16). It is surfaced, not hidden; the headline Δ(loop − ff_matched) is on
two *exactly* budget-matched arms, so depth attribution there is clean.

**Per-arm test accuracy collapses to baseline at T ≥ 8 for EVERY arm** (baseline ≈ 0.50–0.55):

| rule | w | T=4 (loop / ff / um / us) | T=8 | T=12 | T=16 |
|---|---|---|---|---|---|
| 30  | 16 | 0.707 / 0.753 / 0.658 / 0.775 | 0.523 / 0.525 / 0.518 / 0.529 | ≈0.504 all | ≈0.505 all |
| 30  | 20 | 0.659 / 0.695 / 0.620 / 0.719 | ≈0.514 all | ≈0.501 all | ≈0.501 all |
| 110 | 16 | 0.748 / 0.751 / 0.683 / 0.803 | 0.557 / 0.567 / 0.544 / 0.556 | ≈0.538 all | ≈0.535 all |
| 110 | 20 | 0.687 / 0.696 / 0.640 / 0.748 | 0.540 / 0.553 / 0.535 / 0.532 | ≈0.532 all | ≈0.531 all |

**Headline Δ(loop − control) vs T (paired, 10 seeds; sign-test p):**

| rule, w | Δ(loop − ff_matched) | Δ(loop − untied_matched) |
|---|---|---|
| 30,16 | T4 **−0.046** (p=.002) → T8 −0.002 → T16 −0.002 (ns) | T4 **+0.049** (p=.002) → T8 +0.005 → T16 +0.002 (ns) |
| 30,20 | T4 **−0.036** (p=.002) → T≥8 ≈0 (ns) | T4 **+0.040** (p=.002) → T≥8 ≈0 (ns) |
| 110,16 | T4 −0.003 (ns) → T8 **−0.011** (p=.002) → T16 −0.003 | T4 **+0.065** (p=.002) → T8 **+0.013** (p=.002) → T16 −0.003 (ns) |
| 110,20 | T4 −0.009 → T8 **−0.013** (p=.002) → T12 −0.009 → T16 −0.005 | T4 **+0.047** (p=.002) → T8 +0.005 → T16 **−0.013** (p=.002, loop WORSE) |

**Reading (per §8 — the honesty clause fires; the prediction is FALSIFIED).**
- **The loop's advantage does NOT grow with depth — it vanishes.** The one effect that
  replicates from M2-confirm, Δ(loop − untied_matched) > 0 (tying beats the fair untied stack
  at fixed budget), is **largest at the *shallowest* depth (T=4: +0.04 to +0.065, 10/0 seeds,
  p=.002) and shrinks to ≈0 — or flips negative — by T≥12**. That is the **opposite** of the
  prediction. The loop **never** beats the §4a shallow `ff_matched` at any T (≤0 everywhere;
  significantly negative at T=4 for rule 30 and at T=8 for rule 110).
- **Root cause = an optimization / learnability wall shared by ALL arms, not a depth-capacity
  story.** At T ≥ 8 the s₀→s_T target collapses to baseline on **test AND train** for every
  arm. Even the **fat `untied_stack` ceiling (up to 16× params)** only reaches ~0.75–0.79
  *train* accuracy and ~0.50–0.53 *test* at T ≥ 8 — i.e. nobody, at any capacity, learns the
  deep CA map one-shot at this scale/epoch budget. Per the prompt's explicit diagnostic
  (`train acc also low ⇒ optimization failure, not a capacity verdict`), the high-T regime
  **cannot test the claim**: the depths where tying *should* pay off are exactly where the
  target is unlearnable for everyone, so there is no signal to separate the arms.
- **Consequence — the M2 synthesis must be SOFTENED.** "Tied recurrence is the
  parameter-efficient way to buy depth *and* width from one budget" is **not demonstrated** by
  this sweep. The *width* half stands (M2/M2-confirm: tying beats the fair untied stack at
  shallow T). The *depth* half is **unsupported**: at a fixed tiny budget the loop cannot
  actually convert extra unrolled steps into solving deeper computations any better than the
  shallow/untied controls — and the tying edge it does have is a *shallow-depth* phenomenon
  that decays with T. So the loop's established value is "robust width-at-budget vs a fair
  untied stack," **not** "buys usable depth."
- **DS untested here (still Phase 2):** M3a runs final-state loss only. The optimization wall
  is the precise failure step-aligned DS + a T-curriculum is meant to break (supervise the
  intermediate states so the loop learns the one-step operator instead of the impossible
  T-step map). M3a *motivates* M3b rather than refuting it; whether the loop can learn a
  transferable step operator at all is M3b's question.

**Net:** a clean negative on "depth makes the loop win." The loop is not beaten by a
capacity-matched control at the only learnable depth (T=4: ties/loses to ff_matched, beats
untied_matched), but its edge does not scale with depth, and at this budget depth itself is
unlearnable for all arms past T=4. The §9 gate is no closer; the M2 "both axes" framing is
now explicitly hedged to "width-at-budget only."
