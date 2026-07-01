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
flagged **8 cells** where `untied_matched` drifts past ±2% (ratios 0.93–1.04; **worst −6.6%**
at T=16/w=20) — this is the **expected high-T width-quantization finding**, not a blocker: with
T blocks sharing one width, integer-width steps get coarse and the matched stack is forced into
narrow/degenerate blocks (w′→7–8 at T=16). The breach is **one-directional** — at high T the
stack lands *under* budget (fewer params), so if anything it handicaps `untied_matched`, which
only *strengthens* the "loop edge vanishes by T≥8" reading (the loop fails to beat even an
under-budget untied stack). It is surfaced, not hidden; and the headline Δ(loop − ff_matched) is
on two *exactly* budget-matched arms (≤0.7% apart, T-independent), so depth attribution is clean
on that pair regardless.

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

---

## M3b — DONE. Step-aligned DS + T-curriculum (Task B). Layered result: DS is mis-specified, not inert; but no transferable operator.

Stacked the two levers M1/M3a named: a depth **curriculum** (train T ~ Uniform{1..8}) and
**step-aligned DS** (loop step i supervised against the intermediate CA state sᵢ, not the
final one). Task config = M1's exactly (rule 30, w=9, distractors=4) so the extrapolation
curves are directly comparable to M1's collapse. Five arms, 10 seeds, 100 epochs:
`trm_stepDS` (step-aligned DS), `trm_finalDS` (the old final-state DS), `trm_nods` (loop,
final-step loss) — all curriculum-trained — plus `ff_matched` / `untied_matched` grounding.
New substrate: `make_iterated(return_trajectory=True)`, `TrajectoryDataset`,
`CurriculumConfig`, `ModelConfig.ds_mode`, `train_curriculum`. 66 tests, ruff clean. Tracked:
`results/m3b_stepDS_curriculum_20260620T130400_{curve,deltas,extrapolation,extrapolation_deltas}.csv`
+ `..._extrapolation.png`. (Re-run after an adversarial review added per-cell paired sign tests
on the extrapolation diagonal — `_extrapolation_deltas.csv`; the training is deterministic, so
every accuracy reproduces the first run bit-for-bit, now with significance attached.)

**Reference (T=8) per-arm acc, 10 seeds:** trm_nods **0.646** > trm_finalDS 0.629 >
untied_matched 0.592 > trm_stepDS 0.582 > ff_matched 0.543 (baseline 0.507). Paired Δ
(sign-test p): Δ(stepDS − nods) = **−0.064 ± 0.013** (0/10, p=.002); Δ(stepDS − finalDS) =
**−0.047 ± 0.018** (0/10, p=.002); Δ(finalDS − nods) = **−0.017 ± 0.008** (0/10, p=.002).
*At the deep reference depth, step-aligned DS significantly HURTS, and the old final-state DS
stays mildly negative (consistent with M0–M3a).* Note the curriculum-trained plain loop
(`trm_nods`) is the best arm here, beating both param-matched controls — a small loop positive.

**But the extrapolation diagonal (R′ = T, "correct unroll") tells the real, opposite story at
short horizon:**

| T = R′ | baseline | trm_stepDS | trm_finalDS | trm_nods | ff_matched | untied_matched |
|---|---|---|---|---|---|---|
| 4  | 0.505 | **0.838 ± .024** (EM .285) | 0.628 (EM .024) | 0.676 (EM .046) | 0.517 | 0.503 |
| 8  | 0.507 | 0.582 (EM .017) | 0.629 (EM .034) | 0.646 (EM .037) | 0.543 | 0.592 |
| 12 | 0.508 | 0.504 | 0.524 | 0.524 | 0.520 | 0.524 |
| 16 | 0.504 | 0.498 | 0.507 | 0.514 | 0.530 | 0.514 |

**Reading (per §8 — the honesty clause cuts both ways).**
- **Step-aligned DS is NOT inert — it is the first clear DS WIN in the whole project, but only
  at SHORT rollout.** At T=4 / R′=4, `trm_stepDS` hits **0.838 acc (EM 0.285)** vs `trm_nods`
  0.676 (EM 0.046) and `trm_finalDS` 0.628. The paired diagonal Δ is properly sign-tested (not
  just eyeballed bands): **Δ(stepDS − nods) = +0.162 ± 0.023, 10/0 seeds, p=.002**; Δ(stepDS −
  finalDS) = +0.210 ± 0.026, 10/0, p=.002 (per-cell tests in `..._extrapolation_deltas.csv`).
  The mechanism fires exactly as designed: trained to emit sᵢ at step i, the loop nails the
  state after a few steps. **This overturns the M0–M3a "DS is neutral-to-negative" conclusion
  at short horizons** — that null was partly an artifact of *mis-specified* (final-state) DS,
  the M3b hypothesis. So DS's effect is real and large, not inert.
- **The sign of the step-aligned DS effect FLIPS with horizon.** Δ(stepDS − nods) is **+0.162
  at T=4** (10/0, p=.002) but **−0.064 at T=8** (0/10, p=.002) — two distinct, oppositely-signed
  paired tests (and stepDS sits at baseline by T≥12). Step-aligned supervision
  trades *deep-final* accuracy for *shallow-rollout* fidelity: pinning each step to sᵢ helps
  short rollouts but, at the deep T=8 readout, the arm that optimizes *only* the final state
  (`nods`) wins. There is no single "DS helps/hurts" verdict — it is horizon-dependent.
- **No transferable step operator; M1's extrapolation null STANDS.** On the diagonal the
  operator degrades with depth (T=4 0.84 → T=8 0.58) and **fully collapses to baseline at OOD
  T=12, 16 for EVERY arm**, exactly as M1. Within the curriculum there is *weak* compositional
  structure (for a T=8 task, R′=8 (0.58) beats R′=4 (0.48) — more loops help recover the deeper
  state), but it tops out well below the short-horizon quality and vanishes entirely past the
  trained horizon T_max=8. Over-unrolling a fixed-T task past its true depth still decays to
  baseline (the loop does not settle a stable fixed point). So curriculum + step-alignment did
  **not** buy depth transfer — the §3 "loops ≈ algorithm steps, extrapolate by unrolling more"
  thesis remains **unsupported** in this setting; this is a stronger, cleaner null than M1's
  (now with the two obvious levers applied and still failing past the training horizon).

**M3b answers its three questions:** (1) *Can the loop learn the operator?* — yes at short
horizon (T=4 strong, mechanism-consistent), degrading with depth, failing OOD. (2) *Is DS
inert or mis-specified?* — **mis-specified**: step-aligned DS is a large short-horizon win
(first clear DS positive), while final-state DS stays inert-to-negative. (3) *Does it
extrapolate?* — **no**: OOD depth collapses to baseline for all arms; M1 reproduced.

**Net for the loop thesis (§9 stays gated).** Two genuine findings: a *positive* (step-aligned
DS materially improves short-horizon operator fidelity — DS was mis-specified, not inert) and a
*clean null* (no transferable depth; the loop does not learn an operator that composes beyond
its trained horizon). Neither moves the literal §9 gate (still no task where the loop beats
*both* controls), but together they sharpen the picture: the loop's value is local/robustness,
not algorithmic depth-extrapolation. Caveats: one rule (30) / one width (9); a longer or
annealed curriculum, or a fixed-point/halting objective, are untried levers for the
extrapolation null.

---

## M4 — DONE. Replicate & stress-test the Task A parity leg across d × k. Original result REPLICATES; "depth helps, tying neutral" confirmed; no loop-beats-both cell.

The single biggest evidential gap M2-confirm/M3 named: the whole Task A story (M0/M2) rested on
**one** setting, d=20, with the separation only at k=4. M4 re-ran the full parity arm factorial
over a compact 2-D grid **d ∈ {20, 40, 80} × k ∈ {3, 4, 5}** (9 cells × **10 seeds**, 100 epochs)
to decide whether that finding is robust or a single-config artifact. No new generator, no new
substrate — reuses the existing `grid` axis and `budget_audit` (`configs/experiments/m4_parity_grid.yaml`).
Hyperparameters inherited **verbatim** from `m2_parity_sweep.yaml` (hidden=latent=64, n_steps=4,
lr 1e-3, wd 1e-4, batch 256, n_train 4000, n_test 1000); no per-cell tuning (§8). 67 tests, ruff
clean. Tracked summary: `results/m4_parity_grid_20260621T000340_{curve,deltas,params}.csv`. The
d=20,k=4 cell reproduces the M2 separation (loop/um perfect; ff collapses on a minority of seeds).

**Arms & budget (the confound guard).** Four required arms — `trm_nods` (loop, the budget
reference), `trm_ds` (loop+final-state DS), `ff_matched` (§4a shallow), `untied_matched` (§4b clean
tying control) — plus `untied_stack` as a **labelled non-param-matched ceiling** (~4× params, never
the clean control). Per-cell param ratios to the loop: `ff_matched` 0.994–1.007, `untied_matched`
0.986–**1.023**. The audit flagged **3 cells** (all at d=40) where `untied_matched` drifts to
**+2.3%** (ratio 1.023) — the expected integer-width-quantization finding, surfaced not hidden.
All three breach cells (d=40, k∈{3,4,5}) sit *inside* the unlearnability wall below — k=3 is solved
by every arm, k=4/k=5 are at test-chance for every arm — so **no significant Δ rides on them** and
the breach changes no verdict. (Note the drift is *over* budget, which for the "tying neutral"
reading cuts toward the loop, not against it — an over-budget um that still only ties the loop is, if
anything, evidence the loop is not *better* than a strictly-matched um; we do not lean on this, since
the cells are at chance anyway.) The headline `Δ(loop − ff_matched)` is on two arms matched to ≤0.7%
in every cell, so depth attribution is clean regardless.

**Per-arm test accuracy (mean ± std, 10 seeds; us = untied_stack ceiling):**

| d | k | baseline | trm_nods (loop) | trm_ds | ff_matched | untied_matched | us (ceiling) |
|---|---|---|---|---|---|---|---|
| 20 | 3 | .512 | 1.000 ± .000 | 0.988 ± .035 | 1.000 ± .000 | 1.000 ± .000 | 1.000 |
| 20 | 4 | .513 | 1.000 ± .000 | 1.000 ± .000 | **0.772 ± .240** | 1.000 ± .000 | 1.000 |
| 20 | 5 | .519 | 1.000 ± .000 | 1.000 ± .000 | **0.503 ± .015** | 0.901 ± .210 | 1.000 |
| 40 | 3 | .512 | 1.000 ± .001 | 1.000 ± .001 | 0.999 ± .001 | 1.000 ± .000 | 1.000 |
| 40 | 4 | .513 | 0.508 ± .012 | 0.515 ± .026 | 0.504 ± .015 | 0.572 ± .161 | 0.580 |
| 40 | 5 | .511 | 0.492 ± .014 | 0.496 ± .014 | 0.495 ± .015 | 0.526 ± .090 | 0.494 |
| 80 | 3 | .515 | 0.698 ± .206 | 0.672 ± .206 | 0.533 ± .038 | 0.670 ± .230 | 0.842 |
| 80 | 4 | .514 | 0.498 ± .022 | 0.501 ± .023 | 0.504 ± .019 | 0.503 ± .017 | 0.499 |
| 80 | 5 | .512 | 0.500 ± .016 | 0.503 ± .015 | 0.509 ± .020 | 0.506 ± .026 | 0.505 |

*(For single-output parity, exact-match ≡ accuracy, so it is not reported separately — §3.)*

**Paired deltas (accuracy, 10 seeds; sign-test p where a call is meaningful).** Two-sided exact
binomial at 10 seeds: 10/0 → p=.002, 9/1 → p=.021, 8/2 → p=.109; ties (identical accuracy, common
when arms saturate at 1.000) reduce the effective n, so e.g. 6/0 with 4 ties → p=.031.

| d | k | Δ(loop − ff) | Δ(loop − um) | Δ(um − ff) | Δ(ds − nods) |
|---|---|---|---|---|---|
| 20 | 3 | +0.000 (tie) | +0.000 (tie) | +0.000 (tie) | −0.012 (ns) |
| 20 | 4 | **+0.228** (6/0, p=.031) | +0.000 (tie) | **+0.228** (6/0, p=.031) | −0.000 (ns) |
| 20 | 5 | **+0.497** (10/0, p=.002) | +0.099 (2/0, p=.5) | **+0.398** (9/1, p=.021) | +0.000 (tie) |
| 40 | 3 | +0.000 (ns) | −0.000 (ns) | +0.001 (ns) | −0.000 (ns) |
| 40 | 4 | +0.004 (ns) | −0.064 (3/7, p=.34) | +0.068 (6/3, p=.51) | +0.008 (ns) |
| 40 | 5 | −0.003 (ns) | −0.034 (2/8, p=.11) | +0.032 (5/5, p=1) | +0.004 (ns) |
| 80 | 3 | +0.165 (6/4, p=.75) | +0.028 (6/4, p=.75) | +0.137 (6/4, p=.75) | −0.026 (ns) |
| 80 | 4 | −0.006 (ns) | −0.005 (ns) | −0.001 (ns) | +0.003 (ns) |
| 80 | 5 | −0.010 (2/8, p=.11) | −0.007 (3/7, p=.34) | −0.003 (ns) | +0.003 (ns) |

**Reading (per §2/§8 — answering M4's five questions).**

1. **Does the loop still beat `ff_matched`? YES at d=20, and it STRENGTHENS with k.** The M2
   d=20,k=4 separation reproduces (Δ(loop − ff) = **+0.228**, 6/0, p=.031) and *intensifies* at
   k=5, where `ff_matched` sits at **pure chance (0.503)** while the loop is perfect on all 10
   seeds (Δ = **+0.497**, 10/0, p=.002). So the M0/M2 headline "the loop beats its §4a shallow
   control on parity" is **not a single-config artifact** — it holds across the k-ladder at d=20.
   (Mechanistically the Δ "grows" because the *control's* floor drops as k rises — ff_matched slides
   1.000→0.772→0.503 while the deep arms stay pinned at 1.000; the loop is not doing progressively
   *more*, the shallow MLP is failing progressively *harder*. The separation is real either way.)
2. **Does the loop ever beat `untied_matched`? NO — Task A is still "depth helps, tying neutral."**
   Δ(loop − untied_matched) is **non-significant in every one of the 9 cells** (largest is +0.099
   at d=20,k=5, 2/0/8-ties, p=.5 — the loop edges um only because um fails on 2 of 10 seeds while the
   loop is perfect 10/10; a robustness gap, not a significant accuracy delta). Where there
   is separation (d=20, k=4/k=5), the *depth* delta Δ(um − ff) carries the **same sign and
   significance** as Δ(loop − ff): both deep arms beat the shallow MLP and **tie each other**. The
   active ingredient on parity is **depth, not weight tying** — now confirmed across the d=20
   k-ladder, not one cell.
3. **Does `ff_matched` fail more with k and with distractor load? With k, cleanly; with d, it gets
   confounded by a sample-complexity wall.** At fixed d=20, `ff_matched` degrades **monotonically
   with k** (1.000 → 0.772 → 0.503) while the deep arms hold at 1.000 — exactly the predicted
   "shallow MLP can't represent high-order parity." But **raising d does NOT cleanly stress the
   architecture**: at d=40 (k≥4) and d=80 (k≥4) *every* arm collapses to test-chance, but the
   train/test pattern differs by arm and the failure is **not a single mechanism**. The **deep arms**
   (loop/um/us) fit train at 0.90–1.00 yet score chance on test → a **generalization /
   sample-complexity wall** (k-sparse parity is not identifiable from 4000 rows once the distractor
   count is large). `ff_matched`, by contrast, only reaches **~0.74 train** at d=40,k≥4 → it *also*
   **underfits** there (an optimization/representation limit), so it is not the same overfitting
   story. Either way the regime carries **no recurrence verdict** — no arm separates on test. (The
   blanket "high train acc ⇒ generalization wall" should not be read to cover ff_matched.) d=80,k=3
   sits on the wall's edge: the deep arms (and
   the fat ceiling, 0.842) beat ff on the mean (+0.16) but with 6/4 seed splits and ±0.21 bands —
   suggestive, **not significant**.
4. **Is there any cell where the loop beats BOTH mandatory controls? NO.** The loop beats `ff_matched`
   significantly (d=20, k=4/k=5) but only **ties** `untied_matched` everywhere. Per the milestone's
   own interpretation rule ("trm_nods > ff_matched but trm_nods ≈ untied_matched → the loop has not
   beaten both controls on Task A"), **Task A does not supply a loop-beats-both leg.** The loop's
   defensible property remains *robustness* — it is **never the worst** param-matched arm in any
   cell (it is the *only* arm perfect across the entire d=20 column), but never *dominant*.
5. **Does this change the §9 gate? NO.** Still no task where the loop beats *both* its controls.
   Task A now firmly reads "depth-positive, tying-neutral, robustness-not-dominance," replicated
   across k at d=20. The hierarchy stays **gated** (Task C unbuilt, per the milestone instruction).

**Net.** The Task A parity finding **replicated and is no longer single-config**: the loop's edge
over the shallow §4a control is real, robust across the k-ladder, and *grows* with interaction
order — but it is entirely a **depth** effect (the fair untied stack matches it in every cell), and
the loop beats both mandatory controls in **zero** cells. The d-axis stress test mostly surfaced a
**sample-complexity wall** (d≥40, k≥4 unlearnable for all arms at this budget/sample size) rather
than an architecture separation, so the clean architectural signal lives at d=20 (all k) and, more
noisily, d=80/k=3. Deep supervision (final-state) stays inert across all 9 cells (|Δ(ds − nods)| ≤
0.026, never significant), consistent with M0–M3a. The §9 gate is unmoved.

**Caveats / open gaps.** (i) The harder cells are sample-limited, not capacity-limited — a larger
`n_train` (or a curriculum over k) would be needed to tell whether the d=80 hints are a real
depth/tying edge near the wall or noise; this milestone deliberately did not tune to chase them.
(ii) Task A is now multi-d/multi-k but still one task-family and one architecture size. (iii) The
§9 "beats both controls" condition is still unmet on *either* task — as M2-confirm noted, it may be
literally unsatisfiable by a generalist judged against single-axis specialists; re-judging the gate
wording (not building the hierarchy) is the live question, untouched here.

---

## M5 — DONE. Lift the M4 sample wall (Task A parity, larger n_train). Wall is SAMPLE-bound and lifts to all-solve with no separation; M4's d=80,k=3 hint dissolves; d=80,k=5 is a CAPACITY wall, not sample-bound.

M4's biggest open gap (lever §11(c)(ii)): the d≥40 cells collapsed to test-chance for every arm
at `n_train=4000`, and the **d=80,k=3 "depth hint"** (deep arms +0.16 over `ff_matched`, 6/4
seed splits, ns) sat on the wall's edge — was it a real depth/tying edge that more data would
expose, or just `ff_matched` running out of samples? M5 re-ran M4's **d≥40 sub-block** at a
larger-`n_train` ladder, changing **exactly one knob** vs M4 (`n_train`: 4000 → 16000 → 64000),
holding model size, epochs (100), arms, 10 seeds, and the budget guard fixed. **Zero new code** —
`n_train` is a `TaskConfig` scalar and `d`/`k` are the existing `grid` axis. Configs
`m5_parity_wall_n16k.yaml` (all 6 cells, `d∈{40,80}×k∈{3,4,5}`) and `m5_parity_wall_n64k.yaml`
(**focused** to the 4 cells still mid-transition at 16k, `d∈{40,80}×k∈{4,5}` — the k=3 column had
saturated to 1.000 for every arm, so re-running it at 64k would only reconfirm). 67 tests, ruff
clean (no code touched). Tracked summaries:
`results/m5_parity_wall_n16k_20260621T143402_{curve,deltas,params}.csv` and
`results/m5_parity_wall_n64k_20260621T220534_{curve,deltas,params}.csv` (+ JSON records).
The d=40,k=3 cell reproduces M4 (all arms 1.000), anchoring comparability.

**Test accuracy across the n_train ladder (loop = `trm_nods`; * = at/near chance for matched arms):**

| d | k | 4k (M4) loop / ff / um | 16k loop / ff / um | 64k loop / ff / um |
|---|---|---|---|---|
| 40 | 3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | — (saturated at 16k, not re-run) |
| 40 | 4 | 0.51 / 0.50 / 0.57 * | **1.00 / 1.00 / 1.00** | 1.00 / 1.00 / 1.00 |
| 40 | 5 | 0.49 / 0.50 / 0.53 * | 0.50 / 0.62 / 0.55 *(messy, ±.20)* | **1.00 / 1.00 / 1.00** |
| 80 | 3 | **0.70 / 0.53 / 0.67** *(M4 "depth hint")* | **1.00 / 1.00 / 1.00** | — (saturated at 16k, not re-run) |
| 80 | 4 | 0.50 / 0.50 / 0.50 * | **0.60 / 0.50 / 0.51** *(loop-hint, ±.21)* | **1.00 / 1.00 / 1.00** |
| 80 | 5 | 0.50 / 0.51 / 0.51 * | 0.50 / 0.51 / 0.50 * | 0.50 / 0.59 / 0.50 *(still walled)* |

**Key paired Δ (accuracy, 10 seeds; sign-test p).** In every cell that lifts, all arms reach
1.000 together, so the lifted-cell Δs are exactly 0. The informative Δs are at the transition:

| cell | n_train | Δ(loop − ff) | Δ(loop − um) | Δ(um − ff) | Δ(ds − nods) |
|---|---|---|---|---|---|
| d=80,k=4 | 16k | **+0.103** (8/2, p=.109) | +0.097 (3/6, p=.51) | +0.006 (6/4, p=.75) | −0.005 (ns) |
| d=80,k=4 | 64k | −0.000 (saturated) | −0.000 (saturated) | +0.000 | +0.000 |
| d=40,k=5 | 16k | −0.120 (3/7, p=.34) | −0.045 (4/6, p=.75) | −0.074 (4/6, p=.75) | −0.009 (ns) |
| d=40,k=5 | 64k | +0.000 (saturated) | +0.000 (saturated) | −0.000 | +0.000 |
| d=80,k=5 | 64k | −0.091 (5/5, p=1) | −0.002 (4/5, p=1) | −0.089 (4/6, p=.75) | +0.002 (ns) |

**Reading (per §2/§8 — answering M5's question).**

1. **The d≥40 wall is genuinely SAMPLE-complexity-bound, and lifting it reveals NO architectural
   separation.** Four of the five originally-walled cells (d=40,k=4; d=40,k=5; d=80,k=3; d=80,k=4)
   go from all-chance/partial to **all-arms-solve = 1.000** as `n_train` grows. The transition
   pattern is uniform: `chance(all) → high-variance partial → 1.000(all)`. When the wall lifts,
   **every arm gets there together** — there is no hidden edge behind it. So the d≥40 regime
   carries **no recurrence verdict** (as M4 already cautioned), now confirmed by actually lifting it.
2. **The M4 d=80,k=3 "depth hint" was `ff_matched` sample-starvation, NOT architecture.** At 4k,
   `ff` lagged (0.53) while the deep arms reached ~0.70, manufacturing the +0.16 hint. With 4× data
   **everyone hits 1.000** (16k). The hint dissolves — it was the wall, not depth or tying. This is
   the headline answer to lever §11(c)(ii).
3. **The 16k "d=80,k=4 loop-beats-both hint" was a TRANSIENT sample-efficiency ordering, erased by
   saturation.** At 16k the deep tied arms (loop / `trm_ds`) generalized to ~0.60 while `ff` and
   `um` sat at chance — the only loop>both *direction* on the whole ladder (Δ(loop−ff)=+0.103, but
   ns at 8/2, p=.109; Δ(loop−um) mean +0.097 yet a 3/6 seed split, p=.51 — seed-lottery, ±0.21). At
   64k **all arms reach 1.000**, so it is **not** a stable accuracy edge. There is a *mild, honest
   sub-finding* here — the loop reached generalization at a smaller `n` than the single-axis
   controls at d=80,k=4 — but it is high-variance, non-significant, and vanishes at saturation, so
   it is reported as a hint at most, never a claim. **No significant loop-beats-both cell exists
   anywhere on the 4k→16k→64k ladder.**
4. **d=80,k=5 is the exception: a CAPACITY wall, not a sample wall.** It stays at test-chance even
   at 64k, and crucially **train accuracy DROPS** with more data (loop 0.91→0.73, um 0.97→0.75,
   ff 0.89→0.77): the ~14k-param matched arms can no longer even *fit* 64k rows of the
   (80-choose-5)≈24M-subset parity in 100 epochs (overfit→underfit flip). Even the 4× `untied_stack`
   ceiling fits train 0.94 but still tests at chance. So "raise `n_train`" alone does **not** crack
   the hardest cell — it needs a larger model, which is out of scope (would confound the budget).
   `ff_matched` shows the same flaky high-variance partial generalization here (0.59 ± 0.20) that
   d=40,k=5 showed at 16k — a couple of lucky seeds, not a verdict.
5. **Tying stays neutral and DS stays inert at scale.** Δ(loop − um) ≈ 0 in every solved cell
   (largest |·| is +0.0001), and |Δ(ds − nods)| ≤ 0.009 across all cells/rungs — both consistent
   with M0–M4. The §9 gate is **unmoved**: no cell, at any `n_train`, where the loop beats *both*
   controls.

**Budget audit.** `untied_matched` drifts to +2.3% (ratio 1.023) at d=40 (the expected integer
width-quantization, surfaced in the params CSV not hidden); all d=40 cells are saturated so no Δ
rides on it. d=80 arms are matched to ≤0.7%.

**Net.** M5 closes the M4 sample-wall gap cleanly: the d≥40 wall is predominantly
**sample-complexity-bound and lifts to all-arms-solve with no architectural separation**, the two
"hints" M4/M5-16k surfaced (d=80,k=3; d=80,k=4) are both explained as transition artifacts
(data-starvation / transient sample-efficiency ordering), and the single cell that does *not* lift
(d=80,k=5) is **capacity-bound**, not sample-bound, so more data is the wrong lever there. Task A's
verdict is unchanged and now stress-tested across an `n_train` ladder: **depth-positive (at d=20,
M4), tying-neutral, robustness-not-dominance, loop-beats-both in zero cells.** The §9 gate remains
unmet on Task A.

**Caveats / open gaps.** (i) The depth-positive Task A signal still lives only at d=20 (M4) — the
d≥40 cells either lift to all-solve (no separation) or stay capacity-walled (d=80,k=5), so raising
`n_train` did not surface a *new* depth/tying separation; it dissolved the apparent ones. (ii)
d=80,k=5 would need a bigger model to probe, deliberately not done (confounds the budget). (iii) The
§9 gate is still unmet on either task; M5 strengthens the M2-confirm suspicion that "beats both
single-axis controls" may be unsatisfiable by a generalist — **re-judging the gate wording is now
the highest-value live question** (do NOT build Task C on this evidence).

---

## M6a — DONE. The both-axes probe (multi_parity). §9 gate is empirically UNSATISFIABLE by the generalist; loop is depth-positive, NOT a robust generalist (the "never-worst" property is falsified).

The §11(c)(i) lever, run as an experiment rather than settled by fiat. After M0–M5 the §9
gate ("loop beats BOTH controls on A AND B") was unmet for a *structural* reason: each
canonical task needs exactly ONE axis (A→depth, B→width), so the single-axis control owning
that axis always TIES the loop. M6a builds the one task that needs **both depth and width at a
fixed tiny budget** — exactly where a generalist *should* beat both specialists — and asks
empirically whether a loop-beats-both cell exists at all.

**Task = `multi_parity`** (new generator, determinism-tested): predict `w` **independent**
k-parities in parallel from the same `d` bits. Depth axis = each output is order-`k` (shallow
`ff_matched` should fail at k≥4, per M4); width axis = `w` parallel computations (narrow
`untied_matched` blocks should bottleneck). NOT Task C — the `w` parities are independent, no
sub-problem feeds another; `w=1` reduces exactly to Task A (sanity anchor, asserted in tests).
Grid **k∈{3,4} × w∈{1,4,8}** at d=20, 5 arms (4 required + `untied_stack` ceiling), 10 seeds,
hyperparameters inherited verbatim from `m4_parity_grid.yaml` (no per-cell tuning, §8). New
code: `make_multi_parity` (+6 tests), one `make_splits` branch, the `TaskConfig.name` literal,
one config. 80 tests, ruff clean. Tracked:
`results/m6a_multi_parity_grid_20260622T080206_{curve,deltas,params}.csv` (+ JSON).
**Budget parity CLEAN** — all matched arms within ±0.7% in every cell (no width-quantization
breach; the answer rides on no confound).

**Per-arm test accuracy (token-acc, 10 seeds; us = untied_stack ceiling) and the two headline Δs:**

| k | w | baseline | loop (nods) | ff_matched | untied_matched | us (ceiling) | Δ(loop−ff) | Δ(loop−um) |
|---|---|---|---|---|---|---|---|---|
| 3 | 1 | .512 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 (tie) | 0.000 (tie) |
| 3 | 4 | .506 | 0.852 | **1.000** | 0.901 | 0.950 | **−0.148** (0/9, p=.004) | −0.049 (4/5, ns) |
| 3 | 8 | .503 | 0.793 | **0.982** | 0.733 | 0.843 | **−0.189** (0/10, p=.002) | +0.059 (7/3, ns) |
| 4 | 1 | .513 | 1.000 | 0.772 | 1.000 | 1.000 | **+0.228** (6/0, p=.031) | 0.000 (tie) |
| 4 | 4 | .508 | 0.827 | 0.557 | 0.810 | 0.931 | **+0.269** (10/0, p=.002) | +0.016 (7/3, ns) |
| 4 | 8 | .504 | 0.715 | 0.533 | 0.680 | 0.775 | **+0.182** (10/0, p=.002) | +0.035 (7/3, ns) |

**Reading (per §2/§8 — the pre-registered honesty clause fires; outcome (b)).**

1. **ZERO loop-beats-both cells.** The pre-registered condition (Δ(loop−ff) > 0 AND Δ(loop−um) > 0,
   *both* sign-test p<.05) is met in **no cell**. Where Δ(loop−ff) is strongly significant (k=4,
   all w: +0.18…+0.27, 10/0, p=.002), Δ(loop−um) is +0.016…+0.035, **ns** (7/3, p=.34) — the loop
   ties the deep control even with the width axis maximally stressed (w=8). Stressing both axes at
   once did **not** create the regime where tying pays; the two deep arms (loop, um) degrade
   *together* as w grows.
2. **`multi_parity` UNIFIES Task A and Task B into one task; k is the axis dial.** k=4
   (depth-demanding) reproduces the **Task A** pattern — loop beats shallow ff (depth), ties deep
   um (tying neutral) — and now **extends it to multi-output** (w=4, w=8). k=3 (easy parity, depth
   *not* needed) reproduces the **Task B** pattern — the wide shallow `ff_matched` is the **best**
   arm and the loop is **significantly beaten by it** (−0.148/−0.189, p≤.004). So at every tested
   operating point one single-axis specialist matches-or-beats the loop on the axis that matters
   and the other specialist is irrelevant. **In no tested cell does the generalist beat both**, and
   the k-dial argument *explains why*. **Caveat (adversarial review, do not overclaim):** this is
   "unsatisfied in every tested cell," NOT a proof of impossibility. In the k=4 cells the loop does
   edge `untied_matched` in the *predicted* direction (+0.016 w=4, +0.035 w=8) but
   **non-significantly** (7/3 seeds, p=.34) — under-powered, not a demonstrated tie-in-principle;
   and the grid is coarse (`w` 1→4→8, one d, one budget, one model size). Higher seeds on the
   k=4/w≥4 cells (or finer/larger `w`) would settle whether the gate is truly unsatisfiable or
   merely unmet here. The defensible reading: the gate is **unsatisfied in every tested cell with a
   structural reason (the k-dial)** — strengthening, not proving, the M2-confirm suspicion that the
   wording should change.
3. **The "tying buys width" half of the M2 synthesis does NOT replicate on the parity family.**
   Δ(loop−um) is ns in every multi-output cell — adding width pressure to parity does **not** make
   the fair untied stack fail relative to the loop, unlike CA (M2-confirm, where um was worst in
   6/6). So the CA tying advantage was **CA-specific**, not a general "tied recurrence buys width";
   on parity tying is neutral whether w=1 (M4) or w=8 (M6a). Both halves of the M2 synthesis are
   now retracted on parity (depth half already softened in M3a; width half here).
4. **The loop's last defensible property — "never the worst param-matched arm" (M2-confirm/M4) —
   is FALSIFIED.** At k=3,w∈{4,8} the loop is *significantly* beaten by `ff_matched` (a mandatory
   param-matched control), and at k=3,w=4 it is nominally the worst of the three param-matched arms
   (0.852 < um 0.901 < ff 1.000). Once depth is unneeded and width dominates, the loop is a
   middling generalist, not Pareto-safe. The honest residual claim is narrower than "robust":
   **the loop owns the *depth* axis** (beats shallow ff when interaction order is high; replicated
   M0/M2/M4 and extended to multi-output here) and is **tying-neutral** — it is depth-positive, not
   a robust all-rounder.
5. **DS inert, ceiling behaves.** |Δ(trm_ds − trm_nods)| ≤ 0.036, never significant (consistent
   M0–M5). The 4× `untied_stack` ceiling is the best arm in the hard k=4,w≥4 cells (capacity helps,
   as expected) — a labelled reference, not a control.

**Net (§9).** The user-chosen "resolve empirically first" path is resolved as far as 6 cells can:
a task built to need both axes yields **zero loop-beats-both cells in every tested cell**, because
its difficulty collapses onto a single dial (k) that hands the relevant specialist a tie (k=4) or a
win (k=3). This is **strong (not conclusive) evidence** the literal §9 gate is unsatisfiable by a
single-loop generalist judged against single-axis specialists — enough to motivate **relaxing the
wording**, with the under-powered k=4 width cells (finding above) the place to push if a stricter
proof is wanted. M6a additionally shows the natural fallback ("loop never significantly worse across
both tasks") is **definitively unmet** (k=3 wide, p≤.004 — this one IS conclusive). The earned,
defensible statement of the loop's value is: **depth-positive on high-order parity, tying-neutral,
no depth-extrapolation (M1/M3b/M7), not a robust generalist.** Do **not** build Task C on this.

**Caveats / open gaps.** (i) Token-acc is the meaningful metric at w=8 (whole-row EM is low for
all arms, ~0.05, over 8 outputs). (ii) One architecture size; the regime is below the
overfit/sample walls of M4/M5 (no all-arms-at-chance cell here, so every cell carries a verdict).
(iii) The remaining §11(c) levers (broaden M3b across rules/widths; a PonderNet/ACT halting
objective vs the extrapolation null) are untouched and now lower-value, since M6a settles the
top lever (i) negatively.

---

## M7 — DONE. Progressive-loss training (Deep Thinking) vs the depth-extrapolation null (Task B). New mechanism, clean NEGATIVE — and a principled reason: CA is non-convergent, so path-independence is the wrong bias.

First **new-mechanism** milestone (post-M6a, after a literature search added §12's depth-extrapolation
references). The target is the project's most stubborn null (M1/M3b): over-unrolling R′>R decays to
baseline and OOD depth T_test>T_train collapses for every arm — "the loop never settles a stable step
operator." The literature names this **"overthinking"** and a targeted fix: **Deep Thinking nets**
(Bansal et al. 2022, arXiv 2202.05826) = **recall** (re-inject the input every step — TRM already does
via `cat[X,z,a]`) + **progressive loss** (per batch run `(T−k)` steps with gradients **detached**, then
`k` steps **with** gradient, supervising only the grad steps; forces an *iteration-count-independent /
path-independent* operator, the property Anil et al. 2022, arXiv 2211.09961 tie to upward generalization).

New substrate (additive, bit-identical when unused): `TRM.forward(init_state=, return_state=)` so a
rollout can be detached and resumed (composition test: `n+m` steps == `n` then resume `m`, atol=0);
`train_progressive` with two alignments — `progressive_final` (k grad steps ↔ s_T) and
`progressive_step` (k grad steps ↔ s_{T−k+1..T}, combining M3b step-alignment with the DT detach);
`ModelConfig.ds_mode` + `progressive_alpha` (mix of the progressive and standard full-T terms, 0.5).
Task config = M1/M3b **exactly** (rule 30, w=9, distractors=4, curriculum T~{1..8}), so the
extrapolation curves are directly comparable to M3b's collapse. Six arms (4 loop variants isolating
each knob + ff/um grounding), 10 seeds, 100 epochs. 87 tests, ruff clean, budget within ±1.6%. Tracked:
`results/m7_progressive_extrapolation_20260622T084111_{curve,deltas,extrapolation,extrapolation_deltas,params}.csv`.

**Extrapolation diagonal (R′=T) test accuracy, 10 seeds (baseline ≈ .505):**

| T = R′ | baseline | trm_prog_step | trm_prog_final | trm_stepDS | trm_nods | ff | um |
|---|---|---|---|---|---|---|---|
| 4  | .505 | 0.837 | 0.686 | 0.840 | 0.675 | 0.517 | 0.503 |
| 8  | .508 | 0.578 | 0.627 | 0.586 | 0.645 | 0.543 | 0.592 |
| 12 | .508 | 0.511 | 0.522 | 0.509 | 0.524 | 0.520 | 0.524 |
| 16 | .504 | 0.501 | 0.510 | 0.499 | 0.512 | 0.530 | 0.514 |

(The `trm_stepDS` arm reproduces M3b's step-alignment effect **within noise** — it is an independent
re-run with a different arm roster, so the RNG stream differs; numbers are close but not bit-identical
to M3b's, and should not be read as such.)

**Key paired diagonal Δs (sign-test p, 10 seeds):** Δ(prog_final − nods) = **+0.011 (7/3, p=.34, ns)** at
T=4 and **−0.018 (2/8, p=.11, ns)** at T=8; Δ(prog_step − stepDS) = **−0.003 (4/6, p=.75, ns)** at T=4.
At OOD T=12/16: Δ(prog_step − nods) = −0.013 / −0.011 (ns), Δ(prog_final − nods) = −0.002 / −0.002 (ns).

**Reading (per §8 — the honesty clause; the mechanism is INERT here).**

1. **The progressive detach adds NOTHING — and this holds IN-DISTRIBUTION, which needs no extrapolation
   or convergence argument to interpret.** Δ(prog_final − nods) is ns at T=4 *and* T=8; Δ(prog_step −
   stepDS) is ns at T=4. The progressive arms **collapse onto their non-detach counterparts**
   (prog_final ≈ nods, prog_step ≈ stepDS) — operator fidelity is unchanged (prog_step T=4 0.837 ≈
   stepDS 0.840). The lone significant positive, Δ(prog_step − nods)=+0.162 at T=4 (10/0, p=.002), is
   **purely the M3b step-alignment effect**, reproduced within noise — not the new mechanism. *This is
   the load-bearing negative: the detach is inert wherever we can measure it, full stop.*
2. **Progressive loss does NOT crack the OOD collapse either — the M1/M3b null STANDS.** At T=12 and
   T=16 *every* arm, both progressive variants included, sits at baseline (~0.50–0.52); no Δ significant.
3. **Over-unrolling still decays for the progressive arms too** (T=4 task at R′=8: prog 0.49 = nods 0.49).
   Progressive loss did **not** instill a stable fixed point in this setting.
4. **A HYPOTHESIS for *why* (NOT tested in this milestone).** Deep Thinking's progressive loss is
   designed to instill **path-independence** — converge to a fixed *attractor* and stay there under
   over-unrolling. Task B's CA is a **non-convergent** map: `s_T` is a *moving* target (`s_{T+1} ≠ s_T`,
   no attractor), so over-unrolling *should* move away from `s_T` and a steady-state bias may be
   **mismatched to the task** (the mechanism's home turf is mazes/Sudoku, which have a stable answer).
   **This is a plausible rationalization, not a demonstrated mechanism** — M7 has no convergent-target
   control to contrast against, so it cannot distinguish "wrong bias for this task" from "mechanism
   inert at this scale/tuning." It is offered as the lead hypothesis to test next (see Net), and it is
   *consistent with* but not proven by the over-unroll decay in point 3. The simpler, sufficient
   statement of the result is point 1: **the detach is inert here.**

**Net (§9 unmoved).** A clean negative for a genuinely new mechanism: the progressive detach is inert
in-distribution and fails to crack the OOD collapse. The leading (untested) hypothesis for *why* — the
CA target is non-convergent, so path-independence is the wrong bias — yields a concrete next lever: to
fairly test a path-independence mechanism, build a **fixed-point-target** task (iterate-to-convergence:
connectivity / shortest-path / a *converging* CA) where over-unrolling SHOULD hold, then re-apply
`train_progressive` and compare. Until that control exists the non-convergence claim is a hypothesis,
not a finding. The loop's verdict is unchanged: depth-positive on parity (M4), tying-neutral, **no
depth-extrapolation**, not a robust generalist (M6a).

**M7b — α=1 (pure progressive, no anchor term) CONFIRMS the null (adversarial-review follow-up).**
Re-ran M7 with `progressive_alpha: 1.0` on both progressive arms (baselines bit-identical — nods
0.645, stepDS 0.586, um 0.592 reproduce exactly, re-anchoring the comparison). Config
`m7b_progressive_alpha1.yaml`; tracked
`results/m7b_progressive_alpha1_20260622T090114_*`. **α=1 is strictly WORSE, not better:** diagonal
prog_step T=4 0.759 (vs 0.837 at α=0.5), prog_final T=8 0.594 (vs 0.627); removing the full-T anchor
*hurt*. In-distribution the pure-progressive detach now **significantly underperforms** its non-detach
counterpart — Δ(prog_step − stepDS) = **−0.082 (0/10, p=.002)** at T=4 (was ns at α=0.5) and
Δ(prog_final − nods) = **−0.051 (0/10, p=.002)** at T=8. **OOD still collapses to baseline** (T=12/16:
all Δ ns, arms at 0.50–0.52). So the one untested knob the review flagged does not rescue
extrapolation — it makes the mechanism actively harmful in-distribution while leaving the OOD null
intact. The "you didn't tune α" objection is **closed**: both α∈{0.5, 1.0} give an OOD null, and the
more-faithful pure-progressive variant is worse, not better.

**Remaining caveats.** (i) Still one rule (30) / one width (9) / 100 epochs; α now swept {0.5,1.0}.
(ii) **Unequal compute, not disclosed in v1:** the progressive arms run ~2× the gradient
forwards per batch (a `k`-step progressive forward *and* a full-`T` anchor forward) vs the single
`T`-step forward of `nods`/`stepDS`. This cuts *toward* the null (progressive had more signal and still
tied), so it does not threaten the conclusion, but the comparison is not compute-matched. (iii) Budget:
`untied_matched` sits at **+1.59%** over the loop's budget (params CSV) — over budget, which (as in
M3a/M4) handicaps the control, not the loop. (iv) `progressive_step` needs `deep_supervision: true`
(per-step readouts); `progressive_final` does not.

---

## M8 / M8b — DONE. Variable-compute FIXED-POINT task (converging CA). Adaptive compute FAILS (falsifies the M7 hypothesis); but the FIRST (replicated) loop-beats-both surfaces — on whole-row exact-match.

The adaptive-computation angle. Every prior task has a **fixed** required depth, so a fixed-depth
`untied_matched` always matches the loop and tying is neutral (M2…M6a). M8 builds the one regime never
tested: required compute **varies per instance and can exceed any fixed depth** — an
iterate-to-convergence / fixed-point task, where only the loop can spend variable compute (unroll more
at test). Two payoffs: (a) the first real shot at the loop beating BOTH fixed-compute controls via
adaptive computation; (b) a direct test of the M7 hypothesis (on a CONVERGENT target, over-unrolling
should HOLD and progressive loss should fire).

New task `converge` (generator `make_converge`, determinism + fixed-point + trajectory tests): map s0
to the CA's **fixed point** s_inf (iterate a *converging* rule until `ca_step(s)==s`). Verified
non-degenerate (rule 92, w=32: majority baseline 0.562; ~99% of rows need >1 step; convergence depth
varies per instance — ~45% need >4 steps, ~13% >6, max ~19). `make_trajectory_dataset` now dispatches
to `make_converge`; one `make_splits` branch; `TaskConfig.name` literal extended. 92 tests, ruff clean.
Tracked: `results/m8_converge_adaptive_20260622T102356_*` (rule 92, w=32, the R′ over-unroll sweep),
`results/m8b_converge_grid_20260622T120355_*` (replication grid rule∈{13,78,92}×w∈{24,32}), and
`results/m8c_converge_fair_*` (the FAIR-supervision re-analysis that isolates tying from the step-aligned
DS confound flagged in adversarial review — adds `untied_stepDS` and the clean `nods−untied` delta).
94 tests after the M8c hardening (assert the fixed point; parametrize the test over rules 13/78/92).

**M8 — over-unrolling DECAYS even on a convergent target (the adaptive-compute headline FAILS):**

| R′ (test unroll) | trm_stepDS | trm_prog_step | trm_nods | ff_matched | untied_matched |
|---|---|---|---|---|---|
| 6 (trained) | 0.922 | 0.919 | 0.896 | 0.920 (flat) | 0.834 (flat) |
| 8  | 0.909 | 0.905 | 0.884 | 0.920 | 0.834 |
| 12 | 0.885 | 0.878 | 0.862 | 0.920 | 0.834 |
| 16 | 0.871 | 0.860 | 0.847 | 0.920 | 0.834 |
| 24 | 0.857 | 0.842 | 0.828 | 0.920 | 0.834 |

**M8b — the trained-depth EXACT-MATCH picture, replicated across the grid (10 seeds; maj baseline 0.562):**

| cell | stepDS EM | ff EM | untied EM | Δ(stepDS−ff) EM | Δ(stepDS−untied) EM | Δ(stepDS−ff) acc |
|---|---|---|---|---|---|---|
| rule13,w24 | 0.403 | 0.307 | 0.100 | +0.096 (8/2, p=.11 ns) | +0.303 (10/0, p=.002) | +0.003 ns |
| rule13,w32 | 0.102 | 0.101 | 0.038 | +0.001 (ns) | +0.064 (10/0, p=.002) | −0.007 (1/9, p=.02 *worse*) |
| rule78,w24 | 0.427 | 0.311 | 0.126 | **+0.116 (10/0, p=.002)** | +0.301 (10/0, p=.002) | +0.005 (10/0, p=.002) |
| rule78,w32 | 0.169 | 0.121 | 0.037 | **+0.048 (9/1, p=.021)** | +0.132 (10/0, p=.002) | +0.002 (ns) |
| rule92,w24 | 0.418 | 0.331 | 0.115 | **+0.087 (9/1, p=.021)** | +0.303 (10/0, p=.002) | +0.003 (ns) |
| rule92,w32 | 0.162 | 0.112 | 0.037 | **+0.050 (10/0, p=.002)** | +0.125 (10/0, p=.002) | +0.002 (ns) |

**Reading (per §2/§8).**

1. **Adaptive computation FAILS — and it FALSIFIES the M7 non-convergence hypothesis.** Over-unrolling
   the loop **decays** (stepDS 0.922→0.857 as R′ 6→24) **even though the target is a genuine fixed
   point**. The loop does not hold the fixed point and cannot turn extra test-time steps into solving
   the deep tail. Critically, M7 had *hypothesized* the rule-30 over-unroll decay was caused by CA
   being non-convergent — M8 shows the decay happens on a CONVERGENT target too, so it is **intrinsic
   to how the loop bakes in its trained depth, NOT a property of task convergence.** The M7
   rationalization is dead (correctly, it was labelled untested). Adaptive compute is not a free lunch
   the loop can exploit here.
2. **A robust, properly-isolated TYING-POSITIVE: the weight-tied loop beats a FAIR untied stack on
   whole-row coherence, 6/6 cells, at EQUAL supervision (M8c).** [M8b first reported this through
   `trm_stepDS` vs `untied_matched` — a §8 confound, since stepDS got step-aligned DS (6 targets/batch)
   and untied got only final loss; flagged by adversarial review.] **M8c** (`m8c_converge_fair.yaml`)
   computes the clean isolation: `Δ(trm_nods − untied_matched)` (both final-loss) AND
   `Δ(trm_stepDS − untied_stepDS)` (both step-aligned — `untied_stepDS` = the fair untied stack *given*
   the same step-aligned DS). Both are positive **10/0, p=.002 in ALL 6 cells**: nods−untied EM +0.05…
   **+0.37** (token-acc +0.06…+0.09), stepDS−untied_stepDS EM +0.07…+0.30. So tying buys whole-row
   coherence at equal supervision, robustly — the cleanest tying-positive in the project (parity was
   tying-neutral, M4/M6a). It is **tying, not depth**: `untied_matched` is *deep* (6 blocks) yet WORST
   on EM everywhere — depth alone doesn't buy coherence, the weight-tied reuse does.
3. **The loop beats BOTH controls only in 3/6 cells (all w=24), and only on EM — the w=32 "ff win" was
   SUPERVISION, not architecture.** The clean loop-beats-both is `trm_nods` (equal supervision) >
   *both* ff and untied: it holds at **w=24 for all 3 rules** (nods−ff EM +0.11…+0.17 p≤.021;
   nods−untied EM +0.32…+0.37 p=.002) but **fails at w=32**, where nods−ff is ns/negative on EM and
   *significantly negative on token-acc* (−0.015…−0.024, p=.002 — the plain loop LOSES to wide shallow
   ff at w=32). M8b's "4/6 via stepDS" over-counted: at w=32 only the step-DS arm clears ff, i.e. that
   half was carried by richer supervision the controls weren't given, not by the architecture. Honest
   count: **clean loop-beats-both = 3/6 (w=24); robust tying-over-untied = 6/6.**
4. **Progressive loss still adds nothing** (Δ(prog_step − stepDS) ns in all 6 cells, M8b) — consistent
   with M7/M7b: the progressive detach is inert wherever measured.

**Net (corrected after M8c).** A layered result. The headline angle (adaptive computation) is a **clean
negative** that **kills the M7 hypothesis** (over-unroll decay is intrinsic, not convergence-related).
But pursuing it surfaced a genuine, properly-isolated **tying-positive**: the weight-tied loop beats a
*fair untied stack* on whole-row coherence in **all 6 cells at equal supervision** (the cleanest such
result in the project; parity was tying-neutral). The stronger "loop beats **both** controls" claim is
real but **narrower than M8b stated** once the supervision confound is removed: clean (plain loop, equal
supervision) it holds in **3/6 cells, all w=24, on exact-match only**. So there *is* a regime
(fixed-point multi-output, whole-row metric, w=24) where the loop beats both param-matched controls — a
concrete counterexample to "the loop never beats both anywhere" — but it does not generalize to w=32
(where the plain loop loses to wide shallow ff) and is EM-only. It does not satisfy the literal §9 gate
(names Tasks A/B).

**Caveats (adversarial-proofed; S1/S2 resolved by M8c).** (i) **EM-only**; on token-acc the loop ties
ff at w=24 and *loses* to ff at w=32 — not a both-metric dominance. (ii) **Width-dependent**: clean
loop-beats-both is w=24 only (3/6); at w=32 the ff-axis win needs step-aligned DS the controls weren't
given. (iii) The robust claim is **tying > fair untied (6/6)**, isolated at equal supervision two ways
(M8c) — *that* is the defensible architectural finding; "beats both" is the narrower corollary.
(iv) **`untied_matched` is +2.5%/+3.1% OVER budget** (width-quantization, flagged `within_tol=False`) —
but it loses, so over-budget is conservative; the loop>ff comparisons are budget-clean (ff ≤0.2%).
(v) One model size, one operator family (majority-type converging ECAs), n_train=4000. (vi) The
mechanism is whole-row coherence from recurrence/tying, **not** adaptive computation — do not conflate.

---

## M9 — DONE. Width sweep + coherence-mechanism diagnostic (converge). The M8 tying-positive STRENGTHENS (loop-beats-both is a w≤24 regime, not a w=24 knife-edge), and the "whole-row coherence" mechanism is CONFIRMED at matched token-accuracy.

Pursued the project's one pro-loop result (M8/M8c: the weight-tied loop beats a *fair untied stack*
on whole-row exact-match) along §11(c)'s two named sub-levers: **(ii)** sweep the output width `w` to
map *where* "loop-beats-both" survives and *why* the M8 w=24→w=32 edge fades, and **(iii)** add a
**coherence diagnostic** that directly tests *whether* tying buys whole-row coherence beyond raw
token-accuracy. One experimental knob vs M8c (`w`), one additive metric. New metric
`coherence_excess = EM − token_acc**w` (observed whole-row score minus the EM expected if per-cell
errors were *independent* at the same token-acc; >0 ⇒ errors clustered into fewer rows = coherent),
plus a `mean_wrong_per_row` companion — both derived from the existing single prediction pass in
`evaluate`, **bit-identical** for prior metrics, threaded through `run.py` (per-seed → aggregate →
sign-tested paired Δ → curve/deltas CSVs) exactly like `exact_match`. Task = `converge`, **rule 78**
fixed (cleanest loop>ff in M8b; M8b showed rules behave alike, so fixing rule + sweeping `w` is the
clean single-knob design), `w ∈ {12,16,24,32,48}`, M8c fair-supervision arms (`trm_nods`,
`trm_stepDS`, `ff_matched`, `untied_matched`, `untied_stepDS`), 10 seeds, 100 epochs. 97 tests
(+2 coherence-math unit tests), ruff clean. Tracked:
`results/m9_converge_width_20260622T050349_{curve,deltas,params}.csv` (+ JSON). **Sanity anchor —
the `w=24`/`w=32` cells reproduce M8b/M8c's rule-78 numbers BIT-FOR-BIT** (stepDS EM 0.427/0.169,
ff EM 0.311/0.121, untied EM 0.126/0.037), confirming the additive metric perturbed nothing.

**Per-arm exact-match (EM) and token-accuracy (acc), 10 seeds (baseline acc ≈ 0.562):**

| w | nods EM / acc | stepDS EM / acc | ff EM / acc | untied EM / acc | untied_stepDS EM / acc |
|---|---|---|---|---|---|
| 12 | 0.911 / 0.988 | 0.833 / 0.972 | 0.638 / 0.950 | 0.854 / 0.979 | 0.902 / 0.986 |
| 16 | 0.828 / 0.981 | 0.759 / 0.974 | 0.549 / 0.952 | 0.464 / 0.929 | 0.540 / 0.944 |
| 24 | 0.444 / 0.944 | 0.427 / 0.947 | 0.311 / 0.941 | 0.126 / 0.867 | 0.135 / 0.874 |
| 32 | 0.107 / 0.899 | 0.169 / 0.923 | 0.121 / 0.921 | 0.037 / 0.834 | 0.040 / 0.841 |
| 48 | 0.008 / 0.861 | 0.009 / 0.872 | 0.016 / 0.892 | 0.004 / 0.789 | 0.004 / 0.802 |

**Headline paired Δs (sign-test p, 10 seeds). nods/untied/ff = equal (final-loss) supervision:**

| w | Δ(nods−untied) EM | Δ(nods−ff) EM | Δ(nods−ff) acc | Δ(coh: nods−untied) | Δ(coh: nods−ff) |
|---|---|---|---|---|---|
| 12 | +0.057 (10/0, .002) | +0.273 (10/0, .002) | +0.038 (10/0, .002) | −0.033 (2/8, .11 ns) | −0.048 (0/10, .002) |
| 16 | +0.364 (10/0, .002) | +0.279 (10/0, .002) | +0.028 (10/0, .002) | −0.057 (0/10, .002) | +0.008 (6/4, .75 ns) |
| 24 | +0.318 (10/0, .002) | **+0.133 (10/0, .002)** | +0.003 (6/4, .75 ns) | **+0.090 (10/0, .002)** | **+0.107 (10/0, .002)** |
| 32 | +0.070 (10/0, .002) | −0.014 (5/5, 1.0 ns) | **−0.021 (0/10, .002)** | +0.038 (10/0, .002) | +0.024 (8/2, .11 ns) |
| 48 | +0.005 (9/1, .021) | −0.008 (1/9, .021) | **−0.031 (0/10, .002)** | +0.004 (9/1, .021) | −0.004 (3/7, .34 ns) |

**Reading (per §2/§8 — answering M9's three pre-registered predictions).**

1. **P1 (tying robustness) — CONFIRMED.** `Δ(loop − fair untied)` is **positive on token-acc in all
   5 widths, both supervision regimes (10/0, p=.002 every cell)**, and **positive on EM in 9/10
   width×regime cells** — the lone exception is `Δ(stepDS − untied_stepDS)` EM at w=12 (−0.069, 3/7,
   p=.34, ns) where the task is near-saturated (all arms 0.83–0.91 EM, no room). So the M8c
   tying-positive is **width-robust, not a w=24/32 artifact** — it holds from w=12 to w=48. This is
   the strongest, cleanest leg: weight tying beats a fair untied stack at fixed budget across the
   whole width range. (`untied_matched` is the **weakest** param-matched arm on EM in every cell.)
2. **P2 (loop-beats-both boundary) — CONFIRMED, and the regime is WIDER than M8c reported.** The
   *clean* loop-beats-both (plain `trm_nods` at equal supervision beats **both** `untied_matched`
   AND `ff_matched`, both sign-test p<.05 on EM) holds at **w=12, 16, AND 24** — a contiguous
   **w≤24 regime**, not the single w=24 cell M8c's coarse grid surfaced — and **vanishes by w≥32**
   (at w=32 the loop ties ff on EM and *loses* on token-acc −0.021, 0/10, p=.002; at w=48 it loses ff
   on both). On token-acc the wide shallow MLP overtakes the loop monotonically (Δ(nods−ff) acc:
   +0.038 → +0.028 → +0.003(ns) → −0.021 → −0.031), crossing over at ~w=24; **EM (coherence) is the
   loop's durable edge, extending its competitiveness one width-step past where token-acc crosses.**
   With step-aligned DS the loop>ff EM edge stretches to w=32 (Δ(stepDS−ff) EM +0.048, 9/1, p=.021)
   — the *supervision-carried* half M8c flagged; the clean (equal-supervision) regime is w≤24.
3. **P3 (mechanism) — the clean, unconfounded statistic is EM-AT-MATCHED-TOKEN-ACC, and on that
   statistic the mechanism holds. The `coherence_excess` metric is DEMOTED to a per-arm descriptor
   after an adversarial review (see below).** The load-bearing evidence is **loop vs ff at w=24,
   where token-acc is matched** (Δ(nods−ff) acc +0.003, **ns**): at *equal per-cell accuracy* the
   loop wins whole-row **EM by +0.133 (10/0, p=.002)**. With per-cell accuracy held equal, the only
   thing left to differ is how errors are distributed across rows — so the loop is producing coherent
   whole rows the shallow MLP cannot. This is the direct mechanism demonstration the project lacked,
   and it needs **no coherence metric** — it is a plain EM comparison at matched token-acc.
   **Adversarial-review correction (do not repeat the original framing):** the first write-up also
   cited Δ(`coherence_excess`) = +0.107 (loop−ff) as a *second, independent* confirmation. It is
   **not independent** — `coherence_excess = EM − token_acc**w`, so at matched token-acc the baseline
   term cancels and Δ(coh) ≡ Δ(EM); it is the same fact counted twice. Worse, the **cross-arm Δ of
   `coherence_excess` is confounded two ways** and must not be used as evidence: (a) *level* — a
   lower-acc arm has a lower independence baseline, hence more "room" (this is why at w=16 `untied`
   shows coh 0.153 > loop 0.097 despite far lower EM); (b) *per-row dispersion (Jensen)* — since
   EM = mean_row(row_acc**w) ≥ (mean_row row_acc)**w, heterogeneous per-row difficulty inflates
   `coherence_excess` *even with no clustering*, and matching the *mean* token-acc does not match the
   *variance*, so the anchor does not control it. (A per-row baseline mean_row(row_acc**w) removes
   the Jensen bias but also cancels the cross-row clustering that is the signal, so it was not
   adopted.) **Net on P3:** the mechanism — recurrence/tying buys whole-row coherence — is supported
   *by the EM-at-matched-acc comparison at w=24*, the one clean cell; `coherence_excess` is retained
   only as a per-arm descriptor (its width profile peaks at w≈24 for mechanical reasons — EM
   saturates near 1 at small w and collapses to 0 at large w, so any EM-minus-baseline quantity must
   peak in between). The honesty fork does not fire (the matched-acc EM edge is real), but the
   stronger "two independent signals" reading is **withdrawn**.

**Net.** M9 **strengthens** the M8 tying-positive on both axes it set out to probe. (1) The
tying-over-fair-untied advantage is **width-robust** (token-acc 10/10 cells; EM 9/10), confirming
it is the project's durable architectural pro-loop fact. (2) The clean **loop-beats-both** regime is
**w≤24** (broader than M8c's single w=24 snapshot), bounded above between w=24 and w=32 by the wide
shallow MLP overtaking the loop on token-acc as outputs multiply. (3) The hypothesized **mechanism —
whole-row coherence from recurrence/tying — is supported by the EM-at-matched-token-acc comparison**
(loop vs ff @ w=24: token-acc tied, ΔEM +0.133, 10/0, p=.002): at equal per-cell accuracy the loop
makes coherent whole rows the MLP can't. **[Corrected after adversarial review]** the `coherence_excess`
metric does **not** add an independent confirmation — at matched acc Δ(coh) ≡ Δ(EM), and its cross-arm
Δ is confounded by token-acc *level* and per-row *dispersion* (Jensen), so it is demoted to a per-arm
descriptor and its width "peak" at w≈24 is mechanical. The loop's value statement: **weight-tied
recurrence buys whole-row coherence on multi-output fixed-point targets at a fixed budget — robustly
over a fair untied stack across width (P1), and over a shallow MLP at matched token-acc in a w≤24
regime (rule 78; P2/P3) — but NOT a token-accuracy edge at large `w`, NOT adaptive compute (M8), NOT
depth-extrapolation (M1/M3b/M7).** Still does not satisfy the *literal* §9 gate (names Tasks A/B).

**Caveats / open gaps.** (i) **One rule (78), one model size, n_train=4000** — the width axis is now
well-resolved but the rule/size axes are not (M8c covered rules {13,78,92} at w∈{24,32}). So "w≤24
regime" and "boundary between w=24 and w=32" are **rule-78 statements**; the boundary is not swept
over rule or model size. (ii) **`coherence_excess` is for per-arm description only** — its cross-arm
Δ is confounded by token-acc level *and* per-row dispersion (P3), so the cross-arm mechanism claim
rests on EM-at-matched-token-acc (loop-vs-ff @ w=24), not on any coherence Δ. (iii) **Budget breach
carried forward, and it is NOT uniformly conservative:** `untied` lands ratio 1.025/1.031 (OVER
budget) at w=24/32 — there over-budget *handicaps* the control, so the tying-positive is conservative
— but **0.978 (UNDER budget) at w=12** and 0.983 at w=48, where the control has *fewer* params, so
the w=12 Δ(nods−untied) EM +0.057 is mildly **anti-conservative** (partly a capacity gap favouring
the loop). Since w=12 is load-bearing for "regime wider than M8c," note it: the w=16/24 cells (where
`untied` is at/over budget) are the clean support for that claim, not w=12 alone. The strictly-budget-
clean fix (§11(c)(i)) is still deferred. (iv) A **decoupled-head ablation** (does the *joint*
multi-output readout, vs per-cell-independent heads, drive the coherence?) is the natural M10
follow-up — it needs new model
code, so it was kept out of this single-knob milestone.

---

## M10 — DONE. Decoupled-head ablation (converge). The WHOLE-ROW-COHERENCE mechanism is ISOLATED: the JOINT multi-output state is what buys it — severing cross-cell refinement drops the recurrent model BELOW the shallow MLP.

The deepest remaining mechanism question (§11(c) lever, M9 caveat iv). M9 proved the weight-tied
loop buys *whole-row coherence* (at matched token-acc it makes coherent rows a shallow MLP can't:
loop vs ff @ w=24, ΔEM +0.133, p=.002) but did **not** isolate *why*. The canonical `TRM` refines a
**single shared latent** and feeds the **full flat answer** (all `w` cells) back into every update,
so each output cell's refinement conditions on the current estimate of *every other cell*. That
cross-cell coupling was the obvious candidate for "coherence." M10 severs exactly it.

**New model `TRMDecoupled`** (`src/looptab/models/decoupled.py`, registered `trm_decoupled`):
each output cell carries its **own** latent slice and sees **only its own** answer during refinement
— no cross-cell information flow. *Everything else is held identical to `TRM`*: weight-tied
recurrence (one shared update net reused every step **and** across cells), the same input `X`
re-injected each step (recall), the same per-step readout interface (deep supervision), and the
**same total parameter budget** (per-cell latent width `m` solved to the loop's budget exactly as
`UntiedStackMatched`/`FFMatched`; realized ratios 0.992–1.001, all within ±2%). The axis that differs
is joint-state vs per-cell-state refinement — with one **inherent budget-allocation asymmetry** (not a
hidden confound, but worth stating): to distinguish cells the decoupled head needs a per-cell init
latent `z0` of shape `(w, m)`, which consumes **8–13%** of its budget (vs ~0.4% for the joint loop's
single `z0`), compensated by a wider per-cell net (m≈73–80 vs the joint 64). So "only jointness differs"
is true of the *mechanism* (cross-cell info flow) but the parameter *allocation* necessarily differs;
total budget is matched, capacity is not handicapped (the decoupled net is wider), but it is not a
single-knob byte-for-byte edit. Per-cell identity comes from a small *randomly
initialized* per-cell init latent `z0` (`(w, m)`): with a fully shared net and `X` shared across
cells, a zero `z0` (TRM's choice for its single latent) leaves every cell computing the identical
update forever — the inter-cell symmetry then breaks only slowly through per-cell gradients and the
arm sits at the majority baseline for many epochs; `nn.init.normal_(std=0.02)` differentiates the
cells from step 0 so the decoupled head trains on the same footing as the joint loop (deterministic
— the runner seeds torch before building each arm). 105 tests (+7: shape, param-match, the
no-cross-cell-leakage invariant, state-composition), ruff clean. Config
`m10_decoupled_converge.yaml` = M9's `converge`/rule-78 setup, `w∈{16,24,32}` (straddling the M9
coherence regime), 6 arms (joint `trm_nods`/`trm_stepDS`, decoupled `trm_decoupled_nods`/`_stepDS`,
plus `ff_matched`/`untied_matched` grounding), 10 seeds, 100 epochs. Tracked:
`results/m10_decoupled_converge_20260622T092240_{curve,deltas,params}.csv` (+ JSON). **Sanity anchor
— the joint `trm_nods` vs `ff_matched` reproduces M9 bit-consistently** (w=24: Δacc +0.003 ns / ΔEM
+0.133, 10/0, p=.002; each arm self-reseeds, so the new decoupled arm does not perturb it).

**Per-arm test accuracy / exact-match (EM), 10 seeds (majority baseline ≈ 0.562):**

| w | trm_nods EM/acc | trm_decoupled_nods EM/acc | trm_stepDS EM/acc | trm_decoupled_stepDS EM/acc | ff_matched EM/acc | untied_matched EM/acc |
|---|---|---|---|---|---|---|
| 16 | 0.828 / 0.981 | 0.320 / 0.913 | 0.759 / 0.974 | 0.438 / 0.941 | 0.549 / 0.952 | 0.464 / 0.929 |
| 24 | 0.444 / 0.944 | 0.058 / 0.816 | 0.427 / 0.947 | 0.110 / 0.893 | 0.311 / 0.941 | 0.126 / 0.867 |
| 32 | 0.107 / 0.899 | 0.019 / 0.825 | 0.169 / 0.923 | 0.031 / 0.865 | 0.121 / 0.921 | 0.037 / 0.834 |

**Headline paired Δs (sign-test p, 10 seeds):**

| w | Δ(nods − decoupled_nods) EM | Δ(stepDS − decoupled_stepDS) EM | Δ(decoupled_nods − ff) EM | Δ(decoupled_nods − ff) acc |
|---|---|---|---|---|
| 16 | **+0.508 (10/0, p=.002)** | +0.321 (10/0, p=.002) | **−0.229 (0/10, p=.002)** | −0.039 (0/10, p=.002) |
| 24 | **+0.387 (10/0, p=.002)** | +0.317 (10/0, p=.002) | **−0.254 (0/10, p=.002)** | −0.126 (0/10, p=.002) |
| 32 | **+0.088 (10/0, p=.002)** | +0.138 (10/0, p=.002) | **−0.101 (0/10, p=.002)** | −0.095 (0/10, p=.002) |

**Reading (per §2/§8 — the pre-registered honesty fork resolves cleanly, and harder than predicted).**

1. **The JOINT state is the mechanism — decoupling collapses whole-row coherence (the fork's first
   branch fires).** Severing cross-cell refinement costs the loop **+0.51 / +0.39 / +0.09 EM**
   (w=16/24/32, all 10/0, p=.002) at the same budget, recurrence, recall, and supervision. The pre-
   registered alternative ("decoupled keeps the coherence edge ⇒ recurrence per se drives it") is
   **rejected**: per-cell-independent recurrence does *not* reproduce the coherence. So the M9 "whole-
   row coherence" is specifically a property of **refining all cells together through one shared latent
   with cross-cell answer feedback**, not of weight-tied recurrence in the abstract.
2. **Decoupled recurrence is WORSE than a plain MLP — though this is closer to expected than surprising
   once the mechanism is framed as cross-cell flow.** Δ(decoupled_nods − ff_matched) is **significantly
   negative on BOTH token-acc and EM in all 3 widths** (acc −0.039/−0.126/−0.095; EM −0.229/−0.254/
   −0.101; all 0/10, p=.002). Removing the joint state drops the recurrent model strictly *below* the
   §4a feedforward control, so the recurrence's value here is contingent on the joint multi-output state;
   without it the loop is the worst param-matched arm (the M6a "never-worst is false" reading, sharpened
   — it is the joint coupling, not the loop, that was carrying the value). **Caveat (adversarial review,
   do not overclaim this as a shock):** `ff_matched` is NOT a pure per-cell baseline — its output layer
   maps a shared hidden representation to all `w` cells jointly, so it *too* has cross-cell mixing.
   `trm_decoupled` is the **only** arm with literally zero cross-cell mixing. So "the zero-mixing arm
   loses to a some-mixing MLP" is partly definitional given the mechanism, not an independent surprise;
   the load-bearing evidence remains point 1 (joint loop ≫ decoupled at matched everything-else), not
   "below even an MLP."
3. **Not an optimization artifact of the fragile `nods` arm — the step-aligned pair controls for it.**
   `trm_decoupled_nods` is **optimization-fragile** (high seed variance, e.g. w=24 seed-7 partial
   collapse 0.596; std up to ±0.085) because per-cell identity must be learned from final-loss-only
   supervision against a shared `X`. This *could* inflate the `nods` Δ. But **step-aligned DS makes the
   decoupled arm train stably** (`trm_decoupled_stepDS` std ≤ ±0.024, no collapses) — and the joint
   advantage **persists at equal step-aligned supervision**: Δ(stepDS − decoupled_stepDS) EM +0.321 /
   +0.317 / +0.138 (10/0, p=.002). So the coherence gap holds where both arms optimize well; it is the
   architecture (joint vs per-cell state), not the decoupled arm failing to train.
4. **Anchors reproduce M9 / M8c.** Joint `trm_nods` vs `ff_matched`: w=24 Δacc +0.003 (ns, *matched*
   token-acc) / ΔEM +0.133 (10/0) — the M9 mechanism cell, bit-consistent. Joint > `untied_matched`
   (the tying-positive) on EM in all 3 widths (+0.364/+0.318/+0.069, 10/0, p=.002). The token-acc
   crossover (loop loses ff on token-acc by w=32, −0.021, 0/10) reproduces M9's "EM is the durable
   edge, token-acc crosses ~w=24." DS-mode behaviour is consistent with M9 (step-aligned helps EM here
   because the converge trajectory gives genuine per-step targets).

**Net.** M10 isolates the one pro-loop result in the project to its actual cause: **whole-row coherence
on multi-output fixed-point targets comes from the JOINT refinement state** — all `w` cells sharing one
latent and conditioning on each other's running answer — **not** from weight-tied recurrence per se.
The clean proof is two-sided: (a) a budget/recurrence/supervision-matched model that refines cells
*independently* loses the coherence (ΔEM +0.09…+0.51, 10/0, and still +0.14…+0.32 at equal step-aligned
supervision where it trains stably); (b) that same decoupled model falls *below* the shallow §4a MLP on
both metrics (0/10), so the joint state is not a bonus on top of recurrence — it is the thing carrying
the loop's value. This sharpens the loop's earned value statement to: **tied recurrence with a JOINT
multi-output state buys whole-row coherence**; the "joint" qualifier is now load-bearing and demonstrated.

**Caveats / open gaps.** (i) **One rule (78), one model size, n_train=4000, `w∈{16,24,32}`** — same
single-family scope as M9; the mechanism is shown on rule-78 converge, not swept over rule/size.
(ii) `trm_decoupled_nods` is optimization-fragile under final-loss-only supervision; the *defensible*
controlled comparison is the **step-aligned pair** (both train stably) — lean on that, not the noisier
`nods` Δ, when the trainability objection is raised. (iii) Token-acc is **not** matched between joint
and decoupled (the joint trains to higher acc), so the joint-vs-decoupled EM gap is not a pure matched-
acc coherence measurement the way M9's loop-vs-ff @ w=24 is; the clean mechanism statement rests on
(a)+(b) together (decoupled loses coherence *and* falls below ff). (iv) `untied_matched` is +2.5%/+3.1%
OVER budget at w=24/32 (the carried-forward M9 width-quantization breach, surfaced in the params CSV);
it loses, so over-budget is conservative. The decoupled arm itself is budget-clean (0.992–1.001).

---

## M11 — DONE. Generalize the coherence result across MODEL SIZE and OPERATOR FAMILY. Layered verdict: the joint-state mechanism + tying-positive GENERALIZE across size (and STRENGTHEN with it — NOT a tiny-model artifact); but the whole result is OPERATOR-FAMILY-SPECIFIC — it does NOT transfer to two new converging families, and "loop-beats-both" is capacity-contingent.

The project's sole positive finding (M8/M9/M10: tied recurrence + JOINT multi-output state buys whole-row
coherence on `converge`) was pinned to **rule 78 (M8c added 13/92), ONE model size (~14k, hidden=latent=64),
n_train=4000**. Before any §9 reframing rests on it, M11 stress-tests it across the two never-tested axes.
Model size (hidden/latent) is a per-arm scalar, not grid-able, so the size axis = **3 separate configs**
(`m11_size_small` hidden=32 ~5–6k; `m11_size_base` hidden=64 ~14–17k; `m11_size_large` hidden=128 ~44–50k);
the operator axis rides the base config's rule grid. Arms/deltas/curriculum mirror M10 (joint
`trm_nods`/`trm_stepDS`, decoupled `trm_decoupled_nods`/`_stepDS`, `ff_matched`, `untied_matched`),
10 seeds, 100 epochs, the M10 §4a/§4b grounding. Tracked:
`results/m11_size_{small,base,large}_2026062*_{curve,deltas,params}.csv` (+ JSON).

**New operator families screened first (read-only).** Candidate converging ECAs {4,12,36,44,76,104,128,
132,140,200,232} screened over `make_converge` for: reaches a true fixed point, balanced-ish baseline,
non-trivial convergence-depth spread. Most reject (collapse in 1–2 steps; near-degenerate maj 0.93–1.0;
160 doesn't converge). Picked **232** (majority — perfectly balanced maj≈0.50, shallow ~depth≤10) and
**140** (deep spread max≈17, but unbalanced maj≈0.75) as two genuinely-distinct families; added to the
`test_converge_target_is_a_fixed_point` parametrize. **Gotcha (cost one failed run): w=16 is UNUSABLE for
the full rule set** — rules 13 & 232 have **limit-cycle initial states on a w=16 ring** (never reach a
fixed point; the generator correctly raises), which the original single-seed n=5000 screen missed (the
n=4000 run draw hit a cycling row). M8b/M8c only ever ran rule 13 at w≥24 for this reason. Verified
**w∈{24,32} clean for all 5 rules** (0 unconverged over 480k draws each, worst depth ≤22 ≪ the 4·w cap),
so the width grid is **{24, 32}** — better than the planned {16,24} anyway: w=24 inside M9's loop-beats-both
regime, w=32 brackets its boundary. **Anchor verified bit-for-bit (2-D arms):** base rule-78/w-24 reproduces
M9/M10 exactly for the 2-D arms — `trm_nods` EM 0.444, `ff` 0.311, `untied` 0.126, `trm_stepDS` 0.427;
Δ(nods−ff) +0.133, Δ(nods−untied) +0.318. **The `trm_decoupled` arms do NOT reproduce bit-for-bit** (M11 EM
0.0722 / 0.0910 vs M10's 0.0576 / 0.1101) — see caveat (vi) below; the *effect* is unaffected.

**Headline EM deltas (sign-test p; *=p<.05, 10 seeds). nods/decoup/ff/untied = equal (final-loss) supervision:**

| size | rule, w | Δ(nods−ff) | Δ(nods−untied) [P1] | Δ(nods−decoup) [mech] | Δ(decoup−ff) |
|---|---|---|---|---|---|
| small | 78, 24 | **−0.055*** (ff wins) | +0.064* | +0.066* | −0.121* |
| small | 92, 24 | **−0.065*** (ff wins) | +0.074* | +0.066* | −0.130* |
| base | 78, 24 | +0.133* | +0.318* | +0.372* | −0.239* |
| base | 92, 24 | +0.108* | +0.324* | +0.367* | −0.259* |
| large | 78, 24 | **+0.251*** | +0.220* | +0.549* | −0.298* |
| large | 92, 24 | **+0.232*** | +0.195* | +0.536* | −0.304* |
| large | 78, 32 | **+0.118*** | +0.236* | +0.311* | −0.193* |
| base | **140**, 24 | −0.005 (ns) | +0.786* † | **+0.010 (ns)** | −0.015 (ns) |
| base | **232**, 24 | **−0.448*** (ff dominates) | +0.245* | **−0.111*** (reversed) | −0.337* |
| base | **232**, 32 | **−0.516*** (ff dominates) | +0.201* | −0.078 (ns) | −0.438* |

(† Δ(nods−untied) on rule 140 is huge only because `untied` totally collapses there, EM 0.028 — `untied`
failing, not a coherence win.) The trainability-clean mechanism Δ(stepDS−decoupled_stepDS) EM tells the
same story: positive 10/0 at **all three sizes** for {13,78,92} (small +0.025…+0.09, base +0.07…+0.34,
large +0.19…+0.43), but **ns on rule 140** and **significantly NEGATIVE on rule 232** (−0.19/−0.22).

**Reading (per §2/§8).**

1. **MODEL SIZE — the joint-state mechanism and the tying-positive GENERALIZE, and "loop-beats-both"
   STRENGTHENS with size (it is NOT a tiny-model artifact — the opposite).** For the original {13,78,92}
   family: the **joint-state mechanism** (M10) is positive 10/0, p<.05 at **all three sizes** on both the
   final-loss Δ(nods−decoupled) and the trainability-clean Δ(stepDS−decoupled_stepDS) — decoupling collapses
   coherence regardless of capacity (and the gap *grows* with size: large Δ(nods−decoup) +0.53…+0.66 vs base
   +0.37). The **tying-positive P1** (Δ(nods−untied) EM > 0) is positive 10/0 at all three sizes too. The
   one **capacity-contingent** claim is **loop-beats-both (P2, Δ(nods−ff))**: **NEGATIVE at small** (ff
   *beats* the loop on EM) — **−0.04…−0.07, p<.05 at w=24 (all 3 rules)**; at w=32 the same sign but
   smaller/weaker (−0.009 ns for rule 13, −0.018/−0.020 p<.05 for rules 92/78) — positive **w≤24** at base (the M9 regime), and
   **strongly positive at BOTH widths at large** (+0.12…+0.25*). So scaling the model up does **not** erase
   the loop's edge — it amplifies it and extends it past the w=24 boundary. At small size the model simply
   lacks the capacity for the joint refinement to overcome the shallow MLP. **Overfit guard (M5 lesson):**
   at large, train−test gaps are small (train ~0.96–0.98 vs test ~0.94–0.97) — no overfit wall, so the size
   signal is a real architecture effect, not a sample-bound artifact.
2. **OPERATOR FAMILY — the whole result is SPECIFIC to the {13,78,92}-type "hard" convergence; it does NOT
   generalize to the two new families.** On **rule 232** (majority, balanced, *shallow* per-instance depth)
   the shallow `ff_matched` **dominates** — Δ(nods−ff) EM **−0.45…−0.52*** (ff EM 0.83 vs loop 0.24–0.39) —
   and the joint-state mechanism is **absent/reversed** (decoupling neutral-to-helpful: Δ(nods−decoup)
   −0.11*/ns, stepDS mechanism −0.19/−0.22*). On **rule 140** (deep but unbalanced, ff-easy) the loop merely
   **ties** ff (ns) and decoupling does **not** collapse coherence (Δ(nods−decoup) +0.010 ns; stepDS ns) —
   the mechanism is **absent**. The cause is legible: both new rules are per-cell *easy* (`ff_matched` reaches
   EM 0.82–0.83 — the shallow MLP already makes coherent rows), so there is no coherence gap for the joint
   state to fill. The loop's joint-state advantage appears **only where a shallow per-cell map fails on
   coherence** — i.e. {13,78,92}, where `ff` EM is only ~0.31. So the M8/M9/M10 result is not about
   "multi-output fixed-point targets" in general; it is about a **subclass of hard-convergence operators**.
3. **Net for the §9 reframing.** Two of the three legs are now **size-robust and demonstrated across 3 sizes**:
   (P1) tied recurrence beats a fair untied stack on EM, and (mechanism) the **JOINT** multi-output state is
   what carries it (M10 generalizes; decoupling collapses coherence at every size). The "tiny-model artifact"
   worry is **closed** — the edge strengthens with capacity. But the result is **narrower than 'fixed-point
   targets'**: it is **operator-family-specific** (needs a per-cell-hard target; the two new families are
   ff-dominated), and the headline **"loop-beats-both" is capacity-contingent** (ff wins at small; loop wins
   and widens at large). Any §9 rewrite must scope the loop's value as *"whole-row coherence via the joint
   multi-output state, on **hard** multi-output fixed-point targets, robust over a fair untied stack and
   growing with model size — NOT universal across operator families, NOT a token-acc edge, NOT adaptive
   compute, NOT depth-extrapolation."*

**Caveats / open gaps.** (i) Three sizes (32/64/128), still all "tiny"; the size trend is monotonic but
2 points × the base, not a fine sweep. (ii) New-family coverage is two rules (140 ff-easy/unbalanced, 232
shallow/balanced) — both happen to be per-cell-easy, so M11 shows the result fails on *easy* converging
operators but has **not** found a *hard* operator outside {13,78,92} to confirm the "hard-convergence"
boundary is the real axis (vs something idiosyncratic to {13,78,92}); finding a balanced+deep+ff-hard
new rule is the natural follow-up. (iii) `untied_matched` is OVER budget (1.02–1.07; small/w32 worst at
+7.1% from width-quantization) at small/base — conservative for P1 (it loses); large is budget-clean.
(iv) n_train=4000 fixed across sizes; the large model shows no overfit wall, but a much larger model would
need the M5 sample-scaling check. (v) w=16 dropped (limit cycles for rules 13/232); the width axis here is
{24,32} only — M9 already resolved the fuller width sweep at base size. (vi) **`trm_decoupled` is NOT
bit-reproducible across numerical environments** (adversarial-review finding): its 3-D batched matmul
`(B,w,m)` has thread/BLAS-order-sensitive float reductions, unlike the 2-D arms — so its M11 EM (0.0722/
0.0910 at rule78/w24) does **not** match M10's (0.0576/0.1101) even though every 2-D arm reproduces M9/M10
to 4 decimals, and it is NOT bit-identical across `num_threads` (verified: 1-thread reproduces the committed
value exactly, 4-thread gives 0.86267 vs 0.86263). The committed run is internally reproducible at
`num_threads=1`; the EM noise is ~±0.015, dwarfed 30–50× by the +0.37…+0.66 collapse effect, so no
conclusion is affected — but the "bit-identical" guarantees in §11(a) and the per-run determinism tests do
**not** extend to the decoupled arm. (A reduction-order-pinned decoupled forward would fix it at the cost of
re-baselining M10/M11; not worth it given the effect size.)

---

## M12 — DONE. Confirm the "hard-convergence" boundary. The joint-state coherence mechanism reproduces on ALL 5 untested orbit-mates: "balanced+deep convergence" is exactly two ECA symmetry orbits, the result is a property of the REGIME (ff-hardness the operative axis), NOT idiosyncratic to the 3 hand-picked {13,78,92}.

The §11(c) follow-up M11 named. M11 showed the loop's joint-state coherence edge appears **only where a
shallow per-cell MLP fails on coherence** (ff EM ~0.31 on {13,78,92}); M11's two new families (140
deep/unbalanced, 232 shallow/balanced) were **ff-EASY** (ff EM 0.82–0.83) and showed **no** mechanism. So
"ff-hardness" — not depth or balance alone — looked like the operative axis. M12 tests it: find a NEW
balanced+deep+ff-hard converging rule and confirm the mechanism reproduces.

**Screen (read-only, all 256 ECA rules).** Filter = converges cleanly at **both** w∈{24,32} (no
limit-cycle rows over 6 seeds × 4000) **and** balanced (maj∈[0.48,0.62]) **and** deep (max-depth ≥12,
frac>4-steps ≥0.10). Returns **EXACTLY 8 rules**, all with a near-identical profile (maj≈0.563, max-depth
~18, frac>4 ~0.30–0.37), forming **exactly TWO symmetry orbits** (reflection + colour-complement):
**orbit 0 = {13, 69, 79, 93}**, **orbit 1 = {78, 92, 141, 197}**. {13,78,92} already sampled both (13∈orbit0;
78,92∈orbit1). So **"balanced+deep convergence" is the complete closure of two ECA symmetry classes — there
is NO such operator outside it.** The 5 untested rules {69,79,93,141,197} are the mirror/complement
orbit-mates. Config `m12_hardconv_orbit.yaml` = M11 base (hidden=latent=64) with this rule grid; M10 arm set,
10 seeds, 100 epochs, w∈{24,32}. 112 tests (the 5 rules added to the converge fixed-point parametrize), ruff
clean. Tracked: `results/m12_hardconv_orbit_20260623T151943_{curve,deltas,params}.csv` (+ JSON).

**Per-arm EM / token-acc at w=24 (the decisive cell; baseline acc≈0.562) + headline EM deltas (sign-test; *=p<.05, 10 seeds):**

| rule (orbit) | nods EM | ff EM | decoup EM | untied EM | Δ(nods−ff) | Δ(nods−untied) | Δ(nods−decoup) | Δ(stepDS−dec_sDS) | Δ(decoup−ff) |
|---|---|---|---|---|---|---|---|---|---|
| 69 (0)  | 0.496 | 0.304 | 0.059 | 0.097 | **+0.192*** | +0.399* | +0.436* | +0.278* | −0.245* |
| 79 (0)  | 0.513 | 0.342 | 0.073 | 0.098 | **+0.170*** | +0.414* | +0.439* | +0.337* | −0.269* |
| 93 (0)  | 0.517 | 0.328 | 0.072 | 0.102 | **+0.190*** | +0.416* | +0.445* | +0.280* | −0.256* |
| 141 (1) | 0.443 | 0.299 | 0.090 | 0.124 | **+0.144*** | +0.319* | +0.353* | +0.344* | −0.209* |
| 197 (1) | 0.467 | 0.310 | 0.071 | 0.120 | **+0.157*** | +0.347* | +0.396* | +0.349* | −0.239* |

(w=32, as in M9/M11: the clean loop-beats-both fades — Δ(nods−ff) EM ns/≈0 — while the tying-positive and
joint-state mechanism persist: Δ(nods−untied) +0.06…+0.08, Δ(nods−decoup) +0.08…+0.09, Δ(stepDS−dec_sDS)
+0.06…+0.13, all 10/0; decoup−ff negative 0/10. Same boundary as M11.)

**Reading (per §2/§8 — the prediction is confirmed cleanly on every rule).**

1. **ff-HARD confirmed.** ff EM is **0.30–0.34 at w=24** for all 5 rules — squarely the {13,78,92} range
   (~0.31) and far below M11's ff-easy 140/232 (0.82–0.83). The orbit-mates have a per-cell-hard s0→s_inf
   map, as predicted from their balanced+deep profile.
2. **loop-beats-both reproduces (w≤24).** Δ(nods−ff) EM **+0.144…+0.192 (10/0, p<.05)** AND Δ(nods−untied)
   **+0.32…+0.42 (10/0)** in **all 5 rules** at w=24 — the loop beats *both* param-matched controls on
   whole-row EM, the M9/M11 base regime, now on rules never trained on. (token-acc stays matched, Δ(nods−ff)
   acc ~0; the edge is coherence, not per-cell accuracy — the M9 mechanism statistic.)
3. **The joint-state mechanism (M10) reproduces.** Δ(nods−decoupled) EM **+0.35…+0.45** and the
   trainability-clean Δ(stepDS−decoupled_stepDS) EM **+0.28…+0.35** (both 10/0) in all 5; the decoupled arm
   falls **below** the shallow §4a MLP everywhere (Δ(decoup−ff) 0/10). Severing the joint multi-output state
   collapses the coherence — exactly M10 — on both orbits.
4. **Both orbits confirmed, including the previously under-sampled orbit 0.** Before M12, orbit 0 had only
   rule 13; now 69/79/93 reproduce it. Orbit 1 (78/92 before) reproduces on 141/197.

**Net.** The project's one positive result is a property of the **hard-convergence regime**, not 3 lucky
rule numbers: it holds on the **full untested membership of both ECA symmetry orbits**, with **ff-hardness**
(a per-cell-hard fixed-point map a shallow MLP can't make coherent) the operative axis — M11's deep-but-easy
140 and shallow 232 lacked it and showed nothing. Combined with M11 (size-robust, strengthens with capacity)
and M10 (joint state is the cause), the loop's earned value is now well-characterised: **tied recurrence with
a JOINT multi-output state buys whole-row coherence on hard multi-output fixed-point targets — robust over a
fair untied stack, growing with model size, and holding across the entire hard-convergence ECA regime.**

**Caveats / open gaps.** (i) **The orbit-mates are symmetry images (mirror/complement) of {13,78,92}** — to a
non-equivariant model they are genuinely different, never-trained-on datasets (a real robustness test), but
they are not a *dynamically independent* operator; the screen **proves none exists** among 3-neighbour ECAs
(balanced+deep convergence = these two orbits, full stop). Exhibiting a truly independent hard-convergence
operator requires **leaving the ECA family** (larger neighbourhoods, multi-state, or a non-CA fixed-point
substrate) — the genuine open frontier, and the natural next probe if more generality is wanted. (ii) Base
size only (M11 already established the size-amplification). (iii) `untied_matched` over budget (1.025/1.031,
width-quantization) — conservative, it loses. (iv) The `trm_decoupled` cross-environment determinism caveat
(M11 caveat vi) carries: its EM carries ~±0.015 reduction-order noise, dwarfed by the +0.35…+0.45 effect.

---

## Infra — Training/eval performance (no scientific change). Bit-identical, ~2.5× faster.

Not a milestone — a perf pass on the model/training/eval path. **All run outputs are byte-for-byte
unchanged** (verified: parity single-output and iterated multi-output cells reproduce prior
accuracies and exact-match exactly; 67/67 tests pass; ruff clean).

Four bottlenecks resolved:

1. **Data path dominated wall-clock.** For the tiny models here the per-sample
   `Dataset.__getitem__` + default-collate path of `torch.utils.data.DataLoader` cost more than
   the matmuls. Replaced with `InMemoryLoader` (`src/looptab/data/dataset.py`): the RAM-resident
   dataset is stacked into tensors once and batched by slicing a permutation. Determinism is
   preserved **bit-for-bit** by reproducing `DataLoader`'s exact per-epoch global-RNG protocol —
   the `_BaseDataLoaderIter` worker `_base_seed` draw *and then* `RandomSampler`'s seed draw → fresh
   `Generator` → `randperm` — so both the consumed RNG state and the batch composition match the
   loader it replaces (checked against a real `DataLoader` over multiple epochs).
2. **Redundant eval forward pass.** On multi-output (Task B) cells, `accuracy` and `exact_match`
   each ran their own forward over the test set (and once per R' in the extrapolation harness).
   Added `evaluate` (`src/looptab/eval/metrics.py`) which derives both from a single `_predict`;
   `run_point` and `run_extrapolation_point` now use it. Same predictions, half the eval passes.
3. **CPU thread oversubscription.** The tiny models' matmuls fall below torch's parallelization
   threshold, so torch's default intra-op thread count (= core count) adds only dispatch overhead.
   Measured (4-core box): threads 1≈2 < 4 < **8 ≈ 3× slower than 1**. On many-core cloud machines
   the default is far worse (torch defaults to the full core count). Added `TrainConfig.num_threads`
   (default **1**), applied once in `run.main()` via `torch.set_num_threads`. Verified bit-identical
   across thread counts (full-precision, both single- and multi-output) — the small kernels don't
   reorder reductions — so this is a pure speed/portability win. `num_threads: null` restores torch's
   default for when models grow.

Measured: a representative `run_point` (2 arms × 30 epochs, n_train=4000) went 7.19s → 2.85s (~2.5×)
on CPU from (1)+(2); thread pinning takes the warm loop a further ~2.83s → 2.43s here and avoids the
~3×+ oversubscription penalty on big-core boxes. Multi-output runs gain additionally from the
single-pass eval. No config, metric, or conclusion changes — this only makes re-running cheaper.

4. **Serial seed loop left cores idle.** With per-run work pinned to 1 thread (item 3), a
   multi-core CPU sat mostly idle. The per-axis-point seed loop now runs across a process pool
   (`ExperimentConfig.parallel_workers`, default **1** = unchanged serial; `run._compute_seeds`),
   each worker pinned to `train.num_threads` so workers × threads never oversubscribe. Seeds are
   pure functions of their seed and self-reseed, so it is **bit-identical** to serial (verified:
   `parallel_workers=4` reproduces serial accuracies exactly; guarded by
   `test_parallel_seeds_bit_identical_to_serial`). Measured **4.12× on a 4-core box** for a
   4-seed run; scales with cores/seeds. Also switched eval to `torch.inference_mode` (a
   strictly-faster `no_grad`; numerically identical).

Measured: a representative `run_point` (2 arms × 30 epochs, n_train=4000) went 7.19s → 2.85s (~2.5×)
on CPU from (1)+(2); thread pinning (3) takes the warm loop a further ~2.83s → 2.43s and avoids the
~3×+ oversubscription penalty on big-core boxes; seed-parallelism (4) adds ~Ncores× on multi-seed
runs (4.12× measured on 4 cores). Multi-output runs gain additionally from the single-pass eval. No
config, metric, or conclusion changes — this only makes re-running cheaper. **Set `parallel_workers`
to the core count on any ≥5-seed sweep/grid to use the cores; it stays off (1) by default.**

**Model-level changes investigated and REJECTED (negative result, §8).** A pass looking for
faster *model math* found nothing worth landing — the TRM core is tiny and already minimal, so its
cost is the irreducible matmul forward/backward, not removable Python overhead. Measured on
representative configs (d∈{20,40,80}, steps 4–8, threads=1):
  - *Precompute the constant `X` projection out of the weight-tied loop* (mathematically the same
    reassociation of the first linear): **1.01–1.05×**, and **not** bit-identical (maxdiff ~1e-7
    from FP reassociation → would force re-baselining every committed result). Reject.
  - *Batch deep supervision into one `cross_entropy` over stacked per-step logits*: **0.98–0.99×
    (slightly slower** — the `stack`+`expand` cost cancels the fewer-call saving), and not
    bit-identical. Reject.
  - *Functional forward* (`F.linear`/`F.gelu` instead of `Module.__call__`, skipping hook checks):
    **bit-identical (maxdiff 0.0)** but only **1.01–1.04×** — not worth the readability cost of
    reaching into `update_net` internals on the canonical model. Reject.
So the model is left as-is; the wins all live at the harness level (1)–(4). Don't re-litigate these
without first changing the regime (much larger models, or accepting a numerics re-baseline).

---

## M13 — DONE. Leave the ECA family (threshold/Hopfield attractor net). The joint-state coherence result is CA/local-update-specific; only the tying-positive P1 generalizes. Clean NEGATIVE.

M12 closed within-ECA generality: the balanced+deep-converging ECAs are *exactly* two symmetry
orbits, so every `converge` test is dynamically a mirror/complement of {13,78,92}. The one open
scientific question (§11(c) thread 2): **is the joint-state whole-row-coherence result (M8–M12) a
property of the hard-convergence REGIME, or of cellular automata specifically?** Answering it
requires a *dynamically independent* hard-convergence target — i.e. leaving the ECA family.

M13 builds `make_hopfield` (`src/looptab/data/generators.py`, dispatched in `dataset.py`,
exported in `data/__init__.py`, task literal added in `config.py`, determinism-tested in
`tests/test_generators.py`): a **dense, fully-coupled binary threshold / Hopfield attractor net**
— maximally unlike a local 3-neighbour CA, and basin-of-attraction is *intrinsically* a whole-row
property, the strongest possible probe of the joint-state hypothesis. The function (fixed by
`task_seed`) is an **all-integer** symmetric zero-diagonal weight matrix `W` (Hebbian
`Σ_μ ξ^μ ξ^μᵀ` over `n_patterns` random ±1 patterns, or a random integer mode) plus integer
self-coupling `γ`; rows (fixed by `sample_seed`) are `s0 ∈ {-1,+1}^w` iterated synchronously
`s_{t+1}=sign(W·s + γ·s)` (tie→keep) to the global fixed point. **Synchronous convergence is
guaranteed by construction:** `γ ≥ -λ_min(W)` makes `W+γI` PSD ⇒ the parallel energy is
non-increasing ⇒ a fixed point, no 2-cycles (committed runs pin an explicit integer `γ`, so the
generator is purely integer ⇒ **bit-exact**, no float-matmul determinism risk; the loud guard +
a multi-seed screen enforce it). Outputs map to {0,1} for the binary heads / `coherence_excess`.
The contract mirrors `make_converge` exactly, so the M10 arm set, curriculum, step-aligned DS, and
trajectory machinery run unchanged. 121 tests, ruff clean.

**Screen (`m13_hopfield_screen.yaml`) — the regime is balanced + ff-HARD.** Multi-seed over
the real task_seeds 42..51 (M12 lesson): at the locked `weights=hebbian, n_patterns=12, γ=16,
distractors=8` setting, **0/10 non-convergence raises** at w∈{24,32}, balanced (majority ~0.50),
and **ff-HARD** — a shallow `ff_matched` lands at **EM ~0.26 @ w=24 / ~0.14 @ w=32** (token-acc
~0.93), numerically the same hard regime as the hard-convergence ECAs (ff EM ~0.31, M11). So the
substrate clears the precondition the result needs: a genuine multi-output fixed point on which the
per-cell MLP fails to make whole rows. **Convergence depth (a precondition to state HONESTLY — an
adversarial review caught the first draft overstating it):** per-row depth is **typical median ~2–3**
(mean 2.4/3.0, p90 4/5, batch-max ~10 ≪ the 8·w cap); >87% of rows settle in ≤4 steps, so the loop's
`n_steps=6` is **ample, not starving** — and this is **comparable to rule 78's median ~3 where the loop
WON**, so depth is roughly controlled across the CA/non-CA comparison and is NOT the distinguishing
axis. (The earlier "deep ~9–10" was the batch-maximum settling time, not a typical difficulty.)

**The experiment (`m13_hopfield_converge.yaml` base hidden=64, `m13_hopfield_large.yaml` hidden=128;
the M10 six-arm set, 10 seeds, w∈{24,32}).** Per-arm EM, base, w=24:
**`ff_matched` 0.256 > `trm_nods` 0.193 > `trm_decoupled_nods` 0.148 > `untied_matched` 0.113**
(`trm_stepDS` 0.224). **The shallow MLP is the BEST arm on whole-row coherence — the exact inverse
of the hard-convergence ECAs, where ff was worst and the loop topped it.**

**The four load-bearing EM deltas across size × width (paired, 10 seeds; sign-test p):**

| Δ (exact-match) | base w24 | base w32 | large w24 | large w32 |
|---|---|---|---|---|
| nods − decoupled_nods  (joint mech, final loss) | +0.044 (9/1, p=.021) | +0.033 (8/1, p=.039) | +0.064 (9/1, p=.021) | +0.037 (9/1, p=.021) |
| **stepDS − decoupled_stepDS  (trainability-clean mech)** | +0.025 (8/2, **p=.109 ns**) | +0.030 (8/2, **p=.109 ns**) | +0.012 (5/5, **p=1.0 ns**) | +0.020 (6/4, **p=.75 ns**) |
| **nods − ff_matched  (loop-beats-both)** | **−0.063 (0/10, p=.002)** | −0.020 (3/7, p=.34 ns) | −0.053 (2/8, p=.11 ns) | **−0.029 (1/9, p=.021)** |
| nods − untied_matched  (tying-positive P1) | +0.080 (10/0, p=.002) | +0.050 (10/0, p=.002) | +0.047 (10/0, p=.002) | +0.004 (6/4, p=.75 ns) |

**Reading (per §8 — the honesty clause fires; the result is BOUNDED to the CA/local-update regime).**

- **Loop-beats-both does NOT transfer.** The loop never beats `ff_matched` on coherence; it is
  *significantly worse* at base/w24 (−0.063, 0/10, p=.002) and large/w32 (−0.029, p=.021), and
  ns-negative elsewhere. It also loses on token-acc (base/w24 Δacc −0.018, p=.002). This is the
  **opposite** of the hard-convergence ECAs, where the loop beat ff at base/w24 (M9, ΔEM +0.133).
- **The JOINT-STATE MECHANISM (M10's core) essentially does NOT transfer.** The trainability-clean
  Δ(stepDS − decoupled_stepDS) — the comparison M10 said to lean on (the decoupled arm trains
  stably under step-aligned DS) — is **non-significant in ALL FOUR size×width cells** (p = .11,
  .11, 1.0, .75). The final-loss Δ(nods − decoupled) is weakly positive (EM +0.03…+0.06, p≈.02–.04)
  but small, on the fragile arm, and — decisively — **does NOT grow with model size** (base ≈
  large), whereas on the ECAs it grew +0.37→+0.66 from base to large (M11). So on a dense
  (non-local) target, severing the joint multi-output state barely dents coherence: whatever
  coherence exists is not coming from the joint state.
- **CAPACITY DOES NOT REVIVE IT — the decisive M11 contrast.** On the ECAs, scaling 64→128
  *strengthened* the whole result (mechanism and loop-beats-both both grew). Here, scaling does
  nothing: loop-beats-both stays negative, the clean mechanism stays null. So the failure to
  transfer is **intrinsic to the substrate, not a tiny-model artifact** — the obvious "you only
  tested base size" objection is closed.
- **The tying-positive P1 is the one survivor (and it too weakens).** Δ(nods − untied_matched) is
  strongly positive at base both widths and at large/w24 (10/0, p=.002) — the loop beats the fair
  untied stack on coherence in 3/4 cells — but it **vanishes at large/w32** (+0.004, ns). P1 is the
  project's durable architectural fact and it broadly generalizes off-CA, though it is no longer
  uniform. **Budget-parity status of the P1 control (stated explicitly — the committed
  `*_params.csv` flags it, so do not consume P1 as uniformly "budget-clean"):** at BASE,
  `untied_matched` is the integer-width-quantization breach the audit names — **+2.46% (w24) /
  +3.08% (w32) OVER the declared ±2% budget** (`within_tol=False`), the same M3a/M4 width-quantization
  effect. The breach is **one-directional (over-budget ⇒ the control has MORE capacity)**, so the
  base P1 cells are **conservative, not clean**: the loop beats an untied stack that is handed a
  small capacity *advantage*. The strictly-clean P1 evidence is the **LARGE run, where
  `untied_matched` is WITHIN tol and in fact slightly UNDER budget (ratio 0.988 w24 / 0.998 w32):
  at large/w24 P1 = +0.047 (10/0, p=.002) on a budget-clean, under-budget control.** So P1 survives
  both a conservative over-budget control (base) AND a strictly-matched one (large/w24); it is not an
  artifact of the breach. Note `trm_decoupled_nods` *also* beats `untied_matched` on EM at base
  (decoupled 0.148 > untied 0.113), so even the per-cell loop out-coheres the untied stack — the
  untied stack, not the decoupled head, is the coherence-floor here.

**Net — a clean, well-controlled NEGATIVE that bounds M8–M12.** Despite being a genuine multi-output
fixed point with globally-coupled (whole-row) basin structure AND ff-hard (ff EM ~0.26/0.14), the
threshold net does **not** reproduce the loop's coherence story: loop-beats-both fails (ff is the
*best* coherence arm), and the joint-state mechanism is absent under the trainability control — at
base AND large size. **So the M8–M12 result is CA / LOCAL-UPDATE specific, not a property of
hard-convergence multi-output fixed points in general.** The loop's coherence edge on ECAs came from
something specific to *local, spatially-structured* CA dynamics — where a shallow per-cell map
makes spatially-correlated errors that the joint cross-cell state repairs — not from "multi-output
fixed point with global dependencies" per se, and **NOT from depth** (per-row depth here is median
~2–3, comparable to rule 78 where the loop won). On the dense threshold net the shallow MLP (full
row in its receptive field) already realizes the achievable whole-row coherence (equal
`coherence_excess` to the loop, ~0.08), leaving no gap for the joint loop to fill — indeed the loop
is a strictly *worse* per-cell model here (lower token-acc AND train-acc than ff). The lone
regime-independent survivor is the **tying-positive P1** (tied loop > fair untied stack), broad but
no longer uniform.

**Robustness — the negative is NOT depth/compute starvation (adversarial-review probe).** The
obvious objection is that the loop runs `n_steps=6` while the slowest rows take ~10 steps. Re-running
base/w24 at `n_steps=12` with a `T_max=12` curriculum (4 seeds) leaves the loop still losing to ff —
Δacc −0.041, ΔEM −0.124, no better than (slightly worse than) the n_steps=6 result. Doubling the
loop's compute does not close the gap: the loop is genuinely outclassed, not starved.

**Hypothesis (NOT tested here, §8):** the loop's coherence gain requires a target whose per-cell map
is *local* with *spatial* error structure (the CA case), so that a shallow MLP's errors are
correlated in a way the joint refinement state can correct; a target where the shallow MLP already
sees the whole row and captures the cell correlations (the dense-net case) leaves no gap. Testing
this would need an intermediate substrate (e.g. a local-but-non-CA fixed-point map), out of scope here.

**Consequence for §9.** M13 *sharpens* the scope for the pending §9-gate rewrite rather than widening
it: the loop's defensible value is **whole-row coherence via the joint state on LOCAL-UPDATE (CA)
hard-convergence targets** (M8–M12), plus the broader-but-not-uniform **tying-positive P1** (beats a
fair untied stack), now tested off-CA. It is NOT a property of hard-convergence fixed points in
general. Tracked summaries: `results/m13_hopfield_screen_*.{json,csv}`,
`results/m13_hopfield_converge_*.{json,csv}` (base), `results/m13_hopfield_large_*.{json,csv}` (large).

---

## M14 — DONE. The locality probe (local-but-non-CA threshold net). M13's locality hypothesis FALSIFIED: locality makes the task ff-EASY (helps the control, not the loop). Tying-positive P1 survives, now budget-clean across a full local→dense ladder.

M13 left one open scientific hypothesis (§8, and §11(c)): the loop's joint-state coherence edge
might require a *local, spatially-structured* per-cell map (the CA case), not just any
hard-convergence fixed point — M13's *dense* Hopfield net left the shallow MLP no coherence gap
because it already sees the whole row. M14 tests this directly with **one knob** on the M13
substrate: `bandwidth` b on the threshold-net weight matrix W. On a ring of w cells the band mask
zeros every coupling beyond ring distance b — **b small = spatially LOCAL but per-position-irregular
(NON-CA), b = w//2 = dense (= M13)**. This isolates *locality* (the knob) from the
*translation-invariance/uniformity* a true CA also has (absent at every b), the two properties that
distinguish the ECA from M13's dense net. New code: `_ring_band_mask` + a `bandwidth` param on
`make_hopfield`/`_build_hopfield_weights` (all-integer ⇒ still bit-exact; the PSD-guaranteeing γ is
derived from the *masked* W so convergence holds). 131 tests (4 new bandwidth tests), ruff clean.

**The convergence-vs-triviality screen (numpy + a 3-seed training screen) — this dictated the
regime, do not skip it.** A locality probe has an intrinsic confound: reducing bandwidth raises
**triviality** (fraction of inputs already at a fixed point ⇒ identity map ⇒ ff-EASY), because
guaranteed convergence needs a large self-coupling γ that dominates sparse local couplings. The
numpy pre-screen mapped (bandwidth × γ × w) for convergence (loud-guard pass over 10 task_seeds,
both sample seeds, n=5000), balance, and triviality:
- **w=24/32 have a tight margin** — at the clearly-local end (b≤2) it is hard to get both 10/10
  convergence AND low triviality (b=1 ~84% identity or non-convergent; b=2 ~15% trivial), so w=48 was
  chosen for cleaner margins (not because a local regime is strictly impossible at w≤32).
- **w=48 does:** a single **γ=10** gives bandwidth {2,4,8} all 10/10-convergent, balanced (~0.50),
  non-trivial (triv ≤5%), depth median 1–2 (≥99% settle in ≤6 steps ⇒ `n_steps=6` ample). The
  **dense** end (b=24) needs **γ=16** (γ=10 leaves it non-convergent on 3/10 seeds) — so a single γ
  cannot span local+dense, hence the local ladder (`m14_local_ladder`, γ=10, grid b∈{2,4,8}) and a
  **same-w dense anchor** (`m14_dense_anchor`, γ=16, b=24) are two configs. Regime locked at
  w=48, n_patterns=12, distractors=8. The 3-seed training screen (`m14_local_screen`) then flagged
  the headline before the full run: ff_matched **nearly solves b=2** (acc 0.999).

**Full result — w=48, 10 seeds, M10 arm set (paired Δ, sign-test p; baseline ~0.50):**

| b (γ) | ff acc | loop acc | Δ(loop−ff) acc | Δ(loop−ff) EM | Δ(loop−untied) acc | Δ(loop−untied) EM |
|-------|--------|----------|----------------|---------------|--------------------|-------------------|
| 2 (local, γ10)  | 0.999 | 0.956 | **−0.044** (0/10, p=.002) | **−0.800** (0/10, p=.002) | **+0.221** (10/0, p=.002) | **+0.162** (10/0, p=.002) |
| 4 (γ10)         | 0.963 | 0.920 | **−0.044** (0/10, p=.002) | **−0.210** (0/10, p=.002) | **+0.149** (10/0, p=.002) | **+0.038** (10/0, p=.002) |
| 8 (γ10)         | 0.921 | 0.900 | **−0.025** (0/10, p=.002) | **−0.020** (0/10, p=.002) | **+0.075** (10/0, p=.002) | **+0.017** (10/0, p=.002) |
| 24 (dense, γ16) | 0.906 | 0.892 | **−0.015** (0/10, p=.002) | −0.010 (3/7, p=.34 ns)    | **+0.028** (10/0, p=.002) | **+0.018** (9/1, p=.021) |

Joint-state mechanism (the M10 ablation), EM: Δ(loop−decoupled) final-loss +0.143/+0.033/+0.012/+0.019
(b=2/4/8/24); the *trainability-clean* Δ(stepDS−decoupled_stepDS) EM is **ns at the local end**
(b=2: +0.091, 8/2, p=.11) and **ns at dense** (b=24: +0.014, 8/1, p=.039≈borderline), significant but
tiny at b=4/8 (+0.030/+0.010). Budget audit: within ±2% for every matched arm in every cell
(`untied_matched` ratio 0.983 — i.e. ~1.7% *under* budget, so P1 is **conservative** (the loop beats a
marginally *smaller* control), within tol but not exactly matched; cf. M13's base which was *over*
budget). The local ladder pins γ=10, which converges 10/10 by the loud guard but is **not** PSD-
guaranteed for every seed (auto-γ reached ~12 on some functions) — empirical, not by-construction; the
dense anchor's γ=16 is PSD-clean. A single γ cannot span local+dense, hence the split configs and the
γ boundary between the b∈{2,4,8} sub-ladder and the b=24 anchor.

**Reading (per §8 — the locality hypothesis FAILS to revive the edge; mechanistic attribution is a hypothesis, not isolated).**
- **Locality does NOT revive the loop's edge — if anything the reverse.** Δ(loop−ff) is **negative at
  every bandwidth** on accuracy (all 0/10, p=.002); `ff_matched` is the **best arm across the entire
  ladder**. The direct test of M13's hypothesis (does locality revive the *joint-state coherence
  mechanism*?) is the trainability-clean Δ(stepDS−decoupled_stepDS), which is **null at both the local
  and dense ends** (ns) and tiny in between — nothing like the ECA's +0.32…+0.66. So a *local, non-CA*
  per-cell map does not bring the mechanism back: clean null on the hypothesis as posed.
- **De-emphasise the b=2 ΔEM −0.80 — it is the LEAST informative cell.** b=2 is an *easy* task (ff acc
  0.999, near-solved) where **both** recurrent arms collapse vs ff (loop ΔEM −0.80, decoupled −0.94) —
  i.e. "recurrence is pointless on an easy task," not a joint-state result. The load-bearing cells are
  the *hard* end (b=8, dense: ff acc 0.92/0.91, EM low for all arms) where the loop **still** fails to
  beat ff and the mechanism delta is **null**. The headline is "no revival even where the task is hard,"
  not the dramatic easy-task number.
- **Why ff tracks bandwidth: per-cell FAN-IN, and the bandwidth↔depth CONFOUND (the key caveat).** ff
  acc runs 0.999 (b=2) → 0.906 (dense). A shallow MLP is easy⇔each output cell has small fan-in, and
  fan-in ≈ light-cone width ≈ **bandwidth × convergence-depth**. The banded net changes **two** things
  vs the ECA at once: it drops the *uniform/translation-invariant* rule AND it collapses convergence
  depth (median 1–2 here vs the ECA's 2–3 with a tail to ~22). So M14 **cannot cleanly separate**
  "uniform local rule" from "deep convergence / wide light-cone" as the loop-edge ingredient — both are
  removed together. The ECA achieves large fan-in via *depth* (radius-1 rule iterated deep); the dense
  Hopfield via *bandwidth* at shallow depth; both are ff-hard, yet the loop won only on the ECA (M13).
  **Hypothesis (NOT isolated by M14):** the loop's edge needs the *iterated translation-invariant local
  rule* specifically — but "it's depth-composition, not the uniform rule" remains an equally-supported
  alternative this experiment does not rule out.
- **The tying-positive P1 SURVIVES across a full local→dense ladder.** Δ(loop−untied) is positive and
  significant at **all four bandwidths** (acc 10/0, p=.002 throughout; EM 10/0 at b=2/4/8, 9/1 p=.021 at
  dense), strongest local (+0.22 acc) decaying to dense (+0.028). Budget is within ±2% (untied ~1.7%
  under, so conservative) — broader than M13's base P1 but with the under-budget/empirical-γ caveats
  above. P1 remains the **one regime-independent leg**, now demonstrated off-CA at both ends.

**Net — M14 CLOSES the locality thread (the last open experimental question in §11(c)).** A *local*
non-CA threshold net fails to revive the loop's coherence mechanism (trainability-clean Δ null at the
hard end; loop never beats ff), just as M13's *dense* net did — so the M8–M12 result is **not explained
by coupling locality**. The remaining mechanistic question — *uniform local rule* vs *deep convergence*
— is **not separated by this experiment** (the banded net removes both); the writeup states the
uniform-rule reading as a hypothesis, not a result. The only survivor off-CA is the **tying-positive
P1** (with the under-budget / empirical-γ caveats). After M14 the sole remaining §11(c) item is the
**§9-gate rewrite** (a writing task; the experiments are done). Tracked summaries:
`results/m14_local_screen_*.{json,csv}`, `results/m14_local_ladder_*.{json,csv}`,
`results/m14_dense_anchor_*.{json,csv}`.

---

## M15 — DONE. Separate the M14 confound (uniform rule vs deep convergence). RESULT: a clean DECOMPOSITION — the joint-state mechanism needs DEEP+LOCAL (transfers to non-uniform), but loop-beats-the-MLP needs the UNIFORM rule.

M14 closed the locality thread but left one confound flagged in review: the banded net dropped the
*translation-invariant rule* AND collapsed *convergence depth* at once, so it could not say which is
the CA-specific ingredient. M15 breaks the confound with the decisive missing cell: a **deep +
non-uniform + local** fixed-point map, contrasted against a **deep + uniform** anchor at identical
protocol.

**New substrate `mixed_converge` (generator `make_mixed_converge`, `mixed_ca_step`).** A per-position
MIXED CA: each cell runs its *own* radius-1 rule, drawn (by `task_seed`) from orbit1 {78,92,141,197}
(the converging-orbit-mates of rule 78, M12), iterated to a fixed point. Local + temporally-uniform
(same per-position step repeated — what the loop's weight-tying matches) but **spatially non-uniform**
(not a CA). A spatial mix of converging rules is NOT globally convergent (numpy screen: ~15-85% of
random inputs cycle; orbit1 mixes best, orbit0 worst), so rows are **rejection-filtered to the
convergent basin** — inputs drawn, iterated, only those reaching a genuine fixed point kept (target is
then a true fixed point; basin-conditioned input distribution, disclosed, identical across arms). All-
integer ⇒ bit-exact. Determinism/fixed-point/non-uniformity/balance/trajectory/loud-guard tests added
(140 tests, ruff clean). Pre-screen (committed): fills n=4000 with no raises over task_seeds 42..51 at
w∈{24,32}, balanced (maj ~0.50), non-trivial (>90% rows move off s0), deep (depth median 3, max 6-10),
≥97% settle in ≤6 steps (n_steps=6 ample). ff-hardness screen (3 seeds): **ff EM ~0.28 at w=24 — the
hard regime of the converging ECAs**, the precondition met.

**The decisive contrast — SAME M10 arm set / w / curriculum / seeds / distractors; the ONLY difference
is translation-invariance** (uniform `converge` rule 78 anchor `m15_uniform_anchor` vs the per-position
mixed `m15_mixed_converge`). Both deep, both ff-hard, both rule-family orbit1. Paired EM Δ, sign-test p,
10 seeds:

| leg (EM) | UNIFORM rule 78 — w=24 / w=32 | MIXED orbit1 — w=24 / w=32 |
|---|---|---|
| **loop beats ff** Δ(nods−ff)        | **+0.133** (10/0, p=.002) / −0.014 (5/5, ns) | **−0.028 (3/6, p=.51 ns)** / +0.006 (7/3, ns) |
| **joint-state** Δ(nods−decoupled)   | +0.387 (10/0, p=.002) / +0.088 (10/0, p=.002) | +0.209 (10/0, p=.002) / +0.046 (10/0, p=.002) |
| **joint-state, trainability-clean** Δ(stepDS−dec_stepDS) | +0.317 (10/0, p=.002) / +0.138 (10/0, p=.002) | **+0.206 (10/0, p=.002)** / +0.053 (10/0, p=.002) |
| **P1** Δ(nods−untied)               | +0.318 (10/0, p=.002) / +0.069 (10/0, p=.002) | +0.199 (10/0, p=.002) / +0.045 (10/0, p=.002) |
| decoupled vs ff Δ(decoupled−ff)     | −0.254 (0/10, p=.002) / −0.101 (0/10) | −0.237 (0/10, p=.002) / −0.040 (0/10) |

Budget: `ff_matched` within tol (ratio 0.998), so the **loop-vs-ff contrast is budget-clean**; only
`untied_matched` breaches (1.025/1.031 OVER budget ⇒ P1 is conservative, the loop beats a *bigger*
control). The loop-beats-ff is an **EM** effect (uniform w=24: accΔ +0.003 ns, EM +0.133 — the M9
coherence signature reproduced; mixed w=24: accΔ −0.009 ns/2-8 p=.11, EM −0.028 — point estimate
FAVOURS ff, ns).

**TASK-MATCHING AUDIT (adversarial review — the two tasks are NOT cleanly single-variable).** The first
draft claimed "the only difference is translation-invariance." That is **false** and is struck. Measured
from the committed runs, the mixed task also differs from uniform-78 on:
- **Hardness:** ff EM mixed 0.255 / 0.042 vs uniform 0.311 / 0.121 (w=24 / w=32) — the mixed task is
  per-row *harder* (and at w=32 ff EM 0.04 is near the EM floor, leaving little headroom for anyone).
- **Convergence-depth tail:** uniform-78 median 4, **max 16, ~10–13% of rows depth>6**; mixed median 3,
  **max 6, 0% depth>6** — the mixed mix converges *shallower-tailed* (the deep-converging draws are the
  ones that cycle and get rejection-filtered out).
- **Target-fixedness at T_max=6:** uniform has the intentional M8 gap (~10% of curriculum tails are
  non-fixed intermediate states); mixed has none (every tail is a true fixed point).
So `mixed` vs `uniform` is **confounded** (translation-invariance ⊗ hardness ⊗ depth-tail). This bounds
how strongly leg (2) below can be attributed to uniformity alone.

**Reading (per §8) — the M8–M12 result decomposes into two legs; leg (1) is clean, leg (2) is
suggestive-but-confounded.**
- **(1) [CLEAN — a WITHIN-task comparison, immune to the cross-task confound above] The joint-state
  coherence mechanism (joint refinement ≫ per-cell "decoupled" refinement; M10) is driven by DEEP +
  LOCAL structure, NOT translation-invariance — it TRANSFERS to the non-uniform mixed-CA.** The
  trainability-clean Δ(stepDS−decoupled_stepDS) is **+0.206 EM (10/0, p=.002) at w=24** on the mixed task
  (+0.053, 10/0, at w=32), decoupled falling **below ff** (−0.237, 0/10) — the M10/ECA signature on a
  task that is **not a CA**. Because this contrasts two arms *on the same task*, the mixed-vs-uniform
  hardness/depth mismatch does not touch it. First non-uniform substrate where the mechanism is
  significant; with M13 (dense ⇒ null) and M14 (shallow ⇒ null), its requirement is **local + deep**
  (a wide light-cone from composing a *local* update over depth), uniform or not. Attenuated vs uniform
  (+0.206 vs +0.317 at w=24) but clearly present.
- **(2) [SUGGESTIVE, CONFOUNDED — a CROSS-task comparison] Loop-beats-the-shallow-MLP (the
  loop-beats-both headline; M8/M9) does NOT reproduce on the non-uniform task — consistent with the
  uniform rule being required, but not cleanly isolated.** At w=24 the loop beats ff on EM for the
  uniform rule (**+0.133, 10/0, p=.002**) but on the mixed version the loop **does not beat ff** (EM
  −0.028, 3/6, p=.51 — the point estimate FAVOURS ff; "ties" was too generous). The naive reading is
  "uniformity is required," BUT the two tasks also differ in hardness and depth-tail (audit above), so
  this single cross-task Δ cannot attribute the loss to translation-invariance *alone*. Two things keep
  the uniform-rule reading alive as the leading hypothesis: (a) the within-task dissociation — leg (1)
  fires strongly on the *same* mixed task, so the mixed task plainly *has* enough depth/light-cone
  structure for a coherence mechanism, yet leg (2) still fails; and (b) the loop runs only `n_steps=6`,
  so the uniform task's deep tail (depth>6, ~10% of rows) is *unreachable* for the loop and cannot be the
  source of its uniform edge — the edge lives in the depth-≤6 bulk that both tasks share. **Also single-
  width:** leg (2)'s positive cell is w=24 only (uniform w=32 is −0.014 ns; both tasks tie at w=32), per
  M9's w≤24 boundary. **Mechanistic story (HYPOTHESIS, not isolated here):** a uniform rule makes the
  one-step operator maximally shared, which the weight-tied loop matches to beat a one-shot MLP; a
  per-position rule makes it spatially-varying, shrinking that edge. *The clean test leg (2) still lacks
  is a depth/hardness-MATCHED uniform control (e.g. a uniform rule sub-sampled to the mixed task's depth
  distribution and ff EM); until then leg (2) is "consistent with," not "demonstrates."*
- **(3) P1 survives on both** (conservative, untied over-budget) — the regime-independent leg, now also
  shown on a non-CA local deep task.

**Net.** Leg (1) is a solid result: the **joint-state coherence mechanism is deep+local, not
translation-invariance-specific**, and transfers off-CA to a non-uniform local deep map (within-task,
budget-aware, significant at both widths). Leg (2) — **loop-beats-the-MLP appears to need the uniform
rule** — is *suggestive* (the edge is present on uniform-78, absent on the mixed task with leg (1) firing
on the same data) but **NOT cleanly isolated**: the mixed/uniform pair is confounded by hardness and
depth-tail, and the positive cell is w=24 only. So M15 only **partially** supersedes M14's "uniform vs
depth not separated": the *mechanism* leg is separated and assigned to depth+locality; the
*loop-beats-MLP* leg is pointed at uniformity but needs a matched-difficulty uniform control to confirm.
Caveats: one rule family (orbit1; uniform anchor rule 78 ∈ orbit1), "non-uniform" = within-orbit
per-position mixing (a fully random-rule mix does not converge), rejection-filtered (basin-conditioned)
inputs, mixed task harder + shallower-tailed than the uniform anchor, leg (2) is w≤24. Tracked:
`results/m15_mixed_screen_*.{json,csv}`, `results/m15_mixed_converge_*.{json,csv}`,
`results/m15_uniform_anchor_*.{json,csv}`.

---

## M15b — DONE. Leg 2 NAILED: a depth/hardness-controlled uniform control confirms loop-beats-the-MLP needs the uniform (translation-invariant) rule.

The M15 review left leg 2 ("loop-beats-the-MLP requires the uniform rule") *suggestive but confounded*:
the mixed vs uniform-78 contrast differed in hardness and convergence-depth-tail, not just translation-
invariance. M15b removes the confound with the control the review prescribed.

**The depth-matched uniform control.** Added an `accept_max_depth` cap to `make_mixed_converge`: it keeps
only rows reaching their fixed point within the cap (additive; `None` reproduces the committed M15 output
bit-for-bit — tested). A UNIFORM single-rule CA run through the *identical* rejection-filter pipeline
(`rule_set=[r]`) with `accept_max_depth=6` then matches the mixed task's depth tail (**max depth 6, 0%
rows depth>6, target fully fixed at T=6**) — directly removing 2 of the review's 3 confounds (depth-tail
*and* target-fixedness). The 3rd (hardness) is handled by **direction**: capping makes the uniform task
*easier* (ff EM 0.362 / 0.345 for rules 78 / 13 @ w=24 vs the mixed task's 0.255), and per M11 an *easier*
task *handicaps* the loop-beats-ff edge (ff has more room) — so a loop win on the easier uniform control,
against a loop tie on the harder mixed task, is a **conservative** demonstration that the difference is
not hardness.

**Result — depth-matched uniform vs the (already-depth-≤6) mixed task, EM, 10 seeds, sign-test p:**

| task @ w=24 (depth ≤6) | ff EM | Δ(loop−ff) EM | Δ(stepDS−dec_stepDS) EM | Δ(loop−untied) EM |
|---|---|---|---|---|
| **uniform rule 78** (cap6) | 0.362 | **+0.090 (9/1, p=.021)** | +0.313 (10/0, p=.002) | +0.318 (10/0) |
| **uniform rule 13** (cap6) | 0.345 | **+0.175 (9/1, p=.021)** | +0.309 (10/0, p=.002) | +0.411 (10/0) |
| **mixed orbit1** (M15, non-uniform) | 0.255 | **−0.028 (3/6, p=.51 ns)** | +0.206 (10/0, p=.002) | +0.199 (10/0) |

At w=32 the loop-beats-ff edge vanishes for *both* uniform rules (78: −0.015, 2/8, p=.11; 13: −0.005,
2/6, p=.29) — the M9 "edge is w≤24" boundary, the same regime where the mixed task also ties. The 3-seed
screen (`m15b_uniform_matched_screen`) additionally shows all four rules {78,197,141,13} give a positive
loop−ff EM *point estimate* at w=24 (+0.065…+0.222) — directional corroboration only, **none significant
at 3 seeds** (every screen p≥0.25; §5). Budget: `ff_matched` within tol (0.998) ⇒ the loop-vs-ff contrast
is budget-clean; `untied_matched` over budget (1.025) ⇒ P1 conservative.

**ADVERSARIAL REVIEW (2nd pass) — what M15b does and does NOT control. Code/determinism CLEAN (verified):**
`accept_max_depth=None` reproduces the M15 output **bit-for-bit** (independently checked by hashing the
pre- and post-M15b generators; golden-hash test added), the cap is a pure post-hoc filter that does not
perturb the RNG draws, depths are correct (no off-by-one; every accepted `s_inf` is a true fixed point),
configs are one-knob, budget is as stated. **No blocking issue.** But two residual differences between the
uniform control and the mixed task survive the cap, so leg 2 is *strongly supported, NOT fully isolated*:
- **Depth is only MAX-matched, not distribution-matched.** Capping equalises the tail (max 6, 0% rows >6,
  target fixed at T=6) but the *central* depth still differs: uniform-78-cap6 **mean 3.91** (median 4),
  uniform-13-cap6 **mean 3.60** (median 3) vs mixed **mean 2.90** (median 3). The uniform controls are
  ~0.7–1.0 step **deeper** on average — and that residual is **NOT conservative** (deeper ⇒ wider
  light-cone, which could itself favour the loop), so part of the uniform edge could be central-depth, not
  uniformity. (Counter-pressure: the mixed task is *harder* — ff EM 0.255 vs 0.36 — and by M11 harder
  should favour the loop, yet the loop ties there; so depth and hardness push oppositely and the net is
  genuinely entangled.)
- **"Uniform vs non-uniform" conflates two things, and the second is DEFINITIONAL/unfixable:** the mixed
  task uses 4 per-position truth tables; the uniform control uses 1. A non-uniform local rule *necessarily*
  has ≥2 truth tables, so "spatial constancy of the rule" cannot be separated from "single truth table" —
  they are the same property. Hence leg 2 can never be isolated to translation-invariance *alone*; the
  honest claim is "the edge needs a *uniform* local rule (spatially constant ⇒ single shared operator)."
- **The effect is EM-only at matched token-acc** (the M9 coherence signature): the loop beats ff on
  *token accuracy* on NONE of the three w=24 tasks (uniform-78 Δacc −0.004 ns; uniform-13 +0.010 ns; mixed
  −0.009 ns). Leg 2 is a whole-row-coherence claim, not a per-cell-accuracy claim.
- **Power:** each significant cell is 9/1, p=.021 — the *minimum* a 10-seed sign test with one dissenter
  gives (one more adverse seed → 8/2, p=.11). Two rules, one width. The "tie on mixed" is an underpowered
  null (3/6, p=.51). So leg 2 is "supported across two rules at the 9/1 floor," not heavily over-powered.

**Reading (per §8) — leg 2 STRONGLY SUPPORTED (not fully isolated); the decomposition holds.**
- **At max-depth-matched (≤6), hardness running against the result, the loop beats ff on the UNIFORM task
  (+0.090 / +0.175 EM, 9/1, p=.021) but TIES on the NON-uniform mixed task (−0.028, ns).** The mixed task
  is *harder* (ff EM 0.255 < 0.36), which by M11 should *help* the loop, yet the loop wins only on the
  uniform tasks — so hardness cannot explain the pattern (it predicts the opposite). The leading
  remaining difference is rule-uniformity. **Not "isolated":** the uniform controls are also ~1 step
  deeper on average (residual, non-conservative) and "uniform" is entangled with "single truth table"
  (definitional). So: **the loop-beats-MLP EM edge tracks rule-uniformity at matched max-depth and against
  the hardness gradient** — strong evidence uniformity is required, with the central-depth residual the
  one un-eliminated alternative.
- **Leg 1 reconfirmed and orthogonal (CLEAN):** the joint-state mechanism Δ(stepDS−dec_stepDS) is large
  and 10/0 on the uniform controls (+0.31) *and* on the mixed task (+0.206) — present regardless of
  uniformity. Being a within-task arm contrast it is immune to the cross-task depth/hardness/cardinality
  mismatches, so it is the clean leg.
- **Mechanistic reading (HYPOTHESIS, not isolated):** a uniform rule makes the one-step operator a single
  shared operator, which the weight-tied loop matches to beat a one-shot MLP. M15b establishes the
  *dependence* on uniformity; it does not prove the *operator-sharing* account.

**Net.** The M8–M12 result decomposes into **(1)** a **joint-state coherence mechanism** (deep+local,
within-task, transfers off-CA — clean, M15) and **(2)** a **loop-beats-the-MLP** EM edge that needs a
**uniform local rule** (M15b — strongly supported: present on two depth-max-matched uniform CAs, absent on
the non-uniform mixed task, with the hardness gradient against it). Leg 2 is **not fully isolated** —
central depth is only max-matched (uniform ~1 step deeper, non-conservative) and uniformity is
definitionally entangled with rule-cardinality — and is EM-only, w≤24, two-rule, 9/1. The one further
tightening available (B1): a depth-DISTRIBUTION-matched uniform control (subsample/recap uniform to the
mixed mean depth) would remove the central-depth residual; rule-cardinality (B2) is unfixable in
principle. Caveats: w≤24; orbit rules; rejection-filtered (basin-conditioned) inputs; depth max-matched
not distribution-matched; hardness directional-not-exact; EM-only. Tracked:
`results/m15b_uniform_matched_screen_*.{json,csv}`, `results/m15b_uniform_matched_*.{json,csv}`.

---

## M15c — DONE. Close the leg-2 central-depth residual: a depth-DISTRIBUTION-matched uniform control. Leg 2 SURVIVES depth-control on rule 13 (clean), and rule 78's earlier edge is revealed as partly depth.

The 2nd review's one un-eliminated leg-2 alternative was the central-depth residual: M15b only MAX-matched
depth (cap 6), leaving the uniform controls ~1 step DEEPER on average (mean 3.9/3.6 vs mixed 2.9) — and
deeper is NOT conservative (wider light-cone could itself favour the loop). M15c removes it.

**The control.** Added a `depth_profile` parameter to `make_mixed_converge`: it stratified-subsamples
accepted rows to a target per-depth histogram, so any two tasks given the SAME profile have BIT-IDENTICAL
convergence-depth distributions. Ran mixed {78,92,141,197}, uniform {78}, uniform {13} all subsampled to
the **intersection** of their depth histograms (profile [0, .019, .126, .438, .285, .121, .012],
**mean depth = 3.40 for all three, verified**), w=24, full M10 arm set, 10 seeds. Determinism + matched-
histogram + golden-hash tests added (162 tests, ruff clean). Now depth is held fixed bin-for-bin; the only
remaining mixed-vs-uniform differences are rule-uniformity and (definitionally) rule-cardinality.

**Result — Δ(loop−ff) EM at IDENTICAL depth distribution (10 seeds, sign-test p):**

| task @ w=24, depth-dist-matched (mean 3.4) | ff EM | Δ(loop−ff) EM | Δ(stepDS−dec_stepDS) EM | Δ(loop−untied) EM |
|---|---|---|---|---|
| **uniform rule 13** | 0.353 | **+0.210 (10/0, p=.002)** | +0.359 (10/0) | +0.454 (10/0) |
| **uniform rule 78** | 0.443 | +0.032 (7/3, p=.34 **ns**) | +0.287 (10/0) | +0.333 (10/0) |
| **mixed orbit1** (non-uniform) | 0.204 | −0.005 (4/6, p=.75 ns) | +0.188 (10/0) | +0.166 (10/0) |

**Reading (per §8) — the residual is closed; leg 2 survives on rule 13, and rule 78 is shown to have been
depth-inflated.**
- **Leg 2 CONFIRMED depth-controlled on rule 13 — the cleanest single piece of leg-2 evidence in the
  project.** At a depth distribution *identical* to the mixed task (same histogram, mean 3.4), the loop
  beats ff on uniform rule 13 by **+0.210 EM (10/0, p=.002)** while the non-uniform mixed task **ties**
  (−0.005, ns). And the mixed task is *harder* (ff EM 0.204 vs 0.353), so by M11 it should favour the loop
  *more* — yet only the uniform rule shows the edge. Depth is no longer a possible explanation (held fixed
  bin-for-bin), hardness runs against the result, budget is conservative (untied over). So **a uniform
  local rule yields a loop-beats-ff coherence edge that depth cannot explain, absent on the non-uniform
  map at identical depth** — leg 2 stands on its own.
- **Rule 78's M15b edge was PARTLY DEPTH (honest correction).** At max-match (depth mean 3.9) rule 78 gave
  +0.090 (9/1, p=.021); at depth-distribution-match (mean 3.4) it drops to **+0.032 (ns)**. Matching depth
  down also made rule 78 ff-*easy* (ff EM 0.44 — little coherence headroom, an M14-style no-room cell), so
  this is *inconclusive* for rule 78, not a refutation — but it does show the M15b two-rule 9/1 result
  overstated: under full depth control, leg 2 rests on **rule 13 (robust)**, with rule 78 inconclusive.
- **Leg 1 fully confirmed at matched depth:** Δ(stepDS−dec_stepDS) EM is +0.19 / +0.29 / +0.36 (all 10/0,
  p=.002) across mixed and both uniform cells, with decoupled < ff everywhere (0/10) — the joint-state
  mechanism is robust to depth-control and present regardless of uniformity, as expected for a within-task
  contrast. **P1 survives all three** (10/0, conservative).

**Net.** Closing the central-depth residual *strengthens* the evidential basis even as it trims the count:
leg 2 now rests on **one fully depth-controlled rule (13: +0.21, 10/0, hardness-conservative)** rather than
two depth-confounded floor-significant ones, and rule 78 is honestly downgraded to inconclusive-at-matched-
depth. Combined statement: **the loop-beats-the-MLP EM edge requires a uniform local rule — demonstrated
depth-controlled on rule 13 (depth held identical to the non-uniform mixed task, hardness against the
result); it is rule-dependent and was partly depth-inflated for rule 78.** The only leg-2 caveat now
un-removable is the definitional uniformity↔rule-cardinality entanglement (1 vs 4 truth tables). Leg 1
(joint-state = deep+local, transfers off-CA) and P1 remain clean and depth-controlled. Tracked:
`results/m15b_depth_matched_*.{json,csv}`.

---

## M16 — DONE. Reframe the project: retire the unsatisfiable §9 gate; re-imagine Task C around the mechanism we found. (Writing milestone — no new runs; the experimental program M0–M15c is complete.)

This is a documentation/decision milestone, not an experiment. After M15c the experimental
program is complete and the highest-value remaining action (flagged since M13) was to rewrite
§9 so the project's success criterion matches what the evidence actually established. Done here.

**What changed in CLAUDE.md.**
- **§9 rewritten** from a flat "do-not-do" list into four parts: §9.1 retires the old gate,
  §9.2 states the settled finding, §9.3 re-imagines Task C, §9.4 keeps the genuine don'ts.
- **§3 Task C row** repurposed `compositional` → `nested_converge` ("earn H/L against the single
  loop", deferred/gated, see §9.3).
- **§11(c)** re-pointed: the §9-gate rewrite is marked DONE; the two legitimate frontiers are now
  named (the §9.3 Task C, gated; the §9.4 real-tabular bridge).

**Why retire the gate (not just mark it met).** The old gate — "no H/L hierarchy until the single
loop beats its control on Task A *and* Task B" — is **structurally unsatisfiable**, not merely
unmet. M6a built the one task (`multi_parity`) where a generalist *should* beat both single-axis
controls and got **zero** loop-beats-both cells, plus falsified the weaker "never-worst" claim. A
weight-tied generalist judged against single-axis *specialist* controls at a fixed budget cannot
dominate on both axes. So "beats both on A and B" was the wrong success criterion; the honest move
is to withdraw it, not to keep chasing it (M5/M6a already closed that).

**The finding §9.2 now anchors the project on:** *tied recurrence with a JOINT multi-output state
buys whole-row COHERENCE on LOCAL-UPDATE (CA) HARD multi-output FIXED-POINT targets.* Decomposed:
leg 1 = joint-state coherence mechanism (deep+local, transfers off-CA, clean within-task, M10–M15);
leg 2 = loop-beats-the-MLP EM edge needing a uniform local rule (depth-controlled on rule 13, M15c);
P1 = tying-positive over a fair untied stack (broadest, survives off-CA, M9/M13/M14). Scoped by what
it is NOT: not depth-extrapolation, not adaptive compute, not token-acc at large w, not universal
across operator families, not hard-convergence fixed points in general (CA/local-update specific),
not a capacity-independent beats-both.

**How Task C was re-imagined (the substantive design call).** The original Task C was a generic
`compositional` hierarchy probe gated on the loop beating the *FF baselines*. Both halves are now
wrong: (i) the ARC autopsy + our M0–M2 work already showed the *loop*, not the H/L hierarchy, is the
active ingredient; (ii) §9.2 shows the loop's value is coherence on local fixed-point maps, not
depth/composition. So a re-imagined Task C must ask the only open hierarchy question — **does a
two-timescale (H-slow/L-fast) loop buy coherence the validated single-timescale loop CANNOT, on a
target that is itself a hierarchy of local fixed points?** Concretely `nested_converge`: an outer
local map whose every step is the converged fixed point of an inner local CA (local+deep+ff-hard,
basin-rejection-filtered like `converge`/`mixed_converge`; difficulty = nesting levels / inner-vs-
outer depth / block size). **The control becomes the single loop** (`trm`), not the FF baseline —
plus `trm_decoupled` (still the joint state?) and a depth-matched untied stack (two-timescale tying
vs more depth). **The build-gate is now satisfiable:** build it only once a concrete instance shows
the single-timescale loop's coherence plateaus below the target (a within-loop ablation, can be met
or cleanly falsified) — unlike the retired generalist-beats-specialists gate. Until then Task C
stays deferred: building the H/L split before showing single-loop insufficiency would repeat the
exact HRM mistake the autopsy diagnosed.

**Net.** The project's orienting criterion is now the right one (§9.2), Task C is re-scoped to a
question the evidence actually leaves open with a gate that can be satisfied (§9.3), and the two
remaining frontiers (gated Task C; real-tabular bridge) are named without either being "in flight."
§9.3 also carries a **proposed `make_nested_converge` reference-generator sketch** (in the §3 style,
clearly marked NOT-built): a two-timescale fixed point (inner = per-block ring relax via `ca_step`;
outer = one full-ring `ca_step` per round), reusing the `make_mixed_converge` rejection-filter
boilerplate, so the next agent has a concrete starting point if/when the build-gate is met. No code,
configs, runs, or dependencies changed. Tracked: CLAUDE.md §3/§9/§11(c).

---

## M17 — [VERDICT SUPERSEDED BY M18g — gate UNMET, M19 NOT earned; banner below] Build the §9.3 Task C substrate (`make_nested_converge`) and run the single-loop BUILD-GATE. Original (now-overturned) verdict: gate MET — the single-timescale loop's coherence plateaus below the target and the plateau is CAPACITY-ROBUST (not a learnability wall); building the H/L two-timescale loop (M19) is now earned. Critical refinement: a depth-matched UNTIED stack beats the tied loop at scale, so M19's decisive control set must include it.

> ⚠️ **VERDICT SUPERSEDED BY M18g/M18h — DO NOT ACT ON THE "gate MET / M19 earned" READING BELOW.** This
> entire M17 block reached its verdict by comparing a single loop against §4 controls **at unequal
> compute** (the loop's effective 4×, the controls 1×) — the §8 trap. At EQUAL compute (M18g, 400 epochs,
> all arms) a param-matched feedforward SHARES the single-loop's nested ceiling (Δ(trm−ff) EM +0.036 @ w24
> / −0.003 @ w32), and more data lifts every arm together (M18h, sample wall). So the "insufficiency" is a
> shared capacity/generalization wall, NOT a single-*timescale* deficit: **the gate is UNMET, M19 (H/L) is
> NOT earned, Task C is re-DEFERRED.** The block is retained verbatim for the audit trail; the live verdict
> is in the M18g/M18h sections below and in CLAUDE.md §9.3/§11.

§9.3 (M16) re-imagined Task C as a two-timescale fixed point and gated building the H/L hierarchy on a
*satisfiable, within-loop* criterion: build it only once a concrete `nested_converge` instance shows the
validated **single-timescale** loop's whole-row coherence **plateaus below the target** (one joint-state
timescale insufficient) on a genuinely two-timescale structure. M17 builds the substrate and runs that
gate. It does **NOT** build the H/L module — doing so before demonstrating single-loop insufficiency is the
exact HRM mistake the ARC autopsy diagnosed.

**The substrate (`make_nested_converge`, `src/looptab/data/generators.py`).** A hierarchy of local fixed
points, exactly the §9.3 sketch. A ROUND = one SLOW outer full-ring `ca_step` (the only op that couples
across block boundaries) then a full FAST inner relax (each block iterated on its OWN ring of width
`block_w` to a per-block fixed point via `_inner_relax`). Target `s_inf` = the JOINT fixed point of
`round_ = inner_relax ∘ outer_step`; rows are rejection-filtered to the convergent basin EXACTLY as
`make_mixed_converge` (depth = #outer rounds; raises loudly on a non-converging rule pair). All-integer ⇒
bit-exact. Wired into the dataset/trajectory/curriculum machinery + config schema unchanged. 11 new tests
(determinism, golden hash, joint-fixed-point, two-timescale, balance, trajectory-by-round, cycling-raise);
full suite green (96 tests), ruff clean.

**Screened instance** (`scratchpad/screen_nested.py`, a structural screen mirroring M8/M12/M15): a
256-pair-ish sweep over (inner_rule, outer_rule, n_blocks, block_w) for convergence + balance + non-
triviality + genuine two-timescale depth. Locked **inner_rule=13, outer_rule=79, block_w=8** (both orbit-0
ff-hard converging ECAs, uniform per level), `n_blocks∈{3,4}` ⇒ **w∈{24,32}** (the M9–M12 regime): >0.97
convergent, perfectly balanced (0.50), **zero** trivial targets, and genuinely two-timescale (mean ~4–5
outer rounds, each ~4–5 inner steps; one inner relax alone ≠ target for >99% of rows, one round ≠ target
for >50%). The two-timescale-ness is asserted as a determinism-grade test, not just screened.

**Gate run (`m17_nested_converge_gate.yaml`, the M10/M11/M12 arm set + T~1..6 curriculum + step-aligned DS,
10 seeds, base hidden=latent=64).** Absolute whole-row exact-match (target = the true joint fixed point,
i.e. EM=1.0):

| arm | EM @ w=24 | EM @ w=32 | token-acc @ w=24 |
|-----|-----------|-----------|------------------|
| trm_nods (single-timescale loop) | **0.559** | **0.371** | 0.873 |
| ff_matched (§4a) | 0.489 | 0.341 | 0.871 |
| untied_matched (§4b) | 0.533 | 0.365 | 0.837 |
| trm_decoupled_nods | 0.467 | 0.281 | 0.831 |

So the single loop **plateaus far below the target** (EM 0.56/0.37 ≪ 1.0), but every param-matched control
plateaus in the SAME band (ff within token-acc *noise* of the loop). On its own this is ambiguous — it
could be a generic capacity/learnability ceiling (the M3a/M4 shared-wall signature) rather than a single-
*timescale* deficit. **In-regime checks confirm the substrate is the right kind of task:** leg-1 joint-
state mechanism reproduces (Δ(trm_nods−decoupled) EM +0.092 @ w24 / +0.090 @ w32; trainability-clean
Δ(stepDS−decoupled_stepDS) EM +0.127 @ w24; decoupled is the worst arm) and leg-2 holds at w≤24
(Δ(trm_nods−ff) EM +0.070, token +0.003 ns — the M9 EM-at-matched-token-acc signature). P1 base
Δ(trm_nods−untied) EM +0.025 ns (and `untied_matched` is +2.5%/+3.1% OVER budget here, so conservative).

**The decisive probe (`m17b_nested_capacity.yaml`, hidden=latent=128, w=24, 10 seeds) — capacity wall vs
structural ceiling.** M11 established the lever: on ECA-`converge` scaling 64→128 *amplified* the loop and
climbed toward solving. Here it does **not**:

| arm | EM @ hidden=64 | EM @ hidden=128 | Δ |
|-----|----------------|-----------------|---|
| trm_nods | 0.559 | **0.593** | +0.034 (~3× params, barely moves) |
| ff_matched | 0.489 | 0.486 | flat |
| untied_matched | 0.533 | **0.634** | **+0.101 (now the BEST EM arm)** |
| trm_decoupled_nods | 0.467 | 0.415 | − |

Two clean conclusions: **(1) the single-loop plateau is CAPACITY-ROBUST** — 3× params lifts the tied loop
only +0.03 EM (to 0.59, still ≪ 1.0), so it is NOT a capacity wall and NOT the M11 "scale amplifies"
pattern; the single-timescale tied loop has a genuine ceiling on this two-timescale target. **(2) P1
REVERSES at scale** — the depth-matched untied stack climbs +0.10 to 0.634 and **beats** the tied loop:
Δ(trm_nods−untied) EM **−0.042, 0/10, p=.002**, budget-clean (untied 0.988× budget). This is the FIRST
place the project's broadest "tying-positive" leg fails. Leg-1 strengthens with size (Δ(trm−decoupled) EM
**+0.178**, 10/0, p=.002) and leg-2 strengthens (Δ(trm−ff) EM **+0.106**, 10/0, p=.002) — both consistent
with M11, so the joint-state regime is intact; it is specifically *tying-vs-untied-depth* that flips.

**Verdict — gate MET, with a sharpened control requirement for M19. [⚠️ OVERTURNED BY M18g — see the
banner at the top of this M17 block; at equal compute a feedforward shares this ceiling, so the gate is
UNMET and M19 is NOT earned.]** By the §9.3 build-gate as worded:
(a) the single-timescale loop's coherence plateaus below the target — YES, and now shown **capacity-robust**
(the precondition is not a learnability/capacity artifact); (b) the structure is genuinely two-timescale —
YES (built + tested); (c) the "single loop already solves it ⇒ null" escape clause does **not** apply
(EM 0.59 ≪ 1.0). So **building the H/L two-timescale loop (M19) is now earned** [⚠️ OVERTURNED by M18g — a
feedforward shares this ceiling at equal compute; M19 is NOT earned] — the HRM mistake is
avoided because single-loop insufficiency was demonstrated *first*. **Critical refinement from M17b:** the
demonstrated insufficiency is of the single shared *operator* (weight-tying) — an untied multi-operator
stack already extracts headroom the tied single-timescale loop cannot (0.634 vs 0.593). An H/L loop adds a
second *operator at a second timescale* (tied within each), which is the parameter-efficient version of
"more operators." Therefore **M19's decisive control set must include the untied stack**: the H/L loop has
to beat BOTH the single-timescale `trm` (does a 2nd timescale add coherence?) AND the depth-matched untied
stack (is it the second *timescale*, or merely more independent operators?), alongside `trm_decoupled`
(still the joint state?). Only an H/L loop that beats the untied stack at matched budget proves a second
*timescale* — not just more operators — is the active ingredient. This is the autopsy's "earn the
hierarchy" done honestly, now against the untied stack too.

**What M17 did NOT do (scope).** It did not build the H/L module (that is M19 — [⚠️ M18g shows it is NOT
earned: re-deferred]). It
did not sweep more rule pairs / block sizes / a full size ladder (one screened instance + one capacity
point; the verdict rests on capacity-robustness at w=24, the leg-2 regime). It did not do a per-row depth-
stratified error analysis (a stronger but heavier confirmation that the loop's errors concentrate on high-
outer-round rows — a candidate sharpening for M19). Tracked: `make_nested_converge` + tests, the gate/probe
configs, their summary CSVs; CLAUDE.md §3/§9.3/§11.

## M18 — TRM-faithful ingredients + THREE adversarial reviews (M18b converge / M18c ablation / M18a wall / M18d nested / M18e+M18f compute-controls / M18g+M18i equal-compute gate @ h64+h128 / M18h+M18j data-sweep). TWO headlines fell to the reviews: (1) "canonical deep supervision is a large win" was a COMPUTE CONFOUND — the gain is just 4× more optimization, the detached-carry MECHANISM is inert (B1/M18e/M18f); (2) the M17 "Task C gate MET → build H/L (M19)" verdict FAILS the equal-compute control test (M18g/M18i) — at equal compute the single-timescale loop stays below the target at every capacity and the gap is data-bound (M18h/M18j), not a timescale deficit, so M19 is NOT earned and Task C is re-deferred. Net positives that survived all three reviews: the §9.2 legs (joint-state, tying, leg-2) reproduce AND grow with capacity on the two-timescale family at equal compute, and M18i kills the M17b "P1 reverses at scale" confound (a 1×-compute artifact).

**Motivation.** A literature scan of 2024–26 looped-model work (TRM ablations arXiv 2510.04871; HRM
"Perspectives" arXiv 2510.00355; "What Makes Looped Transformers Better" arXiv 2510.10089; Tab-TRM
arXiv 2601.07675) flagged four ingredients the repo's `TRM` lacked, ranked by TRM's own ablations:
(1) **canonical deep supervision** — an OUTER loop of `N_sup` supervised passes carrying `(z,a)`
across them with the carry **detached** between passes (the ARC autopsy's named active ingredient;
the repo's "deep supervision" was a *different* thing — per-step readout losses inside one fully
back-propagated forward); (2) **EMA(0.999)** of weights (TRM's 2nd-largest knob, +7.5 pts); (3)
**RMSNorm** on the latent; (4) **n:1 cadence** — `n_latent` z-updates per answer update (TRM uses 6;
the repo did 1:1). All four were added **additively/opt-in** (bit-identical when off; 155 tests,
new tests cover bit-identity-off, the N_sup routine's determinism, and EMA determinism+effect).
Per the plan (anchor-only, bundle-first), a single `trm_faithful` arm stacks all four at a tractable
"lite" setting (`n_sup=4`, `ema_decay=0.999`, `use_rmsnorm=true`, `n_latent=2` — vs TRM's n=6/N_sup=16,
chosen to bound the faithful arm's ~8× compute), tested on two anchors, then ablated.

**M18b — converge coherence anchor (rule 78, w∈{24,32}, standard-train final loss, 8 seeds): the
bundle is a LARGE win and extends loop-beats-both past M9's boundary.**
- Δ(trm_faithful − trm_nods): w24 **+0.030 acc / +0.359 EM**; w32 **+0.067 acc / +0.596 EM** (all 8/0,
  p=.0078). Per-arm EM @ w24: trm_nods 0.584 → **trm_faithful 0.944**.
- Δ(trm_faithful − ff_matched): w24 **+0.371 EM**; w32 **+0.555 EM** (8/0). **Honest caveat (review
  S5):** on THIS standard-train path the *plain* loop does NOT cleanly reproduce M9's loop-beats-ff —
  Δ(nods−ff) EM is +0.011 ns (5/3) @ w24 and **−0.041, 0/8 @ w32** (the loop ties/loses to the shallow
  MLP). So the faithful arm does not *amplify* an existing plain-loop edge here — at w32 there is none;
  the faithful win is **manufactured by the extra deep-supervision training**, which is the point (it is
  a training effect, see M18c/M18e), but "extends loop-beats-both" should be read as "the FAITHFUL loop
  beats both," not "the loop's M9 edge survived to w=32 on its own."
- Mixed anchor reproduction on the plain-train path: the joint-state mechanism reproduces (Δ(nods−decoupled)
  +0.346/+0.084 EM, 8/0) and tying-positive reproduces @ w24 (Δ(nods−untied) +0.212 EM, 8/0); the
  loop-vs-ff leg does NOT (above). So this path re-derives M10's mechanism but not M9's w≤24 loop-beats-ff.

**M18c — per-ingredient ablation (rule 78, w=24, 8 seeds, one knob per arm): ingredient 1 carries the
ENTIRE effect; the other three are individually neutral-to-harmful.** Δ EM vs trm_nods (per-arm EM):
- **trm_nsup (N_sup detached deep supervision) ALONE: +0.295 EM (0.584→0.879, 8/0, p=.0078)** — ~82%
  of the full bundle. The dominant ingredient. **BUT `trm_nsup` runs n_sup=4× the optimizer steps/batch
  at equal epochs, so this is NOT yet separated from "4× more SGD" (review B1)** — the carry-vs-compute
  isolation is M18e (below).
- trm_ema alone: **−0.411 EM (0.584→0.173, 0/8)** — EMA(0.999) is *catastrophic* alone at this step
  budget (100 epochs ≈ 1600 steps; the 0.999 window lags and under-fits).
- trm_rmsnorm alone −0.061 EM (ns, p=.07); trm_nlatent alone −0.056 EM (0/8). Neither helps a shallow
  6-step loop in isolation.
- trm_faithful (all four) +0.359 EM (0.944): the three off-ingredients add a further +0.065 ON TOP of
  N_sup despite each being individually ≤0. **Which one (or which interaction) is NOT isolated** — M18c
  runs no pairwise N_sup+EMA / N_sup+RMSNorm cells, so attributing this +0.065 to "EMA needs N_sup's step
  count" would be speculation (review S2); the honest statement is just "the off-ingredients net +0.065
  in combination with N_sup, source un-isolated."
- **Seed note (review N3):** M18 uses 8 seeds, not the project's customary 10. 8/0 → p=.0078 is the
  *minimum* that clears .05 on the two-sided sign test, so a single seed flipping would lose significance;
  the EM effect sizes (+0.30…+0.60) are far larger than that margin, but the sign-test headroom is thin.

**M18a — depth wall (rule 30 chaotic CA, non-convergent, T∈{8,16}, 8 seeds): NON-INFORMATIVE FLOOR (all
arms at EM≈0), not a controlled null about the ingredients (review S-2).** Every arm sits at EM≈0.0001 /
test-chance, so no Δ between arms is interpretable — this re-confirms rule-30 unlearnability (the known M7
dead end), it does NOT isolate a property of the faithful bundle. Read it as "the bundle can't rescue an
unlearnable-for-everyone target," nothing finer. At T=8 every arm is at test-chance (faithful 0.525 vs nods 0.523, baseline
0.503); faithful fits *train* marginally better (0.663 vs 0.616) but it does NOT reach test. At T=16
all arms collapse to baseline (~0.505), EM=0 for all. So canonical deep supervision lets the model fit
the trained depth slightly better but does **not** crack the M3a generalization wall on a non-convergent
chaotic target — consistent with M7's hypothesis that the fixed-point/path-independence bias of these
mechanisms is mismatched to a *moving* CA target.

**Net (PRE-REVIEW headline, now WALKED BACK — see M18e/M18f).** The pre-review reading was: the repo's
"deep supervision is inert" verdict (M0–M3b) used a non-canonical DS, and the canonical detached N_sup DS
is "a large clean isolated win" on the convergent anchor. **The adversarial PR review (B1) flagged the
compute confound, and M18e/M18f confirmed it: the win is NOT the deep-supervision mechanism, it is just 4×
more optimizer steps** (see below). So the honest M18b/M18c reading is: more training lifts the loop on the
undertrained convergent anchor; the carry mechanism the autopsy names is ~inert. Implementation is additive
— every committed M0–M15c result is bit-identical (defaults off). Configs: m18a/m18b/m18c.

**M18e/M18f — the review's B1 compute-control: the N_sup "win" is just more optimization, the detached
carry adds nothing (clean correction).** `m18c` showed `trm_nsup` (n_sup=4) +0.295 EM over `trm_nods`, but
n_sup=4 runs 4× the optimizer steps/batch — so "the detached-carry deep-supervision mechanism is the active
ingredient" was confounded with "4× more SGD." Two compute-matched controls settle it (converge rule 78,
w=24, 8 seeds): **(M18e)** a NO-CARRY arm (`n_sup_carry=false` — 4 passes/batch, each restarting from a
fresh `z0`: identical compute, no carry) gets +0.282 EM, so **Δ(carry − no-carry) = +0.012 EM, ns** — the
detached carry, the ARC autopsy's named mechanism, adds essentially nothing. **(M18f)** a PLAIN `trm_nods`
trained 4× the EPOCHS (400 vs 100) reaches **EM 0.873 ≈ trm_nsup's 0.879** — so neither the carry nor the
N_sup pass-structure buys anything a longer plain run doesn't. **Scope (review S-1):** m18e/m18f vary only
the **N_sup axis** (both n_latent=1, no EMA/RMSNorm), so they retire the *N_sup* claim specifically; the
other three ingredients are retired separately by m18c (each ≤0 alone). **Conclusion: the entire +0.295 is
"more optimization" on an undertrained convergent target; the canonical DS *mechanism* is ~inert, which
UPHOLDS the project's prior "DS is inert" verdict.** Consequence: M18b's "faithful loop beats both at w=32" is a
compute-unfair comparison (the loop trained 8× longer than ff; equal-compute ff untested) — do NOT read it
as a loop-beats-both result. Practical takeaway: **train loops longer on convergent fixed-point targets;
the four faithful ingredients add no mechanism (EMA-alone is harmful).** The additive machinery
(`train_deep_supervision`, EMA, RMSNorm, `n_latent`, `n_sup_carry`) stays as useful infrastructure + a
cautionary negative; defaults off, all prior results bit-identical. Configs: m18e_compute_matched,
m18f_epochs_matched. (Credit: the adversarial PR review caught this; the pre-review M18 headline was wrong.)

**M18d — the M17×M18 cross-check: faithful DS on the Task C nested gate (does M17's verdict survive
the lever M17 never had?).** M17 ran the §9.3 build-gate with the OLD training and found the
single-timescale loop plateaus capacity-robustly below the nested target (EM 0.56/0.37 ≪ 1.0). But M18
found a *training* lever M17 did not test — and `nested_converge` is the same class of convergent
local-update fixed-point target. So before the H/L build (M19) is truly earned, the gate's
"insufficiency" had to be re-checked against faithful DS. Ran their locked instance (inner_rule=13 /
outer_rule=79 / block_w=8, w∈{24,32}) on the STANDARD-train path (so the faithful `n_sup` arm applies),
8 seeds, with `trm_nods` in-run. **Result — gate SURVIVES and STRENGTHENS.** Absolute EM: trm_nods
0.689→trm_faithful **0.819** (w24, +0.130), 0.524→**0.616** (w32, +0.091). So the faithful arm (which on
this regime = 4× more optimization, NOT the DS mechanism — M18e/M18f) DOES lift the single-timescale loop
— but only modestly, **far less than the +0.36/+0.60 the identical training gives on the *single*-timescale
`converge` anchor (M18b)**, and **still well below the EM=1.0 target**. The nested target resists more
optimization exactly the way a genuine two-timescale deficit should. Combined with M17b's
capacity-robustness, single-timescale insufficiency is now robust to BOTH levers (capacity AND a 4×-compute
training lever). **[CONCLUSION SUPERSEDED BY M18g — see below.]** M18d concluded "M19 remains earned, bar
~0.82," but M18d only trained the *faithful* lever on `trm`; the §4 controls ran at 1× compute. The 2nd
adversarial review flagged that the decisive comparison — controls at EQUAL compute — was never run. M18g
runs it and overturns the verdict (the nested ceiling is SHARED, not single-timescale). In-regime checks
from M18d (leg-1 Δ(trm−decoupled) +0.082/+0.089; loop beats untied at base +0.154/+0.098) do hold and are
confirmed at equal compute by M18g. Config: m18d_faithful_nested.

**M18g — review-2 Experiment 1 (DECISIVE): equal-compute controls KILL the M17 "gate MET" verdict; the
single-loop's nested ceiling is SHARED by a feedforward, not a single-timescale deficit.** The M17/M18d
gate compared a 4×-compute loop against 1×-compute §4 controls — the §8 trap (the loop trivially
degenerates into a deep net; only matched controls are informative). M18g re-ran the nested gate with EVERY
arm at **equal compute (400 epochs, n_sup=1, standard-train — only the architecture varies)**, w∈{24,32},
8 seeds. **At saturation (all train_acc ≈ 0.99–1.0) the arms cluster:**

| arm | EM w24 | EM w32 | train_acc |
|-----|--------|--------|-----------|
| trm_nods (single loop) | **0.750** | 0.572 | ~1.0 |
| ff_matched (§4a)       | 0.714 | **0.575** | ~1.0 |
| untied_matched (§4b)   | 0.685 | 0.535 | ~1.0 |
| trm_decoupled          | 0.640 | 0.411 | ~1.0 |

Δ(trm − ff) EM = **+0.036 @ w24 (8/0, p=.008)** and **−0.003 @ w32 (4/4, p=1.0 — a TIE)**. The precise,
honest read (review-3 S1, against over-correction): the loop **retains a small but significant leg-2 EM
edge at w24 (+0.036, 8/0)** and **ties ff at w32** — but +0.036 is the M9-style leg-2/coherence edge,
**nowhere near the ~0.25 timescale-sized headroom** the gate needs (the loop is at 0.75, the target is
1.0, and a *feedforward* sits at 0.71 right alongside it). So the framing is NOT "ff exactly matches the
loop" but "ff hits the same generalization BAND ≪ 1.0; the loop's residual edge is small, w24-only, and
of leg-2 size — not timescale size." The "insufficiency" is therefore a **shared capacity/generalization
wall, not a single-*timescale* deficit** (the M3a/M4 shared-wall signature: everyone fits train, everyone
generalizes to the same ~0.7 band). **The §9.3 build-gate is UNMET** (it needs a timescale-*specific*
insufficiency to motivate a second timescale), so **M19 (H/L build) is NOT earned and Task C is
re-DEFERRED.** **What survives equal compute (the real nested result — all three §9.2 legs, at honest
scope):** leg-1 joint-state Δ(trm−decoupled) EM **+0.110/+0.161 (8/0, p=.008)**; P1 tying Δ(trm−untied)
**+0.065 @ w24 (8/0)** / +0.037 @ w32 (7/1, ns) — *conservative: `untied_matched` is +2.5%/+3.1% OVER
budget (`within_tol=False`), the loop beats a slightly bigger control, as in M9/M12/M17b (review-3 S3)*;
and **leg-2 loop>ff EM +0.036 @ w24 (8/0)** — small and w24-only (the M9 w≤24 boundary), present here AND
in M17's curriculum run (+0.070 @ w24, 10/0), but too small/width-bounded to bear on the gate. So all
three §9.2 legs reproduce on the two-timescale family; none of them is timescale-*sized*. (M17b's "P1
reverses at hidden=128" was 1×-compute; M18i below tests it at EQUAL compute and the reversal does NOT
survive — the loop beats untied at hidden=128, budget-clean.)
**Re-gate condition for any future M19:** a nested instance where the loop plateaus below target AND
ff/untied do NOT share the ceiling at equal compute. Config: m18g_nested_equalcompute. Credit: the 2nd
adversarial PR review proposed this exact experiment.

**M18h — review-2 Experiment 2: the shared nested ceiling is a SAMPLE wall (lifts with data), confirming
it is NOT a single-timescale deficit.** Re-ran the nested gate at w24, 400 epochs, with **4× the data
(n_train 4000→16000)**, 8 seeds. Both arms climb ~+0.18 together: trm_nods EM **0.750→0.925**, ff_matched
**0.714→0.900**, both train_acc ~1.0; the loop keeps its small w24 leg-2 edge (Δ(trm−ff) +0.025, 8/0,
*still significant but shrinking* from +0.036 — not exact lockstep, but both rise toward 1.0). So the
~0.75 "ceiling" was **under-data, not a structural single-timescale wall** — more data lifts every arm
substantially (the M5 sample-wall signature). The constructive lever on this nested instance is **more
data/capacity, NOT a second timescale** — confirmation that M19 (H/L) is unmotivated here. **(The 64k
point that settles "sample wall vs capacity plateau" is M18j below: both arms reach ~0.97–0.99 — a PURE
sample wall, lifts to ~1.0.)** Config: m18h_nested_data16k.

**M18i — review-3 EXP-1 (closes the gate-story hole): equal compute at hidden=128 — the verdict HOLDS,
and it KILLS the M17b P1-reversal confound.** The re-deferral rested on hidden=64 equal-compute; the
*original* "structural ceiling" claim came from M17b at hidden=128 (1×-compute), where P1 reversed
(untied 0.634 > loop 0.593). M18i re-runs the gate at hidden=128 with ALL arms at equal compute (400
epochs, train_acc=1.0), w24, 8 seeds. Absolute EM: **trm_nods 0.786, ff 0.722, untied 0.730, decoupled
0.689.** Three reads: **(1)** the loop's edge over ff GROWS with capacity at equal compute — Δ(trm−ff) EM
+0.036 (h64) → **+0.064 (h128), 8/0** — it does NOT vanish (the M11 "edge grows with size on
hard-convergence targets" signature); **(2) M17b's "P1 reverses at scale" was a 1×-COMPUTE artifact —
at equal compute the loop BEATS untied at hidden=128** (Δ +0.056 EM, 8/0, and untied is now *within
budget*, 0.988 — budget-clean), reversing the reversal; **(3)** but the loop STILL plateaus at **0.786 ≪
1.0** even at 2× capacity — capacity (64→128) lifts it only +0.036, whereas data (4×, M18h) lifted it
+0.175. **So the gate verdict HOLDS but with a cleaner argument than m18g's "ff shares the ceiling":** the
loop *is* the best single-timescale arm (beats ff and untied, edge growing with capacity = the §9.2 legs
reproducing/strengthening on the two-timescale family), yet it remains far below the target at every
capacity, and the gap-to-target is **data-bound (M18h), not capacity- or timescale-bound** (M18i). The
loop's edge over ff is the ordinary leg-2 coherence edge (which grows with size on ALL hard-convergence
targets, M11), **NOT** a sign the loop *uniquely* fails where a hierarchy would succeed. The §9.3 gate
needs the latter; it is absent → **M19 still NOT earned.** (Bonus: M18i is the cleanest P1 evidence in the
nested thread — loop > untied at equal compute, budget-clean, at hidden=128.) Config:
m18i_nested_equalcompute_h128. Credit: the 3rd adversarial review proposed this exact experiment.

**M18j — review-3 EXP-2 (the airtight finish): at 64k data the single-timescale loop SOLVES the nested
target — pure SAMPLE wall, §9.3's own null clause triggered.** The data sweep (nested w24, 400 epochs,
trm_nods/ff_matched) completes the picture: EM **4k 0.750/0.714 → 16k 0.925/0.900 → 64k 0.990/0.969**.
With 16× data the single-timescale loop reaches **EM 0.990** (near-perfect) and ff 0.969 — both essentially
SOLVE the two-timescale fixed point. So the 4k "insufficiency" was **purely sample-complexity** (the M5
signature, now nailed with three points climbing to ~1.0), NOT a structural single-timescale deficit of any
kind. This literally triggers the §9.3 build-gate's own NULL clause — *"if the single loop already solves
the nested target, the hierarchy is unearned — report that null and stop"* — the single loop DOES solve it
given data. (Δ(trm−ff) EM shrinks across the sweep, +0.036→+0.025→+0.021, 8/0 throughout — the leg-2 edge
persists but vanishes into the ceiling.) **Final gate verdict: M19 (H/L) is definitively NOT earned; the
nested target is a sample wall the single loop clears with data.** Config: m18j_nested_data64k.

**Net (M18, after three adversarial reviews + M18e/f/g/h/i/j).** Two milestone headlines were retracted by the
review process — a model of the §8 discipline working as intended: **(1)** "canonical detached deep
supervision is a large win" → it is just more optimization; the autopsy's DS mechanism is inert (M18e/f).
**(2)** "the Task C gate is MET → build the H/L hierarchy (M19)" → the single-loop's nested
"insufficiency" is a **SAMPLE wall**: at equal compute the loop stays below target at every capacity (M18g
h64 / M18i h128, edge over ff growing but leg-2-sized), and the gap is data-bound — at 64k the single loop
SOLVES the target (EM 0.99, M18j), triggering §9.3's own null clause. M19 is unearned and Task C is
re-deferred. **What stands (survived all three reviews):** the additive faithful machinery
(`train_deep_supervision`/EMA/RMSNorm/`n_latent`/`n_sup_carry`) as infrastructure + a cautionary §8 case;
the built `make_nested_converge` substrate; the genuine §9.2-extension result that all three legs (leg-1
joint-state, P1 tying, leg-2 loop>ff) reproduce AND grow with capacity on the two-timescale family at equal
compute; and M18i's correction of the M17b "P1 reverses at scale" confound (a 1×-compute artifact). The
practical pointer: **train loops longer / give more data on convergent fixed-point targets; do not infer a
"mechanism" or a "hierarchy mandate" from an undertrained or compute-unfair comparison** — the controls, at
equal compute, are the whole point (§8). All M18 knobs additive/off-by-default → every committed M0–M15c
result bit-identical.

---

## M20 — DONE (headline SOFTENED by adversarial review — see the correction block at the end of this entry). The §9.4 real-tabular bridge, step 1: joint multi-label MODELING (vs binary-relevance) reproduces on real `emotions`/`yeast` (leg-1 direction, 2/2), and on the richer-coupling yeast the tied loop is DIRECTIONALLY the best param-matched arm on subset accuracy via the coherence mechanism — a directional first transfer of the loop's value off the synthetic suite. NOTE: the original reading below said "TRANSFERS / full loop-beats-both"; the review corrected this — significance rests on overlapping single-dataset splits (illustrative, not clean) and Δ(trm−decoupled) isolates joint-output-vs-per-label, not the loop's latent (the loop-specific edge is the small Δ(trm−ff) +0.018). The body is kept verbatim for the audit trail; the corrected verdict is the block at the end. **A SECOND review pass then dropped the certainty further, and the PROPER EVALUATION (F1 + 10-fold CV, the final `M20 — PROPER EVALUATION` section at the very end) settles it: properly evaluated, M20 is a NEGATIVE for the loop thesis. The only robust finding (joint > binary-relevance, leg-1, on EM+F1) is NOT loop-specific — a plain joint MLP gets it too — and every loop-SPECIFIC edge (over the joint MLP / the untied stack) is EM-only and EVAPORATES under micro/macro-F1 (Δ(trm−ff) F1 is a 5/5 tie). Read the final section, not this header, for the verdict.** (M19, the H/L build, remains unearned and unbuilt; M20 is the *other* §9.4 frontier.)

**Why this milestone, and why now.** After M18 the synthetic program is complete and Task C is correctly
re-deferred (the nested "insufficiency" is a sample wall — M18j: the single loop solves it at 64k, EM
0.99). The two legitimate frontiers were (a) re-gating Task C — judged low-value/likely-unsatisfiable for
the same §8 reason the original gate was (the loop trivially degenerates into a deep net, so a matched
feedforward shares its ceiling) — and (b) the §9.4 real-tabular bridge: port the validated §9.2 finding
(*tied recurrence with a JOINT multi-output state buys whole-row COHERENCE*) to a real multi-output target
with the §4 control contract intact. M20 is (b).

**Why multi-label classification is the right port (the clean structural map).** Outputs are
binary-per-label, so the existing multi-output head + metrics apply unchanged; **exact-match = subset
accuracy** (whole labelset correct) is *exactly* leg-1's whole-row coherence metric; and `trm_decoupled`
(per-label latents, each seeing only its own answer) is literally **binary-relevance**, while the joint
latent is a learned label coupling. So **Δ(trm − trm_decoupled) on EM is a direct real-data test of
leg-1**, and the §9.2 legs map one-to-one: leg-1 = Δ(trm − decoupled), leg-2 = Δ(trm − ff), P1 =
Δ(trm − untied_matched).

**Honest prior going in (§8, now partly overturned).** Leg-1 was bounded to *local + deep* structure —
it nulled on a dense net (M13) and a shallow one (M14) — and generic tabular has neither, so the
pre-registered expectation was that leg-1 might null and P1 (the regime-independent leg) was the likelier
survivor. The data **overturned this for leg-1** (it reproduces on both datasets) and showed the *full*
loop-beats-both appearing on the harder dataset.

**Substrate (built + tested, additive).** A `multilabel` task that loads **vendored** numpy `.npz`
caches (`datasets/{emotions,yeast}.npz`, built once by `scratchpad/fetch_multilabel.py` from OpenML
data_ids 40589/40597 — canonical Mulan versions) so the *task path stays network-free and deterministic*
(§5): loaded with numpy only, guarded by a content sha256 = sha256(X.tobytes()+Y.tobytes()) checked
against a golden constant. New module `src/looptab/data/real.py` (`make_multilabel_splits`,
`load_multilabel_pool`); `make_splits` gains one `task == "multilabel"` branch; `TaskConfig.name` gains
`"multilabel"`; `eval/metrics.py` gains `subset_accuracy_baseline` (the EM analogue of
`majority_baseline`). Real-data adaptations, all in the loader: **disjoint train/test from a finite pool**
(a single seed-keyed partition, not two independent draws — no leakage), **continuous features z-scored
on TRAIN stats only** (the synthetic suite was all binary), and the §3 seed discipline adapted (the
dataset is the fixed *function*; per-seed variance = the random disjoint partition + model init). 14 new
tests (golden hash, determinism, **train/test disjointness**, train-only standardization, caps, the EM
baseline); full suite **186 tests** green, ruff clean. **Every M0–M18 result is bit-identical** (purely
additive: a new task name + new module; the only touched path is the gated `make_splits` branch).

**Datasets (canonical Mulan).** emotions: 593 rows, 72 continuous audio features, **6** correlated
labels, label cardinality 1.87, 0 all-zero rows. yeast: 2417 rows, 103 features, **14** correlated labels
(gene-function hierarchy), label cardinality 4.24, 0 all-zero rows. Constant-predictor EM floors
(`subset_accuracy_baseline`): emotions 0.153, yeast 0.103 (token-majority floors ~0.69 — inflated by
label sparsity, hence EM is the honest headline). Arms (mirroring M10): `trm` (joint loop, final loss,
budget reference), `trm_decoupled` (binary-relevance), `ff_matched` (§4a shallow JOINT MLP),
`untied_matched` (§4b fair untied stack), `untied_stack` (labelled ceiling). hidden=latent=64, n_steps=6,
300 epochs (all arms saturate train ≈0.985–1.0 ⇒ **equal, saturated compute**, the §8 requirement),
batch 128, 10 seeds. Budget parity ✓ within ±2% in both runs.

> ⚠️ **[SUPERSEDED — read the final "M20 — PROPER EVALUATION" section, not this block.]** Everything from
> here to that section used **overlapping random 70/30 splits and EM only**. The p-values below are
> anti-conservative (the splits overlap ~0.30) and the loop's apparent EM win is a modal-label-combo
> artifact that ties under F1. Retained verbatim for the audit trail; the corrected, properly-evaluated
> (F1 + 10-fold CV) verdict is the final section. (The `coherence_excess` column in the deltas CSVs is the
> M9-flagged per-arm descriptor — its cross-arm Δ is Jensen/dispersion-confounded and is NOT evidence;
> the verdict does not use it.)

**Per-arm EM (subset accuracy), 10 seeds:**

| arm | emotions EM | yeast EM | yeast token-acc | yeast train-acc |
|-----|-------------|----------|-----------------|-----------------|
| untied_stack (ceiling) | 0.276 | 0.150 | 0.762 | 1.000 |
| **trm (joint loop)** | 0.255 | **0.142** | 0.743 | 0.999 |
| ff_matched (§4a) | **0.269** | 0.125 | **0.752** | 1.000 |
| untied_matched (§4b) | 0.254 | 0.117 | 0.738 | 0.985 |
| trm_decoupled (binary-relevance) | 0.196 | 0.081 | 0.722 | 0.989 |

**Paired Δ on EM (sign test; 10 seeds):**

| Δ (EM) | emotions | yeast |
|--------|----------|-------|
| leg-1: Δ(trm − trm_decoupled) | **+0.059, 9/0, p=.004** | **+0.062, 10/0, p=.002** |
| leg-2: Δ(trm − ff_matched) | −0.014, 2/7, p=.18 (ns) | **+0.018, 10/0, p=.002** |
| P1: Δ(trm − untied_matched) | +0.001, 6/4, p=.75 (ns) | **+0.025, 10/0, p=.002** |
| Δ(trm_decoupled − ff_matched) | −0.073, 0/10, p=.002 | −0.044, 0/10, p=.002 |

**Reading (per §2/§8).**

1. **Leg-1 (joint-state coherence) TRANSFERS — significant on BOTH real datasets.** The joint latent
   beats binary-relevance on subset accuracy (emotions +0.059 9/0; yeast +0.062 10/0), and on both the
   decoupled arm falls **significantly below the shallow MLP** (−0.073 / −0.044, 0/10) — the exact M10
   signature (cutting the cross-label state drops the loop below even a shallow joint MLP). This is the
   project's most-replicated, most-portable mechanism, and it survives the move to real, continuous-feature,
   irregular-coupling tabular data. The pre-registered "leg-1 may null off local+deep" prior is overturned:
   real correlated labels supply enough joint structure for the joint-state mechanism to fire.

2. **Full loop-beats-both (legs 1 + 2 + P1) appears on the HARDER, higher-coupling yeast — a genuine
   real-data loop-beats-both on EM.** On yeast the tied joint loop is the **best param-matched arm** (EM
   0.142 vs ff 0.125, untied 0.117, decoupled 0.081), beating *both* mandatory controls 10/0 — and it is
   **driven by coherence, not raw accuracy**: the loop's per-label (token) accuracy is *lower* than ff's
   (0.743 < 0.752) while its whole-row EM is *higher* (coherence_excess Δ(trm−ff) +0.021). That is the M9
   "EM at matched-or-worse token-acc" mechanism reproduced on real data — and *conservative*, since the
   loop wins EM from a token-acc deficit. (untied_matched fits train slightly worse, 0.985, and is +0.7%
   over budget, so P1 is mildly conservative/optimization-confounded; the joint arms do NOT fit train
   better than ff — ff fits best, 1.000 — so the loop's test edge is generalization/coherence, not
   optimization.)

3. **The emotions/yeast split is the boundary, and it points the right way.** On emotions (6 labels,
   cardinality 1.87) only leg-1 fires: the loop *ties* the shallow joint MLP (leg-2 ns) and P1 is null —
   a shallow joint MLP captures the lighter label coupling just as well. On yeast (14 labels, cardinality
   4.24, 4× the rows) all three legs fire and the loop separates from ff. So the loop's *specific* coherence
   edge (over a shallow joint MLP / a fair untied stack) emerges where there are **more, more strongly
   coupled outputs** — the real-data analogue of the synthetic "needs enough output width / hard coherence"
   boundary (M9 w≤24→regime, M11 grows-with-size). **But this is a 2-point trend, not a controlled sweep**:
   emotions→yeast confounds label count (6→14), coupling strength, and n (4×) at once. The honest claim is
   "leg-1 transfers cleanly; legs 2/P1 + loop-beats-both appear on the richer-coupling dataset," not an
   isolated dependence on any one axis.

**Scope / caveats (what M20 does and does NOT establish).** (i) **Two datasets** — leg-1 is 2/2 and
robust (10/0 & 9/0), but legs 2/P1 rest on a single dataset (yeast); the cross-dataset difference is
uncontrolled. (ii) **EM/coherence-only** — on per-label (token) accuracy the loop ties or loses; this is a
whole-row coherence result, exactly as §9.2 scopes it, NOT a per-label-accuracy win. (iii) **Minor
trainability caveats** — decoupled (0.989/0.995) and untied_matched (0.985) under-saturate train vs the
joint arms (~1.0); the EM gaps dwarf the train-fit gaps and the decoupled arm has comparable/fewer params,
so these do not explain the effect, but they are the standard M10-style caveat surfaced. (iv) **One model
size / one n_steps** — leg-1 "grows with size" (M11) is untested here (the obvious next step: a hidden=128
point on yeast). (v) Not depth/algorithmic anything — M20 is the coherence mechanism, full stop.

**Net.** The single durable positive of the whole synthetic program — *tied recurrence with a joint
multi-output state buys whole-row coherence* — **is not a synthetic-CA artifact: it transfers to real
multi-label tabular.** Leg-1 (joint-state) reproduces on both datasets; on the harder, higher-coupling
yeast the tied joint loop beats *both* §4 controls on subset accuracy via the coherence mechanism (lower
token-acc, higher EM), a real-data loop-beats-both. This is the first time the loop's value has been
demonstrated off the synthetic suite. Bridgehead established; the natural next steps are a size/coupling
sweep to turn the emotions→yeast 2-point trend into a controlled axis, and more multi-label datasets.
Config: `m20_multilabel_emotions_smoke.yaml`, `m20_multilabel_yeast.yaml`. Tracked summaries:
`results/m20_multilabel_{emotions_smoke,yeast}_*_{curve,deltas,params}.csv` + JSON records.

**[Adversarial review — two corrections that SOFTEN (not retract) the headline.]** A post-run adversarial
review (the §8 discipline; same process that walked back M18) verified the implementation is clean —
bit-identical re-run, disjoint splits, train-only standardization, no leakage, budget parity, canonical
data, correct EM/Hamming metrics — and confirmed the coherence mechanism is *real* (it checked predicted
positive-rates are matched: trm 0.303 vs ff 0.298 vs true 0.297, so the "loop predicts more 1s" artifact is
ruled out; errors are genuinely clustered). But it caught two ways the reading above over-states:

1. **The within-dataset sign-test p-values are anti-conservative — do NOT lean on p=.002 as a clean
   significance call.** The 10 "seeds" are 10 *overlapping* 30%-of-pool test splits of the SAME fixed
   dataset (measured mean pairwise test-overlap ≈ 0.30), not fresh function draws like the synthetic seeds.
   So the paired Δs are positively correlated, the effective N is well below 10, and 10/0→p=.002 overstates
   significance. **The real independent evidence is the TWO DATASETS agreeing (and the effect sizes), not
   the 10 within-dataset splits.** (Future real-data runs should report this honestly, and ideally use
   repeated *nested* CV or hold a truly independent test set rather than overlapping random splits.)

2. **"Leg-1 transfers (joint STATE)" overclaims — Δ(trm − decoupled) only isolates joint LABEL MODELING
   vs binary-relevance, not the loop's joint LATENT.** `trm_decoupled` severs the joint latent *and* the
   joint readout; and the non-recurrent `ff_matched` *also* beats decoupled (the "M10 signature" −0.073/
   −0.044). So that Δ shows "any joint label model beats per-label," a benefit `ff` shares — it does NOT
   show the *loop's* joint state is the mechanism. The loop-specific attribution rests entirely on the
   **modest single-dataset Δ(trm − ff) = +0.018 EM (yeast)**, which is small and on overlapping splits.

**Corrected headline (use this):** *Joint multi-label modeling beats binary-relevance on two real datasets
(leg-1 directional, 2/2, decoupled-below-ff reproduced) — and on the richer-coupling yeast the tied loop is
**directionally** the best param-matched arm on subset accuracy via the clustered-error coherence mechanism
(lower token-acc, higher EM; not a positive-rate artifact). But significance rests on overlapping splits of
single datasets (illustrative, not clean), the loop-specific edge over a joint MLP is small (+0.018 EM,
yeast only), and P1 is partly an under-fitting gap (untied trains to 0.985, +0.7% over budget).* The
bridgehead is real and worth keeping; the **certainty drops one notch** — it is a directional first transfer,
not a clean loop-beats-both.

**[Second-pass review — a THIRD caveat that drops the certainty another notch: the loop's yeast edge is
EM-only, modal-concentrated, and REVERSES under F1.]** A second adversarial pass probed *where* the loop's
yeast EM advantage comes from (trm vs ff trained fresh, split seeds 42 & 45, the M20 config). Two robust,
replicated findings: **(1) the EM edge concentrates on MODAL label-sets** — 94% of the rows trm gets exactly
right but ff misses have a training-frequency ≥20 (a "common" label-combination), vs a 69–70% base rate
among all test rows (both seeds). So the loop's "coherence" win is substantially *"better fit of the dominant
label-combination prior,"* a strategy **subset accuracy (EM) specifically rewards on imbalanced multi-label
data** — exactly the EM-vs-F1 concern. (trm even has *more* total wrong cells/row than ff, 3.59 vs 3.53, yet
higher EM — its correctness is concentrated on whole frequent rows.) **(2) Under micro- AND macro-F1 — the
metrics multi-label work actually reports — `ff_matched` BEATS the loop on BOTH seeds** (micro 0.578/0.589 vs
trm 0.574/0.576; macro 0.421/0.433 vs trm 0.417/0.423). The repo never computed F1; had it, the "loop is the
best param-matched arm" claim would not survive. **Net: the loop's real-data advantage exists ONLY under
subset accuracy and is a modal-combination-fitting effect; on per-label predictive quality (F1) the shallow
joint MLP is preferable.** Leg-1 (joint > per-label on EM, 2/2) still stands as a directional finding, but
the *loop-specific* yeast "win" is now best stated as: *EM-only, concentrated on frequent label-sets, and
reversed under F1* — i.e. metric-dependent, not a robust win. **Action for the next run: report micro/macro-F1
alongside EM** (a `multilabel_f1` metric), and use a held-out/nested-CV test set, before any size sweep.

**[Post-review artifact fix.]** `run.py` now records both baselines for multi-output runs: token-majority
`baseline/accuracy` and constant-row `baseline/exact_match` (the EM floor: emotions 0.153, yeast 0.103).
For `task.name == "multilabel"` WITHOUT K-fold, the paired sign-test fields are intentionally blank because
the random splits overlap and are not independent; the console prints the same warning.

---

### M20 — PROPER EVALUATION (F1 + 10-fold CV) + a 3rd-dataset replication: the definitive verdict. The loop's apparent real-data win does NOT survive honest evaluation; the only robust finding (joint > binary-relevance) is NOT loop-specific. Confirmed on yeast AND scene (opposite coupling regimes).

The two review passes flagged that M20's headline rested on (a) overlapping random splits (anti-conservative
p) and (b) EM only (which over-rewards modal label-combinations). Both are now fixed in the substrate and the
runs re-done: a **micro/macro-F1 metric** (`multilabel_f1`, the standard multi-label metric, reported as a
co-headline) and **10-fold cross-validation** (`n_folds` in `make_multilabel_splits` — the 10 test folds are
DISJOINT and partition the pool, so per-fold Δs are evaluated on independent test sets; the sign test is
re-enabled under CV with the Dietterich-1998 caveat that *training* sets still overlap, so p is indicative).
Configs `m20_multilabel_{emotions_smoke,yeast}.yaml` now run 10-fold CV; 192 tests green, ruff clean,
synthetic tasks byte-identical (F1 gated by `want_f1`, set only for `multilabel`).

**Yeast, 10-fold CV — per-arm and paired Δ (sign test over disjoint folds):**

| arm | EM | micro-F1 | macro-F1 |
|-----|-----|----------|----------|
| untied_stack (4× ceiling) | 0.161 | 0.613 | 0.445 |
| **trm (joint loop)** | **0.146** | 0.583 | 0.434 |
| ff_matched (joint MLP) | 0.126 | **0.587** | 0.430 |
| untied_matched | 0.122 | 0.568 | 0.418 |
| trm_decoupled (binary-relevance) | 0.086 | 0.540 | 0.383 |

| Δ | EM | micro-F1 | macro-F1 |
|---|-----|----------|----------|
| **leg-1: trm − decoupled** | **+0.060 (10/0, p=.002)** | **+0.043 (10/0, p=.002)** | **+0.051 (10/0, p=.002)** |
| leg-2: trm − ff | +0.019 (8/1, p=.04) | **−0.004 (5/5, p=1.0)** | +0.004 (5/5, p=1.0) |
| P1: trm − untied | +0.024 (8/1, p=.04) | +0.015 (7/3, ns) | +0.016 (8/2, ns) |
| trm_decoupled − ff | −0.041 (1/9, p=.02) | −0.047 (0/10, p=.002) | −0.047 (0/10, p=.002) |

**Reading — the honest, final M20 verdict.**
- **The ONLY robust, metric-independent finding is leg-1: joint label modeling beats binary-relevance**
  (trm ≫ trm_decoupled on EM **and** micro-F1 **and** macro-F1, 10/0 each; decoupled is the worst arm on
  every metric, significantly below ff). This survives honest CV + F1. **BUT it is NOT loop-specific:**
  `ff_matched` — a plain joint MLP, no loop — *also* beats decoupled on every metric, so leg-1 is "model
  the labels jointly vs independently," a property the shallow MLP shares. It is a real positive about
  *joint output modeling*, not about recurrence/refinement.
- **Every loop-SPECIFIC edge is EM-only and EVAPORATES under F1.** Δ(trm − ff) is +0.019 EM (8/1, p=.04)
  but **−0.004 micro-F1 / +0.004 macro-F1, both 5/5, p=1.0 — a dead tie** (ff is even fractionally ahead on
  micro-F1). Δ(trm − untied) is likewise EM-only (+0.024, 8/1) and ns on F1. So the M20-headline "the loop
  is the best param-matched arm / loop-beats-both on yeast" was a **subset-accuracy artifact**: confirmed by
  the modal-concentration probe (94% of the loop's extra-correct rows were frequent label-sets), it does not
  reflect better per-label prediction. **Under the metric multi-label work actually uses, the iterative loop
  buys nothing over a shallow joint MLP.**
- **Emotions (10-fold CV) is underpowered and null:** leg-1 is directional only (Δ EM +0.044 but **5/4,
  p=1.0**; F1 8/2, p=.11) — the original "9/0, p=.004" was purely the overlapping-split inflation the review
  predicted. With ~59 test rows/fold there is no power; emotions neither confirms nor refutes.

**Third dataset — `scene` (2407×294, 6 labels, near-mutually-exclusive, card 1.07) — REPLICATES the picture
exactly, on a different coupling regime.** Added as a confirming test (vendored/sha-guarded/tested like the
others; the `scene` duplicate-row quirk forced the no-leakage tests onto the index/multiset level — the
index partition is a clean disjoint cover). 10-fold CV: **leg-1 robust on every metric** — Δ(trm−decoupled)
EM **+0.104**, micro-F1 **+0.049**, macro-F1 **+0.048**, all **10/0, p=.002**, decoupled worst everywhere and
significantly below ff (the M10 signature); **and the loop-specific edge is again EM-only and ties under F1**
— Δ(trm−ff) EM +0.037 (9/1, p=.02) but **micro-F1 +0.001 (5/5, p=1.0) / macro-F1 +0.001 (4/6, ns)**; P1
likewise EM-only. Per-arm F1: trm 0.764 ≈ ff 0.763 (micro), 0.772 ≈ 0.771 (macro) — a dead heat. **This also
RESOLVES the emotions null: scene has 6 labels like emotions but large n, and leg-1 fires at 10/0 — so
emotions' null was SAMPLE SIZE, not label count.** Two large datasets with opposite coupling (yeast 14-label
co-occurrence; scene 6-label mutual-exclusion) now give the identical verdict.

**Capacity probe (hidden=latent=128, 2× width, both datasets) — the M11 "loop edge GROWS with size"
prediction is FALSIFIED on real data; the negative is FINAL.** The one open escape was that the loop-specific
edge might emerge with capacity (synthetic M11: the joint-state edge amplified as models grew). It does not.
At hidden=128, **Δ(trm − ff) on F1 stays a tie/slightly favours ff on BOTH datasets** — yeast micro-F1 −0.009
(3/7), macro −0.002 (4/6); scene micro-F1 −0.008 (3/7), macro −0.010 (2/8) — with ff ahead per-arm (yeast
micro 0.603 > trm 0.594; scene micro 0.769 > trm 0.761). So 2× capacity produces **no** loop-specific F1
signal; the loop only keeps its EM-only edge (yeast +0.017 8/1, scene +0.023 9/1). **Bonus (opposite of
synthetic M11):** leg-1 does NOT grow with size on real data — on yeast it *weakens* at h128 (Δ(trm−decoupled)
micro-F1 −0.007 ns; decoupled catches up to ff, their gap → 0), though it holds on scene (9/1 F1). So neither
the loop-specific edge nor even the joint>per-label edge is capacity-amplified on real tabular — the reverse
of the synthetic scaling story. The hidden=128 probe closes the last lever: **the loop's value does not cross
to real tabular at any capacity tested.**

**Net (supersedes all earlier M20 readings).** Properly evaluated, **M20 is a NEGATIVE for the loop thesis on
real tabular, with one non-loop positive — now confirmed on TWO large datasets with opposite coupling
(yeast 14-label co-occurrence; scene 6-label mutual-exclusion).** What is real: *joint multi-label modeling
beats binary-relevance* (leg-1, robust on EM+F1, 10/0 on both yeast and scene) — but a plain joint MLP gets it
too, so it is not a recurrence result. What is NOT real: any loop-specific advantage — the loop's EM edge over
a joint MLP / an untied stack does not survive F1 or honest CV (Δ(trm−ff) on micro-F1 is a 5/5 tie on *both*
datasets) **and does not emerge at 2× capacity** (hidden=128: still a 5/5-ish F1 tie on both, ff fractionally
ahead — the synthetic M11 "grows-with-size" lever fails here too). So the synthetic §9.2 "tied-recurrence
coherence" finding does **not** transfer to real multi-label tabular as a *loop* property at any capacity
tested; the transferable part is the mundane "joint beats per-label," which needs no loop. This is the §8
discipline landing where it should: the real-data bridge, evaluated honestly, does not credit the loop. (The
synthetic results are untouched and intact; this is specifically about whether the loop's value crosses to
real tabular — under proper eval, it does not.) Canonical tracked summaries (10-fold CV, hidden=64):
`m20_multilabel_{emotions_smoke_20260626T120659,yeast_20260626T123047,scene_20260626T133446}_*`; and
(hidden=128) `m20_multilabel_{yeast,scene}_h128_*` `{curve,deltas,params}.csv`. The earlier random-split runs
(215858/221303) are superseded; their numbers are retained verbatim in the M20 body above for the audit trail.
(NB: the `coherence_excess` rows in every deltas CSV are the M9-flagged **per-arm descriptor** — their
cross-arm Δ is Jensen/dispersion-confounded and is **not** evidence; the verdict rests on EM and F1 only.)

---

## M21 — DONE. Latent / weight INTROSPECTION substrate, run on both anchor regimes. The trained loop does NOT settle a latent fixed point even where it WINS — "dressed-up depth" is now MEASURED, not inferred; the win-vs-fail contrast lives in path-independence, not contraction.

Measurement-only milestone (CLAUDE.md §8): **no new model, no architectural change.** M0–M20
characterized *what* the loop buys behaviourally; the signature pathology ("the loop does not
settle a stable step operator; over-unrolling R'>R decays toward baseline", M1/M3b/M7/M8) had
only ever been read off accuracy curves. M21 builds an introspection layer that instruments the
*dynamics* directly, and runs the identical suite on the loop's WIN regime (`converge` rule 78,
w24 — a genuine multi-output fixed-point target) and its FAIL regime (`iterated` rule 30, w9 — a
non-convergent chaotic CA), so the **converge-vs-iterated contrast is the finding**.

**Literature instruments (mapped to this repo's open questions; CLAUDE.md §12 additions):**
Jacobian spectral radius ρ(∂z_{t+1}/∂z_t) at the over-unrolled state (DEQ Jacobian-regularization —
Bai/Koltun/Kolter 2021 arXiv 2106.14342: ρ<1 ⇒ contractive/extrapolation-friendly, ρ≥1 ⇒ the
over-unroll decay we see; an optional newer looped-LM template, "STARS" arXiv 2605.26733, found via
web search, is supplementary — its 2026 ID is offline-unverifiable, so the argument rests on the
2106.14342 / 2410.23451 / 2211.09961 refs), estimated by power iteration on autograd JVP/VJP; path independence /
asymptotic alignment (Anil 2022 arXiv 2211.09961: same attractor regardless of init ⇒ correlates
with upward generalization); fixed-point residual trajectory + orbits (Geiping 2025
arXiv 2502.05171); effective rank / participation ratio (representational collapse); weight
spectral norms / Lipschitz product (Rethinking Deep Thinking arXiv 2410.23451).

**Substrate (additive, off-by-default → all M0–M20 results bit-identical; touches NO model code).**
`src/looptab/eval/introspection.py`: six diagnostics + a `run_introspection` dispatcher
(recurrent arms get Families A+B+C; controls get B+C). It rides the existing M7
`init_state`/`return_state` resumable-rollout API (bit-identical stepwise composition) and forward
hooks — so it reads the trained model without perturbing any metric. `DiagnosticsConfig`
(`config.py`, off by default) gates a post-training pass in `run_point` that writes a side-car
`*_diagnostics.csv`. Determinism + known-answer tests in `tests/test_introspection.py` (spectral
radius of a diagonal/non-symmetric matrix ≈ |λ_max|; effective rank of an isotropic batch ≈ D, of
a rank-1 batch ≈ 1; same seed → identical numbers). 213 tests, ruff clean. (Implementation note:
the plan sketched adding a `return_trajectory` flag to `TRM.forward`; the existing
`init_state`/`return_state` API already gives bit-identical resumable rollouts, so the layer was
built with **zero model-file edits** — strictly lower-risk, and the off path is byte-identical by
construction.) Configs `m21_introspection_{converge,iterated}.yaml`; tracked summaries
`results/m21_introspection_converge_20260627T083814_*` and
`results/m21_introspection_iterated_20260627T083602_*` (5 seeds each; arms trm / trm_decoupled /
ff_matched / untied_matched, standard `train`, 100 epochs).

**Accuracy anchors (5 seeds — INDICATIVE ONLY, not significance calls; the point of these runs is
the diagnostics, not a new Δ).** At 5 seeds the exact binomial **cannot** reach p<0.05 (§5: 5/5 →
p=0.0625), so these merely confirm the loop behaves on these regimes *as the already-established
legs predict* (leg-1 = M10–M12, P1 = M8c/M9, leg-2 = M15c, all at ≥10 seeds). They are NOT a fresh
demonstration of the legs. Per-seed sign tests below:

| regime | Δ(trm−decoupled) EM (p) | Δ(trm−ff) EM (acc) | Δ(trm−untied) EM (p) |
|---|---|---|---|
| converge (win) | +0.333 (5/5, p=.0625) | +0.025 (acc −0.002, 4/1 ns) | +0.217 (5/5, p=.0625) |
| iterated (fail) | +0.192 (5/5, p=.0625) | +0.030 (acc +0.001, ns) | +0.625 (5/5, p=.0625) |

**The diagnostics — trm, mean over 5 seeds (horizon = 4× trained depth; ρ/σ at power_iter_steps=50).**
NB the ρ *magnitude* is power-iteration on a non-normal Jacobian and is mildly upper-biased at finite
k (one-directional, so **ρ>1 is conservative**); it is robust here — raising k 20→50 moved trm ρ only
2.34→2.38 (converge) / 2.94→2.95 (iterated). Treat ρ magnitudes / cross-regime ordering as approximate;
only the qualitative **ρ>1 / σ≫1 in both** is load-bearing.

| diagnostic | converge (WIN) | iterated (FAIL) | reading |
|---|---|---|---|
| spectral_radius ρ (mean / max) | 2.38 / 2.52 | 2.95 / 4.24 | **ρ>1 in BOTH** — locally expanding map |
| operator_norm σ_max (mean) | 2.66 | 5.68 | σ_max≫1 in both (not a 2-norm contraction) |
| frac_expanding (ρ>1) | **1.00** | **1.00** | every example expanding, both regimes |
| residual ‖Δz‖/‖z‖ @ trained / overunroll | 1.24 / 1.29 | 1.29 / **1.75** | latent NEVER settles (≫0); flat vs **growing** |
| acc @ trained / overunroll | 0.967 / 0.636 | 0.971 / 0.511 | readout is depth-tuned; over-unroll decays |
| EM @ trained / overunroll | 0.581 / 0.012 | 0.818 / 0.003 | whole-row answer **collapses** under over-unroll |
| za_alignment (path independence) | **0.970** | **0.431** | the strongest WIN-vs-FAIL contrast (trm; see caveat) |
| readout_agreement across inits | 0.316 | 0.001 | random inits rarely agree with EACH OTHER (not vs truth) |
| effective_rank (n_dims) | 18.4 (64) | 28.6 (64) | joint latent compresses to a low-D code |
| lipschitz_product | 6.50 | 27.2 | (untied: 98 426 / 49 187; ff: 79 / 233) |

**Reading (per §8 — the honesty clause fires; the pre-registered hypothesis is FALSIFIED, and the
falsification is the result).**

1. **The trained loop does NOT settle a latent fixed point — in EITHER regime, INCLUDING the one
   where it wins.** The pre-registered hypothesis was "contractive on converge (ρ≲1, residual→0),
   non-contractive on iterated." Instead the latent residual ‖z_{t+1}−z_t‖/‖z_t‖ ≈ **1.2–1.3 on
   both** (the latent moves by *more than its own norm* every step — the opposite of convergence),
   ρ>1 with **frac_expanding=1.0 on both**, and over-unrolling collapses the readout everywhere
   (EM 0.58→0.01 converge, 0.82→0.003 iterated). (ρ is measured at the over-unrolled state, which is
   *not* a fixed point — so it is a **local-amplification descriptor consistent with non-settling**,
   not a textbook fixed-point-stability certificate; the non-settling itself is shown directly by the
   residual ~1.2.) So the loop uses its N steps as a **fixed-depth
   feedforward circuit whose readout is tuned to exactly the trained depth**, not as iteration
   toward an attractor. This is **§8's "the loop trivially degenerates into a deep net" turned
   into a measurement** — and it now holds *even on the convergent fixed-point target where the
   loop beats its controls*. The §3 "loops ≈ algorithm steps, extrapolate by unrolling more"
   picture is contradicted at the latent level, not just at the accuracy level (a stronger, more
   mechanistic version of the M1/M8 null).

2. **The strongest WIN-vs-FAIL contrast is PATH INDEPENDENCE, not contraction — but scope it to
   `trm` and to n=5.** `za_alignment` (cosine of the deep-unrolled latent across random z0 inits)
   for the loop is **0.97 ± 0.02 on converge vs 0.43 ± 0.12 on iterated**: on the convergent target
   the map funnels random inits into a **shared, init-independent limit set** (a Geiping-style
   *orbit*, not a fixed point — residual stays ~1.2, so it keeps moving) while on the chaotic CA it
   does not. **Caveat (adversarial-review fix): this is NOT a clean discriminator across arms.** The
   *other* winning arm, `trm_decoupled` (which also beats its controls on leg-1 at converge), has za
   only **0.55 ± 0.29 (converge) vs 0.32 ± 0.16 (iterated)** — bands overlapping, separation weak.
   So path-independence tracks the convergent-vs-chaotic *target* for the joint loop, but it is a
   single n=5 descriptor with no sign test, and it does not cleanly co-vary with *winning*. Two
   weaker contrasts agree for `trm`: over-unroll residual **flat on converge (1.24→1.29) but grows
   on iterated (1.29→1.75)**, and milder over-unroll readout decay on converge (acc drop 0.33 vs
   0.46). So the convergent target leaves a dynamical fingerprint *for the loop*, nothing stronger.

3. **That dynamical fingerprint does NOT rescue the loop — consistent with the win being STATIC
   (joint-state coherence), though M21 does not by itself ESTABLISH that.** Despite za 0.97 on
   converge, `readout_agreement` is only **0.316** (0.001 iterated) — but note what this metric
   measures: whether the *random-init unrolls argmax-agree with EACH OTHER* (introspection.py
   `path_independence`), **not** whether they match the trained-trajectory answer or ground truth.
   So it cannot, on its own, prove "the loop can't decode its own attractor"; high cosine za (0.97)
   with low strict whole-row agreement (0.32, all w cells × 5 inits identical) is also what
   near-decision-boundary readouts give. What M21 *does* show is that the loop is non-settling
   (reading 1) and that its representation is a compressed shared code: the joint loop's final
   latent has **effective rank ~18 of 64 dims**, whereas the per-cell `trm_decoupled` latent has
   eff-rank ~66 of **1848 dims** (converge; ~54 of 621 iterated) — i.e. the joint state is far more
   collapsed relative to its dimensionality. That is *consistent with* the §9.2 "coherent low-D code
   in one shared latent" reading, and with the win being static, **but the causal 'static not
   dynamical' verdict rests on the leg-1/leg-2 Δs of M9–M15c, not on these descriptors** — M21
   corroborates, it does not independently prove it. (The eff-rank cross-arm comparison is itself
   confounded by the 64-vs-1848 dim mismatch, so it is suggestive only.)

4. **Family C — tying yields a far smoother map.** The tied loop's Lipschitz product (~6.5
   converge / 27 iterated) is **three to four orders of magnitude** below the fair untied stack's
   (98 426 / 49 187) and well under the shallow MLP's (79 / 233). Weight tying buys a much more
   Lipschitz-controlled map at fixed budget — a static, weight-level correlate of P1 (the loop
   beats the untied stack on coherence) and an instrument-level reason the untied stack is the
   worst-behaved arm.

**Architectural-refinement implication (the point of the milestone — handed to the next agent,
NOT acted on here).** The diagnostics say the loop is non-contractive (ρ>1, residual non-vanishing)
*by construction of how it was trained* (fixed-depth, final-loss). IF the goal is to make the loop
actually *use* its recurrence (extrapolation / adaptive test-time compute — the capability the repo
has repeatedly found ABSENT), the evidence points to a concrete, measurable lever: train the map to
be contractive — **Jacobian-spectral-radius / Lipschitz regularization** (DEQ Jacobian-reg 2106.14342;
Rethinking Deep Thinking 2410.23451) to push ρ<1, **path-independence-promoting** training (Anil
2211.09961) to raise za→1, and a
**fixed-point / convergent loss** (decode the attractor, not the trained-depth transient) to lift
readout_agreement→1. M21 gives both the trigger (ρ>1, residual~1.2, readout_agreement 0.32 even on
the win regime) and the target metrics. **Crucial honesty caveat (§8):** the loop currently *wins*
(on coherence) **while** non-contractive, so forcing contraction is a bet on a *different*
capability and may **trade away** the static coherence win — exactly the kind of confound this repo
guards. So M21 does **not** mandate a stabilized arm; it characterizes the pathology, supplies the
falsifiable target, and flags the risk. Building a `trm_stable` (spectral/Lipschitz-regularized)
arm and re-running the depth-extrapolation harness is the natural, evidence-gated follow-up — to be
judged against BOTH the extrapolation metric AND the coherence Δ it might cost.

**Caveats.** (i) Standard-train (final-loss, fixed depth) is exactly the regime that *produces* a
non-settling latent; a step-aligned-DS or progressive-loss-trained loop (M3b/M7) could show
different dynamics — untested here, an obvious next probe. (ii) Two regimes / one size / one rule
each; the suite is built to run anywhere (reuse on real-tabular M20 is one line). (iii) The
representation-rank comparison across joint vs decoupled is confounded by differing n_dims (64 vs
1848 converge / 621 iterated), so it is corroborative, not a clean Δ. (v) The accuracy anchors are
5-seed (p≥0.0625; cannot clear §5's significance floor) — they confirm the loop behaves as the
≥10-seed legs (M9–M15c) predict, they are not a fresh demonstration of those legs. (iv) Diagnostics
are per-arm descriptors, not Δs with
sign tests — they generate hypotheses for refinement (§8 "measure before building"), they do not
themselves clear any gate.

---


## M22 — DONE. Airline DISRUPTION-RECOVERY as a synthetic joint multi-output FIXED-POINT task (user-requested, from an ops spec). On a genuinely ff-hard, domain-motivated small-world coupling (aircraft-rotation chains + station-bank cliques), the JOINT-STATE coherence mechanism (leg-1) and TYING (P1) reproduce — but the LOOP HAS NO COHERENCE EDGE OVER A SHALLOW FEEDFORWARD MLP (leg-2 null-to-negative). The spec's success criterion ("does the joint recurrent model produce MORE coherent whole-component recovery states than feedforward?") is answered NO — mirroring the M20 real-data verdict (joint modeling helps, but it is not a *loop* property). NOTE: an adversarial review caught a BLOCKER in the first locked instance (it was near-affine / ff-trivial); this entry is the CORRECTED result on a re-pinned ff-hard instance — see the "M22 adversarial-review correction" block at the end.

**What & why.** A user handed an airline disruption-recovery ops spec and asked to implement it as a SYNTHETIC task and evaluate the TRM arms. The spec's shape IS the §9.2 regime: one row = one *disruption component* (a set of `w` coupled flights at decision time t0); X = per-flight features known at t0; y = the settled binary "severe-outcome" vector over all flights — a JOINT multi-output fixed point where a shallow per-flight model can get *local* predictions right but (the spec's hypothesis) fail *coherent* whole-component exact-match. A new operator FAMILY (small-world coupling: rotation chains [LOCAL, adjacent] + station-bank cliques [NON-LOCAL, scattered]) in the spirit of M13/M14/M15 — NOT the forbidden M19 H/L build (no new model arm) and NOT the §9.4 real-tabular bridge (purely synthetic, integer, network-free). User-confirmed v1: FIXED topology (graph fixed by `task_seed`, like every `converge`-family task; only per-row `severe_0` varies).

**Substrate (`make_disruption`, generators.py; determinism/golden/fixed-point/balance+triviality/trajectory tested).** Two design lessons, the second from the adversarial review: **(1) a faithful MONOTONE cascade is ff-EASY/VACUOUS** — with a fixed graph the fixed point is OR-reachability of `severe_0`, which is **linearly separable**, so an MLP nails it (ff EM 0.77–0.85) and there is no coherence gap. So the operator is a NON-MONOTONE attractor: an integer **THRESHOLD net** (`_threshold_step`) on symmetric `W = w_rot·(rotation-chain adjacency) + w_bank·(bank cliques)` with self-coupling `gamma ≥ -λ_min(W)` ⇒ PSD ⇒ synchronous convergence (no rejection filter, all-integer ⇒ bit-exact). **(2) gamma must be the MINIMAL-PSD value** — the review showed an over-damped gamma (margin 1) pushes the dynamics toward IDENTITY (near-affine; a linear baseline solves it). **Locked (corrected) instance:** w=24, n_banks=4, w_rot=6, w_bank=3, gamma=14 (the MINIMAL-PSD value for task_seed=42, margin 0). Screened genuinely ff-HARD: balanced (cell means 0.47–0.52), copy-severe_0 frac 0.82 / ~4.4 flips per row, **LINEAR-baseline EM ~0.23, ff EM ~0.34** (the §9.2 hard band), shallow relaxation depth (max ~9). **Structural caveat (review-surfaced):** this sparse PSD coupling family is ~92% per-cell LINEAR in every instance — it cannot reach the §9.2 *per-cell-hard* regime (M13 had much lower ff token-acc); its hardness is purely in the EM-COHERENCE dimension (getting ~4 correlated flips all right), which is exactly where the loop's joint-coherence mechanism should help. So this is a clean test of leg-2 in the coherence dimension.

**Run (m22_disruption_base.yaml; w=24, 8 seeds, full arm set mirroring m12; exact sign test; ✓ all arms within ±2% budget).** Per-arm test EM (mean): **ff_matched 0.429 > trm_stepDS 0.371 > trm_nods 0.300 > trm_decoupled_stepDS 0.245 ≫ trm_decoupled_nods 0.078 ≈ untied_matched 0.056.**

- **leg-1 — joint-state (trm vs trm_decoupled) — REPRODUCES, trainability-clean.** Final-loss Δ(trm_nods−trm_decoupled_nods) EM **+0.223 (8/0, p=.008)** is partly inflated by `trm_decoupled_nods` underfitting train (train_acc 0.771, below the ~0.92 copy-severe_0 token baseline — an optimization collapse, the M10 caveat). But the **trainability-clean** Δ(trm_stepDS−trm_decoupled_stepDS), where the decoupled arm trains fine (train_acc 0.916), is **EM +0.127 (8/0, p=.008)** AND acc +0.032 (8/0) — a genuine joint-state coherence edge that SURVIVES the trainability control. So leg-1 transfers to this deep+local family (consistent with M15, where leg-1 transferred to a non-uniform deep+local map). This is the milestone's real positive.

- **leg-2 — loop-beats-the-MLP (trm vs ff_matched) — DOES NOT HOLD (the headline answer).** The loop does NOT beat the shallow feedforward MLP: Δ(trm_nods−ff_matched) **acc −0.025 (0/8, p=.008, significant)** and **EM −0.128 (1/7, p=.070, NS)** — ff is significantly more accurate per-cell and directionally more coherent. So on a genuinely ff-hard, coherence-headroom instance, the iterative loop produces **no more coherent** whole-component recovery states than a param-matched MLP — it produces fewer (ff EM 0.429 vs loop 0.300). The spec's success criterion is answered **NO**. (This is the honest, corrected leg-2: NOT the dramatic −0.263/0/8 the flawed near-affine instance reported.)

- **P1 — tying (trm vs untied_matched) — REPRODUCES, NOW BUDGET-CLEAN.** Δ(trm_nods−untied_matched) EM **+0.244 (8/0, p=.008)**, acc +0.131 (8/0); the stronger coupling widened all arms so `untied_matched` now lands WITHIN ±2% (ratio 1.012, within_tol=True — the breach on the first instance is gone). Caveat: `untied_matched` still underfits train (0.801), so part of the gap is the width-split arm's optimization difficulty, but it is now a budget-clean comparison. (M10 sanity check reproduces: Δ(trm_decoupled_nods−ff_matched) EM −0.351, 0/8 — the per-cell head is far below the MLP.)

**Net verdict.** On a genuinely ff-hard disruption (structured threshold-net) instance, the §9.2 WITHIN-LOOP structure transfers — **joint-state coherence (leg-1, trainability-clean +0.127 EM, 8/0) and tying (P1, +0.244 EM, 8/0, budget-clean)** both reproduce — but the loop has **NO coherence edge over a shallow feedforward MLP (leg-2)**: ff is significantly more accurate and at least as coherent. So the loop's value here is the same as everywhere off the local-uniform-CA regime: it beats its *internal* ablations (per-cell, untied) but not the external param-matched MLP. This is exactly the M20 real-multilabel pattern (joint modeling beats per-label, but a plain joint MLP gets it too — not a *loop* property). The spec's hypothesis — the joint recurrent model produces MORE coherent recovery states than feedforward — is NOT supported on this synthetic instance.

**Caveats / scope.** (i) ONE width (w=24), one locked operator, one size; a w=32 / size sweep is the obvious extension (gamma is w-specific). (ii) This coupling family is ~92% per-cell LINEAR (a structural limit of sparse PSD threshold nets) — its hardness is EM-coherence only, NOT the §9.2 per-cell-hard regime, so it is NOT the same negative as M13/M14 (there leg-1 also failed; here leg-1 transfers). (iii) leg-2 EM is directionally negative but NS at 8 seeds (1/7); the significant leg-2 statement is the token-accuracy one (ff > loop, 0/8). (iv) `trm_decoupled`'s 3-D matmul is the M11 determinism caveat (bit-exact only at num_threads=1, the committed default). (v) Per-row varying topology (the fully faithful spec form) is a gated v2 (expected to hit the flat-MLP unlearnability wall). (vi) All M22 code is additive; every M0–M21 result is bit-identical (226 tests pass, 11 new).

### M22 adversarial-review correction (why this entry was re-run)

The user requested an adversarial review of M22. It found a **[BLOCKER]**: the FIRST locked instance (w_rot=2, w_bank=1, gamma=6) was **near-affine / ff-trivial** — y equalled severe_0 in ~92% of cells, a LINEAR model scored EM 0.55 (beating the loop's 0.39), and the screen's "ff EM ~0.65 = room to win" check mistook "ff is the ceiling" for "the task is hard." Root cause: a sparse non-PSD W forces gamma ≥ -λ_min, and an over-damped gamma (margin 1) drove the dynamics to identity. The first run's dramatic "clean negative" (leg-2 −0.263, 0/8) was therefore an artifact of a degenerate instance. **Fix applied:** re-pinned to a genuinely ff-hard instance (stronger coupling w_rot=6/w_bank=3, MINIMAL-margin gamma=14; verified LINEAR EM 0.23 ≪ ff EM 0.34 with real flips/row), added a LINEAR baseline + copy-frac to the triviality test (`test_disruption_balance_and_nontrivial` now asserts copy_frac<0.88, moved>0.9), and re-ran. The corrected result (above) is materially different and MORE informative: leg-1 now survives the trainability control (the first instance's +0.063 ns becomes +0.127, 8/0), P1 is now budget-clean, and leg-2 is an honest null-to-negative (loop ties/loses ff on coherence) rather than the inflated −0.263. The review's other findings (determinism, seed contract, fairness, statistics all OK; leg-1/P1 trainability framing — sharpened above) are incorporated.

Canonical summary: `m22_disruption_base_20260627T160259_*`.

### M22 size×width robustness sweep (the user-requested "fill out the study") — DONE

To check the single-cell verdict against scale and output width — and specifically to run the **M11
lever** (on the ECA hard-convergence regime, leg-2 [loop-beats-ff] was NEGATIVE at small size and
flipped POSITIVE + grew at large; does the disruption family do the same, or stay null/negative at
every size like M13/M20?) — ran a hidden∈{32,64,128} × w∈{24,32} grid (3 configs
`m22_size_{small,base,large}.yaml`, auto-derived MINIMAL-PSD γ per width = 14 @ w24 / 15 @ w32, both
screened ff-hard, 8 seeds, full arm set, ✓ budget-clean every cell).

**leg-2 (trm_nods − ff_matched) — the decisive axis — STAYS NEGATIVE AT EVERY SIZE AND WIDTH; it does
NOT flip positive with scale:**

| size | w24 acc | w24 EM | w32 acc | w32 EM |
|------|---------|--------|---------|--------|
| h32  | −0.043 (0/8) | −0.200 (1/7) | −0.043 (0/8) | −0.026 (4/4) |
| h64  | −0.023 (0/8) | −0.170 (0/8) | −0.030 (0/8) | −0.053 (3/5) |
| h128 | −0.030 (0/8) | −0.163 (0/8) | −0.019 (0/8) | −0.036 (4/4) |

ff beats the loop on **token-accuracy in all 6 cells (0/8, p=.008)** and on EM at w24 (sig at h64/h128),
tie at w32. Crucially the deficit **does NOT close with capacity** — it plateaus (~−0.02 acc / −0.16 EM
at w24 across h64→h128). This is the **exact OPPOSITE of the ECAs (M11)**, where 2× hidden flipped leg-2
positive and grew it; here more capacity buys the loop nothing over the MLP — **the M11 "grows-with-size"
lever FAILS on the disruption family, reproducing the M20 real-data capacity-probe null.** So the
loop-vs-MLP coherence negative is robust across scale AND width, not a single-cell artifact.

**leg-1 (joint-state) and P1 (tying) REPRODUCE across the grid (the §9.2 within-loop structure transfers):**
- leg-1 nods Δ(trm−decoupled) EM is **8/0 in all 6 cells** (+0.06…+0.23); the trainability-clean
  step-aligned Δ is positive in all 6 (significant 8/0 at h32/h64-w24, h64/h128-w32; marginal 7/1 at
  h128-w24 and 6/2 at h32-w32) — joint refinement beats per-cell refinement throughout, though the
  clean magnitude is modest (~0.05–0.17 EM) and does not systematically grow with size.
- P1 Δ(trm−untied) EM is positive in all 6 cells (8/0 at h32/h64; 7/1, 6/2 at h128), **budget-clean
  at every size** (the width-quantization breach of the first w24 instance is gone). Caveat unchanged:
  untied underfits train, so part of P1 is the width-split arm's optimization difficulty.

**Sweep verdict.** The corrected single-cell result is fully robust across hidden∈{32,64,128} × w∈{24,32}:
**joint-state coherence (leg-1) and tying (P1) transfer to the disruption family, but the loop has NO
coherence edge over a shallow feedforward MLP (leg-2) at any scale or width — and the deficit does not
shrink to zero with capacity (the M11 size lever fails).** The spec's success criterion is answered NO
robustly. The loop beats its INTERNAL ablations (per-cell, untied) but never the external param-matched
MLP — the M20 verdict (joint modeling helps, it is not a *loop* property), now confirmed across the
synthetic size/width grid. Canonical summaries: `m22_size_{small_20260627T193419,base_20260627T203248,
large_20260627T222418}_*`.

**Minor determinism note (to investigate, non-blocking):** the grid w24/h64 cell and the standalone
`m22_disruption_base` run agree qualitatively (leg-2 negative, leg-1/P1 positive) but differ ~0.02 on a
couple of per-arm EMs (e.g. trm_nods EM 0.280 grid vs 0.300 standalone) despite identical instance/seeds
(γ=14 both, pinned vs auto-margin-0). The 2-D arms should be bit-identical; the likely culprit is the
curriculum trajectory-RNG path interacting with the grid-vs-single axis or the `trm_decoupled` 3-D-matmul
thread-sensitivity bleeding into shared state — worth a determinism audit, but it does not affect any
sign-test call (every cell is the same sign with comfortable margins).

## M23 — DONE (positive-control TRIPWIRE on a canonical TRM task: SUDOKU). Synthetic, network-free, deterministic Sudoku built to check whether the repo's `trm` loop reproduces TRM's HEADLINE win (loop ≫ param-matched feedforward) on its HOME regime — constraint-propagation grid reasoning — so the program's broad tabular NEGATIVES can be trusted (a clean win) or flagged as confounded by a weak loop / too-small scale (a null even here). **Two-stage verdict: (1) the MINIMAL smoke is INCONCLUSIVE (EM floored for all arms); (2) SCALED UP, the loop SIGNIFICANTLY beats the param-matched MLP on whole-grid EM (+0.092, 15/16 seeds, p=0.0005) — but it is the repo's COHERENCE edge in the EASY regime, NOT TRM's hard-puzzle-solving win.** So the implementation is sound (the loop genuinely beats the MLP once EM has signal) AND the program's thesis is corroborated (the loop buys whole-grid coherence, not algorithmic depth) — the tabular negatives are trustworthy. The canonical hard-solving win (advantage that GROWS with puzzle difficulty) is NOT reproduced at the CPU scale reachable here (likely needs ACT halting — absent — + far more compute). Smoke detail first, scale-up second.

**What was built (reusable infrastructure, all additive ⇒ M0–M22 bit-identical).**
- `make_sudoku(n, size, n_givens, task_seed, sample_seed, distractors=0)` in `generators.py`: one row = one puzzle. A shuffled valid solution (canonical pattern + validity-preserving band/stack/digit symmetry shuffles, `_sudoku_full_grid`) is dug to EXACTLY `n_givens` clues while a deterministic MRV bitmask backtracking solver (`_count_sudoku_solutions`, the ground-truth algorithm) confirms a UNIQUE solution after each removal (`_dig_sudoku`); grids that can't be dug that far are rejection-redrawn (the `mixed_converge` pattern), loud guard if `n_givens` is unreachable. `X` = one-hot per cell over {blank,1..size} (`size*size*(size+1)` float feats); `y` = solution as `(n, size*size)` classes `0..size-1` — a multi-output FIXED POINT, shape-compatible with converge/disruption so the joint-state / EM / `trm_decoupled` machinery applies unchanged. Box geometry 9→3×3, 6→2×3, 4→2×2. `task_seed` is inert (the function is "solve sudoku") except for `distractors` — the §3 seed split is automatically honoured (rows from `sample_seed`, disjoint train/test).
- **Multi-class support (the one shared-code change):** `run.py` hardcoded `num_classes=2` (the whole M0–M22 suite is binary-per-cell). Now `num_classes = max(2, int(train_ds.y.max())+1)` — exactly 2 for every binary task (so all committed results stay bit-identical), `size` for sudoku. Metrics/loss/models were already generic over the class dim.
- 10 determinism/validity tests (`test_generators.py`, golden-hash pinned `c19954a0…`/`eff4b5f1…`): determinism, valid-completed-grid, puzzle⊂solution, exact-`n_givens`, per-puzzle uniqueness, solver correctness, split-disjointness/task_seed-inertness, distractors, loud guard. Full suite 236 passed (was 226), ruff clean.
- Configs `m23_sudoku_screen.yaml` (size=6, sweep n_givens, trm vs ff, 2 seeds) and `m23_sudoku_base.yaml` (full §4 arm set, paper-faithful loop, 5 seeds).

**Results (6×6, the minimal-smoke scale).**
*Screen (FAIR training: standard `train`, NO EMA, n_sup=1, but the loop keeps RMSNorm + n_latent=6; 2 seeds):*
| n_givens | trm acc | ff acc | Δ(trm−ff) | EM (all arms) |
|---|---|---|---|---|
| 12 (hard) | 0.507 | 0.558 | **−0.051** | 0.000 |
| 16 (med)  | 0.598 | 0.643 | **−0.045** | 0.000 |
| 20 (easy) | 0.681 | 0.725 | **−0.044** | 0.000 |

*Base (paper-faithful loop: detached deep-supervision n_sup=2 + EMA 0.999 + RMSNorm + n_latent=4; 5 seeds; budget-parity ✓ ±5%):*
| n_givens | trm | ff | untied | decoupled | Δ(trm−ff) | Δ(trm−untied) | Δ(trm−decoup) | EM (all) |
|---|---|---|---|---|---|---|---|---|
| 20 | 0.582 | 0.715 | 0.313 | 0.167 | **−0.133** (0/5) | +0.270 | +0.416 | ~0.000 |
| 28 (very easy) | 0.679 | 0.864 | 0.339 | 0.167 | **−0.186** (0/5, EM −0.003) | +0.340 | +0.512 | ~0.000 |

**Reading (honest, with the confound called out).**
1. **EM is FLOORED at ~0 for every arm at all 5 difficulty cells.** ff reaches 0.86 per-cell at the easiest setting, but 0.86^36 ≈ 0.004 ⇒ whole-grid EM ≈ 0; to lift EM you need ~0.99+ per-cell, which no arm approaches at this scale. So the one metric where TRM's iterative win is claimed has **no signal here** — the minimal smoke is below the learnability threshold and cannot function as the positive control as-is.
2. **The loop never beats ff on per-cell accuracy** anywhere; the ff lead actually *widens* as the task eases (0.56→0.73 screen, 0.72→0.86 base). Under fair training the gap is small (−0.04…−0.05); the larger base gap is partly a knob artifact (next point). This is NOT where TRM's advantage is claimed (it is whole-grid EM via iteration), so it is not by itself a verdict against the loop — but it is no evidence *for* it either.
3. **EMA 0.999 actively HURT at this short training** (~1000 updates ≪ the EMA time-constant), folding near-initial weights into the eval: it dropped the loop at n_givens=20 from the screen's 0.681 to 0.582 and **collapsed `trm_decoupled` to exact baseline 0.167** at both difficulties. This reproduces M18's "EMA-alone is catastrophic" — so the base run's leg-1 Δ(trm−decoupled) is NOT a clean coherence result (the ablation arm simply didn't train), and the screen (no EMA) is the fairer per-cell read. Lesson for any scale-up: EMA needs a long-enough run to warm up, or leave it off.
4. **P1 (loop > `untied_matched`) holds, budget-clean (±5%)** (+0.27/+0.34) — but `untied_matched` itself badly underfits multi-class Sudoku (0.31–0.34), so this is the loop beating a weak control, not a strong positive.

**Verdict & next step.** The minimal 6×6 smoke does **not** reproduce TRM's qualitative win, and — because EM is floored for all arms — it does not yet provide a working positive control either way. We have therefore **not** validated that the implementation has any iterative advantage on its home task, which means the program's tabular negatives remain **uncorroborated by a positive control** (neither confirmed-trustworthy nor shown-confounded). The clear next milestone is to **scale the tripwire toward the paper regime until EM lifts off the floor** — bigger model, more refinement/supervision steps (and possibly ACT halting, which the repo lacks), more data, an easy→hard curriculum, and/or 9×9 with a real training budget — then read the loop-vs-ff EM contrast. Only a regime where whole-grid solving is feasible can serve as the tripwire. Canonical artifacts: `m23_sudoku_screen_20260628T085635_*`, `m23_sudoku_base_20260628T110224_*`.

### M23 scale-up (the "scale-up required" follow-through) — DONE: a SIGNIFICANT but coherence-flavored win, NOT the canonical hard-solving win

The smoke floored EM because per-cell accuracy capped ~0.86 ≪ the ~0.99 needed over 36 cells. So the scale-up pushed the three levers that lift per-cell accuracy — **6.4× data (16k), 2× capacity (hidden/latent 128), 2× refinement (n_steps 16)** + per-step deep supervision toward the solution — **EMA OFF** (the smoke showed EMA 0.999 collapses arms before warmup). `trm_decoupled` was dropped from the scaled runs (at hidden=128/n_steps=16 its 3-D matmul is ~12× slower — 951s/8 epochs ⇒ ~2.5h/seed — AND it collapsed to baseline 0.167; the leg-1 attribution needs a cheaper config). `untied_matched` collapsed to a non-learning baseline (acc ~0.33, EM 0) at this width — uninformative, so P1 is not cleanly testable here.

**Stage A — scale-up screen (6×6, 16k data, hidden 128, n_steps 16, 80 ep, 3 seeds; `trm` vs `ff_matched`), difficulty easy→medium:**
| n_givens | trm acc | ff acc | trm EM | ff EM | ΔEM |
|---|---|---|---|---|---|
| 30 (6 blanks, easiest) | 0.984 | 0.981 | **0.571** | 0.509 | **+0.062** |
| 24 | 0.949 | 0.953 | 0.154 | 0.181 | −0.027 |
| 18 (medium) | 0.858 | 0.876 | 0.003 | 0.006 | ~0 (re-floored) |

EM lifts off the floor at the easy end. Crucially the pattern is the **repo's coherence signature, not TRM's**: at n_givens=30 per-cell is a near-tie (Δacc +0.002) but the loop gets **+0.062 more whole grids exactly right** (the M9 leg-2 "EM-at-matched-token-acc" mechanism). And the loop's edge is at the EASY end — at medium/hard difficulty ff is equal-or-better and EM re-floors — the OPPOSITE of TRM's "advantage grows with hardness via iteration."

**Stage B — focused base, 8 seeds, two easy cells (`trm`, `ff_matched`, `untied_matched`):**
- n_givens=30: Δ(trm−ff) EM **+0.078, 7/1, p=0.070** (acc +0.0036, 7/1) — directional, just shy of significance.
- n_givens=28: Δ(trm−ff) EM −0.021, 5/3, p=0.73 (tie). So the edge is narrow (only the very easiest cell).

**Stage C — significance, 16 seeds at n_givens=30 (`trm` vs `ff_matched`; budget-clean ±5%):**
- **Δ(trm−ff) exact_match = +0.0923, 15/16 seeds, p=0.00052** (trm EM 0.611 ± 0.043 vs ff 0.519 ± 0.018).
- Δ(trm−ff) accuracy = +0.0043, 15/16, p=0.00052 (trm 0.986 vs ff 0.982).
- The EM gain (+0.092) is far larger than the per-cell gain (+0.004) implies ⇒ a genuine whole-grid **coherence** component (the loop's errors cluster into fewer whole-grid failures), now SIGNIFICANT on a canonical TRM task.

**Verdict.** The positive control is **MET in a precise, scoped sense**: given enough scale, the `trm` loop **significantly beats a param-matched feedforward MLP on whole-grid Sudoku EM** (+0.092, 15/16, p<0.001) — the first clean loop-beats-ff result on a canonical TRM task in this repo. Two scope limits, both load-bearing: **(1)** the edge is the repo's established whole-grid **COHERENCE** mechanism (leg-2), not a new capability — the EM gain is coherence, not a per-cell-accuracy leap; **(2)** it appears ONLY at the easy, near-saturated end and does **not** grow with difficulty (ff ties-or-wins at medium/hard, EM re-floors below n_givens≈24) — so it is **NOT** TRM's canonical hard-puzzle-solving win, which would need the advantage to widen as propagation deepens. That canonical win is not reproduced at the CPU scale reachable here (the most likely missing pieces: ACT/PonderNet halting — the repo lacks it — and far larger compute/9×9). **Net for the program:** the implementation is sound (the loop is not broken — it genuinely beats the MLP where EM has signal), and the result CORROBORATES the central thesis (the loop's value is whole-grid coherence on fixed-point targets, §9.2, not algorithmic depth/extrapolation) rather than revealing a hidden iterative capability the tabular suite missed. The tabular negatives are therefore trustworthy. Canonical artifacts: `m23_sudoku_scaleup_screen_20260628T*`, `m23_sudoku_scaleup_base_20260628T*`, `m23_sudoku_scaleup_sig_20260628T*`. Open follow-on (compute-gated, NOT done): test whether the advantage GROWS with difficulty once medium puzzles are trained to high per-cell accuracy (bigger model / more data / ACT) — only that would upgrade this from a coherence edge to TRM's hard-solving win; and a cheaper `trm_decoupled` config to attribute the EM edge to the joint state (leg-1) on Sudoku.

### M23 ACT (adaptive computation) — DONE: ACT is faithfully implemented and demonstrably ADAPTIVE, but it does NOT unlock the canonical hard-solving win — so the prior negatives are NOT an artifact of missing ACT

The scale-up win was the coherence edge in the EASY regime; the canonical TRM result is an advantage that GROWS with difficulty via iteration. The one TRM ingredient the repo lacked (§4/§12) was **ACT / adaptive-computation halting**. We built it and ran the decisive test.

**Built (additive, off-by-default ⇒ all M0–M22 bit-identical; 241 tests pass).** A halt head on `TRM` (`use_act=True` → `nn.Linear(latent_dim, 1)`; applied OUTSIDE forward via the M7 `return_state` API, so the forward path and param set are byte-identical when off). `train_act` (`train/loop.py`): segmented detached-carry deep supervision (the autopsy's active ingredient, = `train_deep_supervision`) plus the halt head trained by BCE to predict per-example EXACT-MATCH ("have I solved this row?") — HRM's Q-halt simplified to direct correctness prediction. `evaluate_act`/`act_predict` (`eval/metrics.py`): per-example adaptive inference — each row's answer is frozen at the first segment whose halt prob crosses the threshold; `avg_segments` (mean segments used) is the adaptive-compute diagnostic. Wired into runner/config (`use_act`, `halt_weight`; `n_sup` = max segments). Tests: bit-identity when off, halt-head presence, train_act determinism/guard, adaptive eval bracketing (threshold 0 ⇒ 1 segment, threshold 1 ⇒ max).

**Pre-test (before ACT): more DEPTH alone doesn't crack hard 6×6.** Segmented deep supervision at `n_sup=8` (effective depth 8×8×2 = 128 vs the scale-up's 48) TIES `ff` at hard `n_givens=18` (both acc 0.835, EM≈0). So fixed extra depth is not the lever — motivating the adaptive test.

**ACT sweep (6×6, 16k data, hidden 128; arms `trm_act` [ACT, max 8 seg] vs `trm_seg` [fixed 8 seg, no halt] vs `ff_matched`; 6 seeds, budget-clean ±5%):**
| n_givens | avg_segments | trm_act EM | trm_seg EM | ff EM | Δ(trm_seg−ff) EM | Δ(trm_act−trm_seg) EM |
|---|---|---|---|---|---|---|
| 30 (easy) | **1.00** | 0.487 | 0.543 | 0.470 | **+0.073 (6/0, p=0.031)** | −0.057 (1/5, ns) |
| 24 (med)  | **7.15** | 0.112 | 0.114 | 0.138 | −0.024 (1/5) | −0.003 (3/3, tie) |
| 18 (hard) | **8.00** | 0.003 | 0.002 | 0.004 | −0.002 (hard floor) | +0.001 (3/1, tie) |

**Three findings.**
1. **ACT WORKS — it is genuinely adaptive.** `avg_segments` rises monotonically with difficulty, **1.00 → 7.15 → 8.00**: the halt head solves easy puzzles in a single segment and spends all 8 on hard ones. This is the first real adaptive test-time compute in the repo, and it validates the implementation (the mechanism behaves exactly as TRM/HRM intend).
2. **Adaptivity buys NOTHING on accuracy/EM.** Δ(trm_act − trm_seg) is a tie on EM at every difficulty (and on accuracy `trm_act ≤ trm_seg` everywhere). On EASY it is slightly WORSE (EM −0.057): the halt head halts at segment 1, discarding the refinement that `trm_seg`'s 8 segments use to build coherence (trm_seg EM 0.543 vs trm_act 0.487). Adaptive early-halting sacrifices the very coherence that is the loop's only edge. (A higher inference halt-threshold would recover `trm_seg`'s value by halting less eagerly — but the ceiling is `trm_seg`, which still loses at medium/hard, so no ACT tuning changes the verdict.)
3. **The loop's advantage does NOT grow with difficulty — it SHRINKS and REVERSES, the OPPOSITE of TRM's canonical signature.** Δ(trm_seg − ff) EM: **+0.073 (easy, 6/0, p=0.031)** → −0.024 (med) → −0.002 (hard), and on accuracy ff is significantly ahead at medium/hard (Δacc −0.006/−0.016, 0/6). The fixed-segment loop replicates the scale-up coherence win at EASY (a clean +0.073 EM, 6/0 — consistent with the 16-seed +0.092), but that edge erodes to nothing and flips as puzzles harden.

**Verdict.** We implemented the FULL TRM recipe — segmented detached-carry deep supervision AND ACT halting — and ACT is faithful and demonstrably adaptive (compute scales 1→8 with difficulty). **Yet the canonical hard-solving win does not appear at CPU scale:** the loop's edge over `ff` is the easy-regime COHERENCE effect (leg-2), it does NOT grow with difficulty (it reverses), and adaptive compute buys nothing over fixed depth. **This decisively answers the thread's question: the prior tabular negatives are NOT an artifact of a missing/buggy loop or of missing ACT.** With ACT present, faithful, and working, the loop still shows only the coherence edge, never algorithmic hard-solving — exactly the §9.2 thesis. The remaining gap to the canonical Sudoku-Extreme result is therefore NOT an ingredient we lacked but **raw scale** (Sudoku-Extreme was trained far beyond a 4-core CPU) and, per M21, the deeper fact that this loop does not learn genuine iterative search (it runs fixed-depth pattern-matching, which is why more depth / segments / adaptive compute do not help on puzzles that need search). Stretch runs (9×9, ≥16 segments) were judged not worth the budget: hard 6×6 already shows the loop LOSING to ff, 9×9 EM would be even more floored, and every depth-family lever (fixed depth, segments, ACT) has now come back null on the hard regime. Canonical artifacts: `m23_sudoku_segments_pretest_*` (pre-test), `m23_sudoku_act_sweep_20260702T*` (the sweep). ACT machinery is reusable for any future scaled-up / GPU attempt.

### M23 — ADVERSARIAL REVIEW CORRECTION (the "positive control" is CONFOUNDED — verdict downgraded from "negative confirmed" to "not a fair test")

An adversarial review (websearch against the official TRM code + paper) found the M23 Sudoku positive control does **NOT** clear the loop of implementation confounds, so the strong claim above — "the prior negatives are NOT an implementation/ACT artifact; the gap is raw scale" — is **UNSAFE and withdrawn pending a re-test.** The mechanics are clean (no bugs: halt target detached, no eval leakage, budget matching fair), but the ARCHITECTURE differs from the real TRM on the exact axis that makes Sudoku work:

- **(A, decisive) Our TRM flattens the grid into ONE vector through a structureless MLP; the real TRM keeps per-cell TOKENS and MIXES them (self-attention / MLP-mixer over the cell sequence).** `make_sudoku` reshapes to `(n, size*size*(size+1))` and `TRM.forward` concatenates `[X, z, a]` into a single `(B, in+lat+ans)` vector — no cell/row/col/box structure, no cross-cell operator. The TRM paper's OWN Sudoku ablation credits the mixer (MLP-mixer 74.7% → 87.4% vs attention). Constraint propagation is structured cell-to-cell message-passing; a flat MLP cannot express it. So M23 tested a *flat-MLP recurrent net*, a categorically different model from the one that solves Sudoku-Extreme — the "if the loop can't win here, the impl isn't broken" tripwire is INVALID as run.
- **(B) We do full BPTT through the unroll; TRM/HRM use the 1-step / no-grad-recursion gradient** (run `H_cycles-1` recursions under `torch.no_grad()`, backprop only the last — the DEQ/HRM ingredient). Plausibly why M21 found our loop runs fixed-depth pattern-matching rather than iterating.
- **(C) Effective recursion depth ~48 vs TRM's ~288** (H×L=18 × halt_max_steps=16), compounded by (B).
- (E) The LOG credited the whole gap to "raw scale"; scale is real but CONFOUNDED with A/B/C — "it's just scale" is unproven, and the paper shows the architecture (A), not scale, flips Sudoku accuracy.

**Scope of the correction.** This narrows to the M23 SUDOKU control. The broad M0–M22 tabular/CA negatives are less affected — those targets have little exploitable grid structure for attention/mixing to leverage — so they remain sound within their regime. But M23 no longer licenses "the implementation is validated / the loop's algorithmic-solving absence is confirmed." What M23 DID establish stands: the joint-state COHERENCE edge reproduces on a real fixed-point task (+0.092 EM, 15/16, p<0.001, easy regime), ACT is faithfully built and demonstrably adaptive (avg_segments 1→8), and the generator/ACT machinery are reusable.

**Decisive re-test (to settle it).** Add a per-cell-token encoding + a cross-cell MIXING operator (self-attention or an MLP-mixer over the `size*size` cells) as a `trm`-family arm, ideally also the 1-step gradient, keep weight-tying + param budget matched, and re-run the easy→hard `n_givens` sweep. **Overturns M23 if:** the loop's edge over `ff_matched` GROWS (or survives) as puzzles harden. **Strengthens the negative if:** it still reverses even with cell-mixing. Sources: TRM code github.com/SamsungSAILMontreal/TinyRecursiveModels (mixer/attention block; L207-217 no-grad recursion; arch/trm.yaml H_cycles=3/L_cycles=6/hidden=512/halt_max_steps=16), paper arXiv 2510.04871 (§Sudoku mixer ablation).
