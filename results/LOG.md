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

## Infra — Training/eval performance (no scientific change). Bit-identical, ~2.5× faster.

Not a milestone — a perf pass on the model/training/eval path. **All run outputs are byte-for-byte
unchanged** (verified: parity single-output and iterated multi-output cells reproduce prior
accuracies and exact-match exactly; 67/67 tests pass; ruff clean).

Three bottlenecks resolved:

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

**Still on the table (not done):** seeds/grid-cells are embarrassingly parallel (each is a pure
function of its seed, already reseeded independently), so running them across processes — one torch
thread each — would give near-linear speedup on the full suite. Deferred: it's a runner-level change
(process pool + result aggregation) with more surface area than the wins above, so it should be a
deliberate decision, not folded into this pass.
