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
