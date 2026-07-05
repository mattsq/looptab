# M3a — DONE. Depth-at-fixed-budget sweep (Task B). Prediction FALSIFIED.

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
