# M3b — DONE. Step-aligned DS + T-curriculum (Task B). Layered result: DS is mis-specified, not inert; but no transferable operator.

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
