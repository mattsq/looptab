# Milestone log (index)

Progressive-disclosure entry point. `CLAUDE.md` holds current guidance; each `results/log/*.md` file preserves the full narrative evidence for one milestone.

## Update rule

When adding a milestone, create `results/log/<milestone>.md` with the full evidence and add one concise row here. Keep this index skimmable.

## Milestones

| Milestone | Full narrative | One-line verdict |
|---|---|---|
| M0 | [`m0.md`](log/m0.md) | DONE. Harness + parity (Task A). |
| M1 | [`m1.md`](log/m1.md) | DONE. Task B (iterated CA) + depth-extrapolation harness. |
| M2 | [`m2.md`](log/m2.md) | DONE. Untied-stack control (§4b), two forms, Task A + Task B. |
| M2-confirm | [`m2-confirm.md`](log/m2-confirm.md) | DONE. Replicate the Task B tying result across a rule × width grid. |
| M3a | [`m3a.md`](log/m3a.md) | DONE. Depth-at-fixed-budget sweep (Task B). Prediction FALSIFIED. |
| M3b | [`m3b.md`](log/m3b.md) | DONE. Step-aligned DS + T-curriculum (Task B). Layered result: DS is mis-specified, not inert; but no transferable op… |
| M4 | [`m4.md`](log/m4.md) | DONE. Replicate & stress-test the Task A parity leg across d × k. Original result REPLICATES; "depth helps, tying neu… |
| M5 | [`m5.md`](log/m5.md) | DONE. Lift the M4 sample wall (Task A parity, larger n_train). Wall is SAMPLE-bound and lifts to all-solve with no se… |
| M6a | [`m6a.md`](log/m6a.md) | DONE. The both-axes probe (multi_parity). §9 gate is empirically UNSATISFIABLE by the generalist; loop is depth-posit… |
| M7 | [`m7.md`](log/m7.md) | DONE. Progressive-loss training (Deep Thinking) vs the depth-extrapolation null (Task B). New mechanism, clean NEGATI… |
| M8 / M8b | [`m8___m8b.md`](log/m8___m8b.md) | DONE. Variable-compute FIXED-POINT task (converging CA). Adaptive compute FAILS (falsifies the M7 hypothesis); but th… |
| M9 | [`m9.md`](log/m9.md) | DONE. Width sweep + coherence-mechanism diagnostic (converge). The M8 tying-positive STRENGTHENS (loop-beats-both is… |
| M10 | [`m10.md`](log/m10.md) | DONE. Decoupled-head ablation (converge). The WHOLE-ROW-COHERENCE mechanism is ISOLATED: the JOINT multi-output state… |
| M11 | [`m11.md`](log/m11.md) | DONE. Generalize the coherence result across MODEL SIZE and OPERATOR FAMILY. Layered verdict: the joint-state mechani… |
| M12 | [`m12.md`](log/m12.md) | DONE. Confirm the "hard-convergence" boundary. The joint-state coherence mechanism reproduces on ALL 5 untested orbit… |
| Infra | [`infra.md`](log/infra.md) | Training/eval performance (no scientific change). Bit-identical, ~2.5× faster. |
| M13 | [`m13.md`](log/m13.md) | DONE. Leave the ECA family (threshold/Hopfield attractor net). The joint-state coherence result is CA/local-update-sp… |
| M14 | [`m14.md`](log/m14.md) | DONE. The locality probe (local-but-non-CA threshold net). M13's locality hypothesis FALSIFIED: locality makes the ta… |
| M15 | [`m15.md`](log/m15.md) | DONE. Separate the M14 confound (uniform rule vs deep convergence). RESULT: a clean DECOMPOSITION — the joint-state m… |
| M15b | [`m15b.md`](log/m15b.md) | DONE. Leg 2 NAILED: a depth/hardness-controlled uniform control confirms loop-beats-the-MLP needs the uniform (transl… |
| M15c | [`m15c.md`](log/m15c.md) | DONE. Close the leg-2 central-depth residual: a depth-DISTRIBUTION-matched uniform control. Leg 2 SURVIVES depth-cont… |
| M16 | [`m16.md`](log/m16.md) | DONE. Reframe the project: retire the unsatisfiable §9 gate; re-imagine Task C around the mechanism we found. (Writin… |
| M17 | [`m17.md`](log/m17.md) | [VERDICT SUPERSEDED BY M18g — gate UNMET, M19 NOT earned; banner below] Build the §9.3 Task C substrate (`make_nested… |
| M18 | [`m18.md`](log/m18.md) | TRM-faithful ingredients + THREE adversarial reviews (M18b converge / M18c ablation / M18a wall / M18d nested / M18e+… |
| M20 | [`m20.md`](log/m20.md) | DONE (headline SOFTENED by adversarial review — see the correction block at the end of this entry). The §9.4 real-tab… |
| M21 | [`m21.md`](log/m21.md) | DONE. Latent / weight INTROSPECTION substrate, run on both anchor regimes. The trained loop does NOT settle a latent… |
| M22 | [`m22.md`](log/m22.md) | DONE. Airline DISRUPTION-RECOVERY as a synthetic joint multi-output FIXED-POINT task (user-requested, from an ops spe… |
| M23 | [`m23.md`](log/m23.md) | DONE (positive-control TRIPWIRE on a canonical TRM task: SUDOKU). Synthetic, network-free, deterministic Sudoku built… |
| M24 | [`m24.md`](log/m24.md) | DONE. The M23-MIXER re-test carried to the ORIGINAL ring tasks (`converge` + `iterated`). The verdict SPLITS along th… |
| M24f | [`m24f.md`](log/m24f.md) | DONE (headline stands post-adversarial-review; one mechanism claim reworded off EM). The M24 cross-cell MIXER re-test… |
| M25 | [`m25.md`](log/m25.md) | DONE. Mixer re-test on REAL multi-label (`yeast`/`scene`/`emotions`). Clean NEGATIVE: mixer TIES the shallow MLP on micro-F1/accuracy — the naive feature→label reshape lacks a shared input/output cell topology. |
| M26 | [`m26.md`](log/m26.md) | DONE. Mixer on REAL multivariate forecasting (`etth1`/`weather`). Clean POSITIVE: cross-variable mixer beats the MLP on MSE (edge grows with #vars) — forecasting supplies the shared topology M25 lacked. First regression path. |
| M27 | [`m27.md`](log/m27.md) | DONE. The `trm_stable` contractive arm (M21 lever). Constructive NULL, architecture-independent: contraction is achievable + free-when-solved but buys no accuracy / test-time compute / OOD crack. |
| M28 | [`m28.md`](log/m28.md) | DONE. Mixer re-test on the last two cross-cell tasks. `mixed_converge` = mixer WIN; `nested_converge` (Task C) = informative EXCEPTION — mixer improves but shares a data-bound ceiling, so Task C stays CLOSED. |
| M29 | [`m29.md`](log/m29.md) | DONE (FINAL after two adversarial reviews; the "m29c reversal" is RETRACTED). DS MECHANISM re-tested on the mixer: M18 HOLDS — the detached-carry mechanism is inert; the N_sup gain is just more compute. |
| M30 | [`m30.md`](log/m30.md) | DONE (adversarial-review-hardened). M26 mixer-forecasting positive carried across the HORIZON axis {192,336,720} on `etth1`/`weather`. The win is horizon-ROBUST (grows on weather — direction-robust by median, but the −0.99 h720 mean is ~2× inflated by one hard block; etth1 non-monotonic). Robust driver = GENERALIZATION (MLP overfits past persistence on 10/10 weather-h720 blocks). Primarily ARCHITECTURE not recurrence (untied mixer carries it; tying is a swamped tie, not a reversal) — but CONFOUNDED with shared-readout efficiency (no arm separates mixing from readout-sharing → new control needed). CD>CI persists in direction but loses M26 significance; the one CI "divergence" is a training artifact, not demonstrated instability. Budget-clean via per-horizon mixer re-widening. |
| M31 | [`m31.md`](log/m31.md) | DONE (adversarial-review-hardened). Ran M30's named shared-readout control (`trm_mixer_nomix` = shared-readout, token-mixing REMOVED, budget-matched). RESOLVES the M30 confound: the forecasting mixer win is the SHARED READOUT / channel-independent parameterization, NOT cross-variable MIXING (the FIRM claim is the negative one). At a held shared readout, mixing adds NOTHING — mean-positive Δ(mixer−nomix) +0.03→+0.16, sign-sig *against* mixing on all 3 weather cells but NULL on etth1 (8/2, 5/5, 8/2); "worsens with horizon" is a mean artifact (non-monotone by median). The whole M30 headline is Δ(nomix−ff) (all 0/10–1/9, p≤.021; weather median amplification −0.24→−0.43→−0.65, the −1.13 h720 mean ~1.7× inflated by one ff outlier block). `nomix` (channel-INDEPENDENT) is the BEST arm in all 6 cells by mean — mild evidence FOR the DLinear/PatchTST CI direction. Caveats: mixing leg matched to trm_flat not trm_mixer (≤1.6% width gap); "shared readout" bundles readout+CI+weight-sharing; M30 reproduced to rounding (trm_mixer weather-h720 drifts 2.6e-3). Corrects §11.2 #7–8/#10 attribution (forecasting only; synthetic Sudoku/`converge` mixing wins untouched). |
