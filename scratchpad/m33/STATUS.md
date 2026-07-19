# M33 status — COMPLETE + adversarial-review-hardened (2026-07-19).
#
# Adversarial subagent review found: (1) MAJOR disruption_w32 gamma bug (14 vs baseline 15) — FIXED,
# re-run at gamma=15, now reproduces M24f w32 exactly (Δ(mixer−ff) acc +0.034 / EM +0.478); stale
# gamma=14 CSVs archived in scratchpad/m33/stale_gamma14_w32/. (2) MAJOR overclaim: readout/weight-share
# "0.000" is DEGENERATE SATURATION (3 CI arms converge to identical per-cell decisions), not a measured
# null — write-up softened to "untestable here." (3) CI-harmful leg re-flagged as BUNDLED (nomix−ff),
# not a clean CI isolation. (4) additive identity deflated to arithmetic tautology. (5) fixed mixing-leg
# range +0.71→+0.51 (hopfield w32 = +0.507), ±1.5%→±5%, cell-count 9→12. Verdict was OVERCLAIMED not
# wrong; the FIRM result (mixing is the clean single-flag win, +0.51…+0.99 EM 8/0, flips sign vs
# forecasting) stands. All corrections propagated to m33.md + CLAUDE.md §11.2 #15 + LOG.md row.
#
# --- earlier completion note ---
#
# M33 status — COMPLETE (2026-07-18). All 10 configs done; write-up landed.
#
# results/log/m33.md, results/LOG.md row, and CLAUDE.md §11.2 #15 + §11.4 closed-lever line all written.
# Mixer flag tests pass (22/22). Result: the M31/M32 decomposition FLIPS on synthetic constraint-coupled
# tasks — token-MIXING is ~the entire win (Δ(mixer−nomix) EM +0.71…+0.99, 8/0), channel-INDEPENDENCE is
# HARMFUL (Δ(nomix−ff) negative, 0/8), shared readout + weight-share exactly 0.000 (incl. Sudoku nc=6).
# #8's dividing line is now measured, not asserted. Nothing left to run.
#
# --- original in-flight note kept below for history ---
#
# M33 status — shared-readout / channel-independence decomposition on the synthetic mixer-win tasks

**What M33 is:** port the M31/M32 forecasting decomposition (token-MIXING vs channel-INDEPENDENCE vs
shared-READOUT vs WEIGHT-SHARING, via off-by-default `TRMMixer` control arms) to the five SYNTHETIC
tasks where `trm_mixer` genuinely wins. Tests whether CLAUDE.md §11.2 #8's *asserted* dividing line
("mixing helps only where outputs are genuinely constraint-coupled") holds when *measured*. No code
changes — all arms already exist. Pure config + run + write-up. Approved plan:
`/Users/mattsimmons/.claude/plans/we-recently-introduced-some-compressed-haven.md`.

## Where we stopped (2026-07-16, manual shutdown mid-run)

**DONE (6/10 configs, results in `results/m33_*_deltas.csv`):**
converge_w24, converge_w32, converge_w48, hopfield_w24, hopfield_w32, mixed_converge_w24.

**TODO (4 configs):** mixed_converge_w32, disruption_w24, disruption_w32, sudoku.
(mixed_converge_w32 was interrupted mid-run; run.py writes CSVs only on completion, so it left no
output and re-runs from scratch.)

All background processes were killed. Runtime ≈ 1.5–2.5 h/config on CPU (sudoku cheaper: 15 ep / 6 seeds).

## To resume

```bash
bash scratchpad/m33/run_remaining.sh        # idempotent: skips the 6 done, runs the 4 remaining
# or, to background it:  bash scratchpad/m33/run_remaining.sh > scratchpad/m33/run_remaining.log 2>&1 &
```

Then do the write-up (task #5): `results/log/m33.md` (per-task decomposition tables), one row in
`results/LOG.md`, and a CLAUDE.md §11.2 conclusion (#15) + §11.4 closed-lever line.

## Result so far (unanimous across the 6 done configs — a clean MIRROR of forecasting)

Every headline reproduces its shipped baseline exactly (converge w24/32/48 = +0.345/+0.676/+0.944 EM
vs M24 +0.345/+0.676/+0.942; hopfield w24 = +0.454 vs M24c +0.454) — the fork did not perturb the
baseline arms. The decomposition (EM, POSITIVE Δ favours first arm — classification, OPPOSITE of the
forecasting MSE sign):

| config          | MIXING Δ(mixer−nomix) | CI/readout Δ(nomix−ff) | headline Δ(mixer−ff) |
|-----------------|-----------------------|------------------------|----------------------|
| converge_w24    | **+0.988** (8/0)      | −0.643 (0/8)           | +0.345               |
| converge_w32    | **+0.976** (8/0)      | −0.300 (0/8)           | +0.676               |
| converge_w48    | **+0.966** (8/0)      | −0.022 (0/8)           | +0.944               |
| hopfield_w24    | **+0.706** (8/0)      | −0.251 (0/8)           | +0.454               |
| mixed_conv_w24  | (see deltas csv)      | (see deltas csv)       | (reproduce M28a)     |

- **Token-mixing IS the mechanism** on these coupled tasks (mixing leg large positive, 8/0) — the exact
  opposite of forecasting, where mixing was net harmful.
- **The channel-independent parameterization is WORSE than a plain MLP here** (CI leg negative, 0/8) —
  it is exactly the thing that *won* forecasting, and it loses on constraint-coupled synthetic tasks.
- **Shared readout and weight-sharing are ~0 exactly** on the binary tasks (num_classes=2 ⇒ tiny
  readout): Δ(nomix − nomix_unsharedro) = 0.000, Δ(nomix − distinctw) = 0.000 — confirming they were
  forecasting parameter-efficiency artifacts. Sudoku (num_classes=6, TODO) is the one place the readout
  is non-trivial — worth watching whether it stays ~0.
- Nuance on converge: as w grows the mixing leg stays ~1.0 while the CI penalty shrinks toward 0
  (−0.64→−0.30→−0.02) because `ff` itself collapses on the wider ring — mixing carries the whole win at
  every width regardless.

**Headline for the write-up:** #8's dividing line is now MEASURED, not asserted. The mixer decomposition
*flips sign* between regimes — forecasting is channel-independent (mixing harmful, CI wins), synthetic
constraint-coupled is mixing-led (mixing is ~the entire win, CI is harmful). Same architecture, opposite
attribution, set by whether the outputs are genuinely cross-cell coupled.

## Budget-audit note (expected, documented)
`trm_mixer_nomix_distinctw` breaches ±5% (ratio ~0.905, CONSERVATIVE/under-budget) on the w24 binary
cells because per-cell distinct weights make the param count jump coarsely — but Δ(nomix − distinctw)=0
there anyway, so the breach is moot. `trm_mixer` at converge_w48 is 0.866 (the inherited baseline
conservative breach, unchanged from the shipped M24 config). All other matched arms within ±1.5%.
