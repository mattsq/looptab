# M30 — forecasting horizon sweep: ✅ COMPLETE (2026-07-13 04:58). All 6 configs done + budget-clean.
# Write-up landed: results/log/m30.md, LOG.md index row, CLAUDE.md §11.2 #10 + §11.3.
# Verdict: mixer win is horizon-robust and AMPLIFIES; it's the ARCHITECTURE not recurrence
# (sharpens at long H); CD>CI does NOT reverse at this scale (channel-indep destabilizes).
# (history below kept for provenance)
#
# ---- ORIGINAL IN-PROGRESS STATUS (paused 2026-07-10 ~22:40) ----

**Goal (approved plan).** Carry the M26 mixer-forecasting positive across the HORIZON axis
(96→{192,336,720}) on `etth1` (7 var) + `weather` (21 var), budget-clean at each horizon.
Two headline reads (MSE; NEGATIVE Δ favours the first arm = lower MSE = better):
- `Δ(trm_mixer − ff_matched)` — does the mixer's MSE win survive long horizons?
- `Δ(trm_flat − trm_decoupled)` — does CD>CI (cross-variable coupling) survive, or does
  channel-independence take over (DLinear/PatchTST prediction)?

## Config structure (why 6 configs, not a grid)
A grid over horizon CANNOT hold budget parity: `trm_flat`/`ff` use a full unshared M×H readout
that balloons with H, while `trm_mixer`'s shared per-cell readout does not — so at h720 the mixer
drifted to 0.42× of the reference. Fix (approved): per-(dataset,horizon) configs with `trm_mixer`
RE-WIDENED per horizon to re-match `trm_flat`'s budget. Widths found by a param probe
(`get_model` builder), all ratios 0.98–1.00:

| dataset | H192 | H336 | H720 | latent |
|---|---|---|---|---|
| etth1 | hidden 424 | 512 | 624 | 96 |
| weather | hidden 912 | 1200 | 1592 | 192 |

Configs: `configs/experiments/m30_{etth1,weather}_h{192,336,720}.yaml` (all 6 committed-to-disk).

## Progress: 5 of 6 configs DONE, all budget-clean ✓  (updated 2026-07-12 16:50)
DONE: etth1_h192, etth1_h336, etth1_h720, weather_h192, **weather_h336**.
REMAINING: **weather_h720 ONLY** (running now, heaviest: hidden-1592 mixer, horizon 720; ~4–6h).
NOTE: overnight the Mac idle-slept and suspended the workers (~18h lost). FIXED — the run is now
held awake by `caffeinate -ims -w <sweep_pid>`, and the resume script re-execs itself under
caffeinate (keep laptop on AC; `-s` only holds on AC).

Raw results in `results/m30_*_{deltas,params,curve}.csv` + `.json` (persist on disk).
Run-1 log: `scratchpad/m30/m30_sweep_run1.log`.

## Results so far (10 seeds, budget-clean; MSE Δ, negative = first arm better)

| dataset | H | Δ(mixer−ff) | Δ(mixer−flat) | Δ(untied_mix−ff) | Δ(flat−decoupled) CD>CI |
|---|---|---|---|---|---|
| (M26 ref, h24) | 24 | −0.089 (9/1) | −0.097 (9/1) | −0.087 (9/1) | −0.035 (8/2, ns) |
| etth1 | 192 | −0.152 (0/10, p=.002) | −0.095 (1/9, p=.021) | −0.142 (0/10, p=.002) | −0.115 (2/8, ns) |
| etth1 | 336 | −0.115 (2/8, ns) | −0.031 (3/7, ns) | −0.090 (2/8, ns) | −0.042 (4/6, ns) |
| etth1 | 720 | −0.184 (3/7, ns) | −0.009 (2/8, ns) | −0.257 (0/10, p=.002) | −0.103 (3/7, ns) |
| weather | 192 | −0.343 (1/9, p=.021) | −0.092 (2/8, ns) | −0.277 (0/10, p=.002) | −0.112 (2/8, ns) |

## Preliminary reading (to be finalized after weather_h336/h720)
1. **Mixer beats the MLP at every horizon in the MEAN, and the magnitude does NOT decay** — if
   anything it grows (etth1 −0.15→−0.18; weather −0.34 at h192 ≫ etth1). The M26 win is
   horizon-robust in size.
2. **But per-block sign-consistency erodes at long horizon on `trm_mixer`** (etth1 0/10 → 3/7 as
   std grows 0.14→0.29) — longer horizons = noisier backtest blocks. So the *tied loop's* headline
   is significant at h192, mean-only at h336/h720.
3. **The mixing ARCHITECTURE (non-recurrent untied mixer) is the ROBUST winner**: Δ(untied_mixer_
   matched − ff) stays 0/10 p=.002 even at etth1 h720 where the tied mixer is noisy (3/7). Sharpens
   M26's "it's the architecture, not recurrence" — the tying adds noise, not signal, at long H.
4. **CD>CI stays weak/ns on etth1 at all horizons** (as at h24) — expected; the coupling effect
   only showed on 21-var weather in M26. weather_h336/h720 are the cells that test whether CD>CI
   survives or reverses at long horizon (the DLinear/PatchTST question). NOT YET ANSWERED.

## HOW TO RESUME
```
cd /Users/mattsimmons/looptab
./scratchpad/m30/run_m30_sweep.sh          # resumes: skips the 4 done, runs weather_h336 + h720
```
The script skips any config with an existing `results/m30_<cfg>_*_deltas.csv`. `FORCE=1` re-runs all.

## AFTER the sweep completes (write-up TODO)
- Verify weather_h336/h720 budget audits clean (their `_params.csv`, ratio ~1.0 vs trm_flat).
- Finalize `results/log/m30.md` (horizon×dataset Δ table + verdict) + one `results/LOG.md` index row.
- Update CLAUDE.md §11.2 finding #10 and §11.3 "Open work" (mark horizon sweep done; state whether
  the mixer win / CD>CI is horizon-robust). Leave "more datasets" + "benchmark-scale" as open.
- Carry M26's caveats: sign-test p INDICATIVE (nested train sets, Dietterich 1998); tiny CPU models.

Branch: `claude/m30-forecast-horizon-sweep`. Nothing committed yet (results are raw/gitignored).
