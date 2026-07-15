"""Generate the 6 M32 configs (etth1/weather x {192,336,720}) — the ingredient decomposition.

7 arms: the 4 M31 arms {trm_mixer, trm_mixer_nomix, trm_flat, ff_matched} at their LOCKED M31/M30
widths (so the M31/M30 headline Δ re-derives as a cross-check) + the 3 new controls that split M31's
Δ(nomix−ff) "shared-readout / channel-independent parameterization" into its three bundled
ingredients, each a clean single-flag flip from nomix, budget-matched to trm_flat (widths from
probe_budget.py, hidden=latent=w):

  trm_mixer_unsharedro        mix ON,  UNSHARED readout   → readout isolation on the CD body
  trm_mixer_nomix_unsharedro  mix OFF, UNSHARED readout   → readout isolation on the CI body (clean)
  trm_mixer_nomix_distinctw   mix OFF, DISTINCT weights   → per-variable weight-sharing isolation
"""
from pathlib import Path

LOOKBACK = 96
DS = {
    "etth1": dict(vars=7, latent=96,
                  mixer_h={192: 424, 336: 512, 720: 624},
                  nomix_h={192: 428, 336: 516, 720: 624},
                  unsh_w={192: 118, 336: 116, 720: 114},
                  distinctw_w={192: 70, 336: 78, 720: 90}),
    "weather": dict(vars=21, latent=192,
                    mixer_h={192: 912, 336: 1200, 720: 1592},
                    nomix_h={192: 916, 336: 1196, 720: 1620},
                    unsh_w={192: 142, 336: 134, 720: 128},
                    distinctw_w={192: 70, 336: 82, 720: 96}),
}
OUT = Path("configs/experiments")

TEMPLATE = """# M32 — decompose M31's shared-readout attribution on `{ds}` (H={H}). One axis vs M31: it ADDS three
# controls that split M31's Δ(trm_mixer_nomix − ff_matched) — the "shared-readout / channel-independent
# parameterization" win — into its THREE bundled ingredients, each a clean single-flag flip from nomix,
# all budget-matched to trm_flat (widths from scratchpad/m32/probe_budget.py). NEGATIVE Δ favours the
# first arm (lower forecast MSE is better).
#   Δ(trm_mixer_nomix − trm_mixer_nomix_unsharedro) = the SHARED READOUT (CI body, clean flip)
#   Δ(trm_mixer      − trm_mixer_unsharedro)         = the shared readout (CD body, cross-check)
#   Δ(trm_mixer      − trm_mixer_nomix)              = channel-independence / mixing (= M31 mixing leg)
#   Δ(trm_mixer_nomix − trm_mixer_nomix_distinctw)   = per-variable WEIGHT-SHARING (CI+shared-RO held)
#   Δ(trm_mixer_nomix − ff_matched)                  = M31 headline (reproduced) = the sum to decompose
# The four M31 arms keep their EXACT M30/M31 widths so those Δ reproduce M31 to within rounding.

task:
  name: {ds}
  objective: regression
  params:
    dataset: {ds}
    lookback: {lookback}
    horizon: {H}
    n_folds: 10                 # seeds 0..9 = the 10 DISJOINT chronological backtest blocks
    test_frac: 0.3
  n_train: 6000
  n_test: 1000
  task_seed: 0

arms:
  - name: trm_mixer
    label: trm_mixer            # cross-VARIABLE mixing loop (M30/M31 width, matches trm_flat)
    hidden_dim: {mixer_h}
    latent_dim: {latent}
    token_hidden: 8
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm_mixer_nomix
    label: trm_mixer_nomix      # M31 control: SHARED readout, token-mixing REMOVED (M31 width)
    hidden_dim: {nomix_h}
    latent_dim: {latent}
    token_hidden: 8
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm_mixer_unsharedro
    label: trm_mixer_unsharedro  # M32: mix ON, UNSHARED per-cell readout (readout isolation, CD body)
    hidden_dim: {unsh_w}
    latent_dim: {unsh_w}
    token_hidden: 8
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm_mixer_nomix_unsharedro
    label: trm_mixer_nomix_unsharedro  # M32: mix OFF, UNSHARED readout (readout isolation, CI body)
    hidden_dim: {unsh_w}
    latent_dim: {unsh_w}
    token_hidden: 8
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm_mixer_nomix_distinctw
    label: trm_mixer_nomix_distinctw   # M32: mix OFF, per-cell DISTINCT channel weights (weight-share)
    hidden_dim: {distinctw_w}
    latent_dim: {distinctw_w}
    token_hidden: 8
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm
    label: trm_flat             # flat loop (channel-dependent, UNSHARED M*H readout) — budget ref
    hidden_dim: 64
    latent_dim: 64
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: ff_matched
    label: ff_matched           # §4a param-matched shallow joint MLP (UNSHARED M*H readout) — anchor
    hidden_dim: 64
    latent_dim: 64
    n_steps: 8

deltas:
  # ingredient i — shared readout (clean single-flag flip; NEG Δ favours the shared-readout arm)
  - [trm_mixer_nomix, trm_mixer_nomix_unsharedro]        # CI body (the clean estimate)
  - [trm_mixer, trm_mixer_unsharedro]                    # CD body (cross-check)
  # ingredient ii — channel-independence / mixing (readout held)
  - [trm_mixer, trm_mixer_nomix]                         # shared-RO (= M31 mixing leg)
  - [trm_mixer_unsharedro, trm_mixer_nomix_unsharedro]   # unshared-RO cross-check
  # ingredient iii — per-variable weight-sharing (CI + shared-RO held)
  - [trm_mixer_nomix, trm_mixer_nomix_distinctw]
  # anchors: reproduce the M31 headline + expose the residual the ingredients must explain
  - [trm_mixer_nomix, ff_matched]                        # M31 headline Δ(nomix−ff)
  - [trm_mixer_nomix_unsharedro, ff_matched]             # per-cell CI vs joint MLP at unshared RO
  - [trm_mixer, ff_matched]                              # M30 headline (context)
  - [trm_flat, ff_matched]                               # context

budget_reference: trm_flat
budget_tol: 0.05

train:
  epochs: 25
  lr: 1e-3
  weight_decay: 1e-4
  batch_size: 128
  device: cpu

seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
parallel_workers: 4
results_dir: results
"""

for ds, spec in DS.items():
    for H in (192, 336, 720):
        text = TEMPLATE.format(
            ds=ds, H=H, lookback=LOOKBACK, latent=spec["latent"],
            mixer_h=spec["mixer_h"][H], nomix_h=spec["nomix_h"][H],
            unsh_w=spec["unsh_w"][H], distinctw_w=spec["distinctw_w"][H],
        )
        path = OUT / f"m32_{ds}_h{H}.yaml"
        path.write_text(text)
        print("wrote", path)
