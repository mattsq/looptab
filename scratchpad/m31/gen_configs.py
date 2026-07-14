"""Generate the 6 M31 lean configs (etth1/weather x {192,336,720}).

Lean = 4 arms {trm_mixer, trm_mixer_nomix, trm_flat, ff_matched}: drops M30's two expensive untied
mixer arms (the ~7.9x ceiling) since the shared-readout confound is resolved by the nomix arm alone.
trm_mixer/trm_flat/ff_matched keep the EXACT M30 widths (so those arms reproduce M30 bit-identically
and the M30 headline Δ is re-derived as a cross-check); trm_mixer_nomix uses the budget-matched
hidden_dim from probe_budget.py.
"""
from pathlib import Path

LOOKBACK = 96
DS = {
    "etth1": dict(vars=7, latent=96, mixer_h={192: 424, 336: 512, 720: 624},
                  nomix_h={192: 428, 336: 516, 720: 624}),
    "weather": dict(vars=21, latent=192, mixer_h={192: 912, 336: 1200, 720: 1592},
                    nomix_h={192: 916, 336: 1196, 720: 1620}),
}
OUT = Path("configs/experiments")

TEMPLATE = """# M31 — the M30 shared-readout confound control on `{ds}` (H={H}). One knob vs M30: it ADDS the
# `trm_mixer_nomix` arm (a shared-readout, NON-mixing loop) and drops M30's two untied-mixer arms.
#
# M30 flagged that both mixer arms beat trm_flat/ff with TWO advantages at once — (i) cross-cell
# token-MIXING and (ii) a per-cell SHARED readout Linear(latent,H), vs the flat/ff arms' UNSHARED
# M*H readout that horizon inflates — and no M30 arm separated them. `trm_mixer_nomix` holds the
# shared readout and removes ONLY the mixing (re-widened to re-match trm_flat's budget: hidden={nomix_h}).
# The decomposition (NEGATIVE Δ favours the first arm — lower MSE is better):
#   Δ(trm_mixer − trm_mixer_nomix)  = token-mixing contribution at a HELD shared readout
#   Δ(trm_mixer_nomix − ff_matched) = what the shared-readout / channel-independent param buys alone
#   Δ(trm_mixer − ff_matched)       = M30 headline (reproduced; = sum of the two above as a check)
# trm_mixer/trm_flat/ff_matched keep the EXACT M30 widths so those arms reproduce M30 bit-identically.

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
    label: trm_mixer            # cross-VARIABLE mixing loop (M30 width, matches trm_flat)
    hidden_dim: {mixer_h}
    latent_dim: {latent}
    token_hidden: 8
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm_mixer_nomix
    label: trm_mixer_nomix     # M31 control: SHARED readout, token-mixing REMOVED (re-widened to budget)
    hidden_dim: {nomix_h}
    latent_dim: {latent}
    token_hidden: 8            # accepted + ignored (no token-mix built)
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm
    label: trm_flat            # flat loop (channel-dependent, no mixing, UNSHARED M*H readout) — budget ref
    hidden_dim: 64
    latent_dim: 64
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: ff_matched
    label: ff_matched         # §4a param-matched shallow joint MLP (UNSHARED M*H readout) — external control
    hidden_dim: 64
    latent_dim: 64
    n_steps: 8

deltas:
  - [trm_mixer, trm_mixer_nomix]
  - [trm_mixer_nomix, ff_matched]
  - [trm_mixer, ff_matched]
  - [trm_flat, ff_matched]
  - [trm_mixer, trm_flat]

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
        )
        path = OUT / f"m31_{ds}_h{H}.yaml"
        path.write_text(text)
        print("wrote", path)
