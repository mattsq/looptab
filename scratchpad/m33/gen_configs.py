"""M33: emit the per-(task,width) configs that port the M32 7-arm shared-readout decomposition to the
SYNTHETIC mixer-win tasks (converge, hopfield, disruption, mixed_converge, sudoku).

One config per (task, width) because the geometry — and hence the trm_flat budget the control arms are
matched to — changes with the output width (exactly disruption's existing per-width rationale). SUDOKU
is the exception: its encoding size (in=252, cells=36, num_classes=6) is constant across n_givens, so it
stays a single swept config. Control-arm widths are from scratchpad/m33/probe_budget.py (each within
±5% of trm_flat except the coarse-grained `distinctw` arm, whose per-cell distinct weights make the
param count jump in big steps — those breaches are CONSERVATIVE/under-budget and documented, the same
pattern as the M24 w=48 mixer). trm_mixer / trm_flat / ff_matched keep their LOCKED baseline widths so
Δ(trm_mixer − ff) and Δ(trm_flat − ff) reproduce the shipped M24/M24c/M24f/M28a/M23 headline numbers.

The 9-pair delta list is the M32 decomposition, read in CLASSIFICATION sign (POSITIVE Δ favours the
first arm on accuracy/exact_match — the OPPOSITE of M31/M32's negative-Δ-favours-first MSE):
  [trm_mixer, trm_mixer_nomix]              = token-MIXING at a held shared readout (the key leg;
                                              expected LARGE POSITIVE on coupled tasks = mixing is
                                              the mechanism, mirror-image of forecasting)
  [trm_mixer_nomix, ff_matched]             = shared-readout / channel-independence vs the joint MLP
                                              (expected SMALL — little free lunch when num_classes∈{2,6})
  [trm_mixer_nomix, trm_mixer_nomix_unsharedro]  = the shared readout (clean single-flag flip, CI body)
  [trm_mixer, trm_mixer_unsharedro]              = the shared readout (CD body, cross-check)
  [trm_mixer_unsharedro, trm_mixer_nomix_unsharedro] = mixing at an unshared readout (cross-check)
  [trm_mixer_nomix, trm_mixer_nomix_distinctw]   = per-variable weight-sharing (CI + shared-RO held)
  [trm_mixer_nomix_unsharedro, ff_matched]       = per-cell CI vs joint MLP at unshared readout
  [trm_mixer, ff_matched]                        = HEADLINE mixer win (reproduces the baseline)
  [trm_flat, ff_matched]                         = context (flat≈ff on Sudoku; flat>ff on converge)
"""

import textwrap

# ---- probed control-arm widths (hidden=latent=w) from probe_budget.py -------------------------------
# converge / hopfield / mixed_converge share geometry per w (in=w, cells=w, num_classes=2).
RING = {
    24: dict(nomix=82, unsharedro=66, nomix_unsharedro=72, distinctw=16),
    32: dict(nomix=88, unsharedro=66, nomix_unsharedro=74, distinctw=16),
    48: dict(nomix=96, unsharedro=68, nomix_unsharedro=78, distinctw=14),
}
DISRUPT = {
    24: dict(nomix=100, unsharedro=66, nomix_unsharedro=90, distinctw=20),
    32: dict(nomix=110, unsharedro=68, nomix_unsharedro=96, distinctw=18),
}
SUDOKU = dict(nomix=232, unsharedro=184, nomix_unsharedro=188, distinctw=38)

DELTAS = """deltas:
  # KEY leg — token-mixing at a held shared readout (expected LARGE POSITIVE = mixing is the mechanism)
  - [trm_mixer, trm_mixer_nomix]
  # shared-readout / channel-independence vs the joint MLP (expected SMALL on synthetic)
  - [trm_mixer_nomix, ff_matched]
  # the shared readout, isolated (clean CI-body flip + CD-body cross-check)
  - [trm_mixer_nomix, trm_mixer_nomix_unsharedro]
  - [trm_mixer, trm_mixer_unsharedro]
  # mixing at an unshared readout (cross-check of the key leg)
  - [trm_mixer_unsharedro, trm_mixer_nomix_unsharedro]
  # per-variable weight-sharing (CI + shared readout held)
  - [trm_mixer_nomix, trm_mixer_nomix_distinctw]
  # per-cell CI vs joint MLP at unshared readout
  - [trm_mixer_nomix_unsharedro, ff_matched]
  # HEADLINE mixer win (reproduces baseline) + context
  - [trm_mixer, ff_matched]
  - [trm_flat, ff_matched]
"""


def arms_block(mixer_hidden, mixer_latent, token_hidden, flat_hidden, w):
    """7-arm block: mixer/flat/ff at baseline widths; the four controls at probed width w=hidden=latent."""
    return f"""arms:
  - name: trm_mixer
    label: trm_mixer                   # cross-cell MIXING loop (baseline width, budget ref = trm_flat)
    hidden_dim: {mixer_hidden}
    latent_dim: {mixer_latent}
    token_hidden: {token_hidden}
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm_mixer_nomix
    label: trm_mixer_nomix             # SHARED readout, token-mixing REMOVED (channel-independent)
    hidden_dim: {w['nomix']}
    latent_dim: {w['nomix']}
    token_hidden: {token_hidden}
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm_mixer_unsharedro
    label: trm_mixer_unsharedro        # mix ON, UNSHARED per-cell readout (readout isolate, CD body)
    hidden_dim: {w['unsharedro']}
    latent_dim: {w['unsharedro']}
    token_hidden: {token_hidden}
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm_mixer_nomix_unsharedro
    label: trm_mixer_nomix_unsharedro  # mix OFF, UNSHARED per-cell readout (readout isolate, CI body)
    hidden_dim: {w['nomix_unsharedro']}
    latent_dim: {w['nomix_unsharedro']}
    token_hidden: {token_hidden}
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm_mixer_nomix_distinctw
    label: trm_mixer_nomix_distinctw   # mix OFF, per-cell DISTINCT channel weights (weight-share isolate)
    hidden_dim: {w['distinctw']}
    latent_dim: {w['distinctw']}
    token_hidden: {token_hidden}
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: trm
    label: trm_flat                    # flat loop (channel-dependent, unshared M*H readout) — budget ref
    hidden_dim: {flat_hidden}
    latent_dim: {flat_hidden}
    n_steps: 8
    n_latent: 1
    use_rmsnorm: true
    deep_supervision: true
  - name: ff_matched
    label: ff_matched                  # §4a param-matched shallow joint MLP — anchor
    hidden_dim: {flat_hidden}
    latent_dim: {flat_hidden}
    n_steps: 8
"""


TRAIN = """train:
  epochs: {epochs}
  lr: 1e-3
  weight_decay: 1e-4
  batch_size: 256
  device: cpu

budget_reference: trm_flat
budget_tol: 0.05

seeds: {seeds}
parallel_workers: {workers}
results_dir: results
"""

HEADER = ("# M33 — port the M31/M32 shared-readout / channel-independence DECOMPOSITION to the synthetic\n"
          "# mixer-win task `{task}`{wnote}. Same lean recipe as the baseline ({base}); the ONLY change is the\n"
          "# 7-arm decomposition set (adds trm_mixer_nomix / _unsharedro / _nomix_unsharedro /\n"
          "# _nomix_distinctw) budget-matched to trm_flat, to measure whether the mixer win here is genuine\n"
          "# token-MIXING (expected — the mirror-image of forecasting, where mixing was net harmful) or the\n"
          "# shared-readout / channel-independent parameterization (M31/M32). Control widths from\n"
          "# scratchpad/m33/probe_budget.py. num_threads = repo default 1 (mixer/decoupled 3-D matmul).\n")


def ring_task_block(task, extra):
    return "task:\n  name: {}\n  params:\n{}\n  n_train: 4000\n  n_test: 1000\n  task_seed: 42\n  train_sample_seed: 1\n  test_sample_seed: 2\n".format(task, extra)


CONFIGS = {}

# converge (rule 78) — per width 24/32/48
for wv in [24, 32, 48]:
    params = "    rule: 78\n    distractors: 0\n    w: {}".format(wv)
    body = (HEADER.format(task="converge", wnote=f" (w={wv})", base="m24_mixer_converge.yaml")
            + "\n" + ring_task_block("converge", params) + "\n"
            + arms_block(96, 64, 48, 64, RING[wv]) + "\n" + DELTAS + "\n"
            + TRAIN.format(epochs=100, seeds="[0, 1, 2, 3, 4, 5, 6, 7]", workers=4))
    CONFIGS[f"m33_converge_w{wv}"] = body

# hopfield (hebbian dense) — per width 24/32
for wv in [24, 32]:
    params = ("    weights: hebbian\n    n_patterns: 12\n    gamma: 16\n    distractors: 0\n"
              "    T: 6\n    w: {}".format(wv))
    body = (HEADER.format(task="hopfield", wnote=f" (w={wv})", base="m24c_hopfield_mixer.yaml")
            + "\n" + ring_task_block("hopfield", params) + "\n"
            + arms_block(96, 64, 48, 64, RING[wv]) + "\n" + DELTAS + "\n"
            + TRAIN.format(epochs=100, seeds="[0, 1, 2, 3, 4, 5, 6, 7]", workers=4))
    CONFIGS[f"m33_hopfield_w{wv}"] = body

# mixed_converge (rule_set) — per width 24/32
for wv in [24, 32]:
    params = ("    rule_set: [78, 92, 141, 197]\n    distractors: 0\n    T: 6\n    w: {}".format(wv))
    body = (HEADER.format(task="mixed_converge", wnote=f" (w={wv})", base="m28a_mixed_converge_mixer.yaml")
            + "\n" + ring_task_block("mixed_converge", params) + "\n"
            + arms_block(96, 64, 48, 64, RING[wv]) + "\n" + DELTAS + "\n"
            + TRAIN.format(epochs=100, seeds="[0, 1, 2, 3, 4, 5, 6, 7]", workers=4))
    CONFIGS[f"m33_mixed_converge_w{wv}"] = body

# disruption — per width 24/32 (token_hidden 192). gamma is the MINIMAL-PSD int per WIDTH (task_seed=42):
# 14 @ w24, 15 @ w32 — must match the shipped m24f_disruption_mixer_w{24,32}.yaml or the target changes.
DISRUPT_GAMMA = {24: 14, 32: 15}
for wv in [24, 32]:
    params = ("    w: {}\n    n_banks: 4\n    w_rot: 6\n    w_bank: 3\n    min_tail: 2\n"
              "    max_tail: 8\n    gamma: {}\n    distractors: 0".format(wv, DISRUPT_GAMMA[wv]))
    body = (HEADER.format(task="disruption", wnote=f" (w={wv})", base=f"m24f_disruption_mixer_w{wv}.yaml")
            + "\n" + ring_task_block("disruption", params) + "\n"
            + arms_block(96, 64, 192, 64, DISRUPT[wv]) + "\n" + DELTAS + "\n"
            + TRAIN.format(epochs=100, seeds="[0, 1, 2, 3, 4, 5, 6, 7]", workers=4))
    CONFIGS[f"m33_disruption_w{wv}"] = body

# sudoku — single swept config over n_givens (geometry constant: in=252, cells=36, num_classes=6)
sud_task = ("task:\n  name: sudoku\n  params:\n    size: 6\n  n_train: 8000\n  n_test: 1000\n"
            "  task_seed: 42\n  train_sample_seed: 1\n  test_sample_seed: 2\n\n"
            "sweep:\n  param: n_givens\n  values: [24, 18, 14]   # medium -> hard -> very hard\n")
sud_body = (HEADER.format(task="sudoku", wnote=" (n_givens sweep; geometry constant)",
                          base="m23_sudoku_mixer_lean.yaml")
            + "\n" + sud_task + "\n"
            + arms_block(272, 192, 64, 128, SUDOKU) + "\n" + DELTAS + "\n"
            + TRAIN.format(epochs=15, seeds="[0, 1, 2, 3, 4, 5]", workers=3))
CONFIGS["m33_sudoku"] = sud_body

if __name__ == "__main__":
    import pathlib
    outdir = pathlib.Path("configs/experiments")
    for name, body in CONFIGS.items():
        p = outdir / f"{name}.yaml"
        p.write_text(body)
        print(f"wrote {p}")
