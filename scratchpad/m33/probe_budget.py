"""M33: solve control-arm widths to trm_flat's budget on the SYNTHETIC mixer-win tasks.

M31/M32 decomposed the FORECASTING mixer win into mixing / channel-independence / shared-readout /
weight-sharing and found mixing is net harmful there. That decomposition was never run on the
SYNTHETIC constraint-coupled tasks where the mixer genuinely wins (Sudoku, converge, hopfield,
disruption, mixed_converge). M33 ports the exact M32 7-arm decomposition to those tasks.

This probe fixes the four control arms' widths so each lands within ±5% of `trm_flat`'s param count
(the same budget_reference every baseline mixer-win config uses). Geometry (in_features / num_classes
/ out_features) is read from the REAL data via make_splits — identical to run.py's inference path — so
the printed widths are exactly what run.py will construct.

Arms solved (each a single width w = hidden = latent; token_hidden fixed per task as in the baseline):
  trm_mixer_nomix             mix OFF, SHARED readout            (channel-independence body)
  trm_mixer_unsharedro        mix ON,  UNSHARED per-cell readout (readout isolation, CD body)
  trm_mixer_nomix_unsharedro  mix OFF, UNSHARED per-cell readout (readout isolation, CI body)
  trm_mixer_nomix_distinctw   mix OFF, per-cell DISTINCT weights (weight-sharing isolation)

NOTE (expected, and itself part of the result): on these tasks num_classes is 2 (binary suite) or 6
(sudoku), so the per-cell readout is TINY relative to the channel MLP — unlike forecasting where the
M×H readout ballooned. So the unshared-readout / distinct-weight arms should re-widen only modestly,
i.e. the forecasting "cheap-parameterization frees width" mechanism is largely absent here. trm_mixer
itself keeps its LOCKED baseline width (printed as a cross-check, not re-solved).
"""

from looptab.data.dataset import make_splits
from looptab.registry import get_model

# Each entry mirrors the baseline mixer-win config: task name, params, the swept width values with the
# per-point task-param override, the flat-ref width, the mixer's locked (hidden, latent, token_hidden),
# and token_hidden for the control arms (matches the mixer's, ignored by nomix arms).
CONFIGS = {
    "converge": dict(
        params=dict(rule=78, distractors=0),
        sweep_param="w", sweep=[24, 32, 48],
        flat_hidden=64, mixer=dict(hidden=96, latent=64, token_hidden=48), token_hidden=48,
    ),
    "hopfield": dict(
        params=dict(weights="hebbian", n_patterns=12, gamma=16, distractors=0, T=6),
        sweep_param="w", sweep=[24, 32],
        flat_hidden=64, mixer=dict(hidden=96, latent=64, token_hidden=48), token_hidden=48,
    ),
    "disruption_w24": dict(
        params=dict(w=24, n_banks=4, w_rot=6, w_bank=3, min_tail=2, max_tail=8, gamma=14, distractors=0),
        sweep_param=None, sweep=[None],
        flat_hidden=64, mixer=dict(hidden=96, latent=64, token_hidden=192), token_hidden=192,
    ),
    "disruption_w32": dict(
        params=dict(w=32, n_banks=4, w_rot=6, w_bank=3, min_tail=2, max_tail=8, gamma=14, distractors=0),
        sweep_param=None, sweep=[None],
        flat_hidden=64, mixer=dict(hidden=96, latent=64, token_hidden=192), token_hidden=192,
    ),
    "mixed_converge": dict(
        params=dict(rule_set=[78, 92, 141, 197], distractors=0, T=6),
        sweep_param="w", sweep=[24, 32],
        flat_hidden=64, mixer=dict(hidden=96, latent=64, token_hidden=48), token_hidden=48,
    ),
    "sudoku": dict(
        params=dict(size=6),
        sweep_param="n_givens", sweep=[24, 18, 14],
        flat_hidden=128, mixer=dict(hidden=272, latent=192, token_hidden=64), token_hidden=64,
    ),
}
# disruption_w{24,32} gamma is the baseline's; task name for make_splits strips the _wNN suffix.
TASK_NAME = {"disruption_w24": "disruption", "disruption_w32": "disruption"}

NEW_ARMS = ["trm_mixer_unsharedro", "trm_mixer_nomix_unsharedro", "trm_mixer_nomix_distinctw"]


def geometry(task, params):
    """Replicate run.py's dim inference from the real data (seed 0, tiny n)."""
    train_ds, _ = make_splits(task=task, task_cfg=params, task_seed=42,
                              train_sample_seed=1, test_sample_seed=2, n_train=256, n_test=64, seed=0)
    X0, _ = train_ds[0]
    in_features = int(X0.shape[0])
    num_classes = max(2, int(train_ds.y.max()) + 1)
    multi_output = train_ds.y.ndim > 1
    out_features = int(train_ds.y.shape[-1]) if multi_output else None
    return in_features, num_classes, out_features


def params_of(name, geo, **kw):
    in_features, num_classes, out_features = geo
    m = get_model(name, in_features=in_features, num_classes=num_classes,
                  out_features=out_features, **kw)
    return m.count_params()


for cfgname, spec in CONFIGS.items():
    task = TASK_NAME.get(cfgname, cfgname)
    th = spec["token_hidden"]
    mx = spec["mixer"]
    for pt in spec["sweep"]:
        params = dict(spec["params"])
        label = ""
        if spec["sweep_param"] is not None:
            params[spec["sweep_param"]] = pt
            label = f"{spec['sweep_param']}={pt}"
        geo = geometry(task, params)
        in_f, nc, of = geo
        ref = params_of("trm", geo, hidden_dim=spec["flat_hidden"], latent_dim=spec["flat_hidden"],
                        n_steps=8, deep_supervision=True, use_rmsnorm=True)
        mixer = params_of("trm_mixer", geo, hidden_dim=mx["hidden"], latent_dim=mx["latent"],
                          n_steps=8, deep_supervision=True, use_rmsnorm=True, token_hidden=mx["token_hidden"])
        nomix_best = None
        for w in range(4, 4000, 2):
            n = params_of("trm_mixer_nomix", geo, hidden_dim=w, latent_dim=w, n_steps=8,
                          deep_supervision=True, use_rmsnorm=True, token_hidden=th)
            r = n / ref
            if nomix_best is None or abs(r - 1.0) < abs(nomix_best[2] - 1.0):
                nomix_best = (w, n, r)
            if n > 1.2 * ref:
                break
        print(f"\n{cfgname:16s} {label:12s} in={in_f} nc={nc} cells={of}  "
              f"ref(trm_flat)={ref:8d}  trm_mixer ratio={mixer/ref:.4f}")
        w, n, r = nomix_best
        print(f"    {'trm_mixer_nomix':28s} w={w:5d} params={n:8d} ratio={r:.4f}")
        for name in NEW_ARMS:
            best = None
            for w in range(4, 4000, 2):
                n = params_of(name, geo, hidden_dim=w, latent_dim=w, n_steps=8,
                              deep_supervision=True, use_rmsnorm=True, token_hidden=th)
                r = n / ref
                if best is None or abs(r - 1.0) < abs(best[2] - 1.0):
                    best = (w, n, r)
                if n > 1.2 * ref:
                    break
            w, n, r = best
            print(f"    {name:28s} w={w:5d} params={n:8d} ratio={r:.4f}")
