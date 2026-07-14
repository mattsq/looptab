"""M31: find trm_mixer_nomix hidden_dim that matches trm_flat's budget, per (dataset, horizon).

trm_mixer_nomix removes the token-mix params, so at trm_mixer's width it sits UNDER budget; we
re-widen its channel `hidden_dim` (holding latent_dim = trm_mixer's latent) to re-hit trm_flat's
count, exactly as M30 re-widened trm_mixer. Uses the REAL get_model builder so the printed widths
match what run.py will construct. Also prints trm_mixer's realized ratio as a cross-check that the
M30 configs are reproduced here.
"""

from looptab.registry import get_model

LOOKBACK = 96
# (dataset, M vars, latent used by trm_mixer, trm_mixer hidden per horizon from the M30 configs)
DATASETS = {
    "etth1": dict(M=7, latent=96, mixer_hidden={192: 424, 336: 512, 720: 624}),
    "weather": dict(M=21, latent=192, mixer_hidden={192: 912, 336: 1200, 720: 1592}),
}
HORIZONS = [192, 336, 720]


def params(name, in_features, num_classes, out_features, **kw):
    m = get_model(name, in_features=in_features, num_classes=num_classes,
                  out_features=out_features, **kw)
    return m.count_params()


for ds, spec in DATASETS.items():
    M, latent = spec["M"], spec["latent"]
    in_features = M * LOOKBACK
    for H in HORIZONS:
        num_classes = H          # readout width per cell = horizon
        out_features = M         # cells = variables
        ref = params("trm", in_features, num_classes, out_features,
                     hidden_dim=64, latent_dim=64, n_steps=8, deep_supervision=True,
                     use_rmsnorm=True)
        mixer_h = spec["mixer_hidden"][H]
        mixer = params("trm_mixer", in_features, num_classes, out_features,
                       hidden_dim=mixer_h, latent_dim=latent, n_steps=8, deep_supervision=True,
                       use_rmsnorm=True, token_hidden=8)

        # search nomix hidden_dim (latent fixed to trm_mixer's) to minimize |ratio-1|
        best_h, best_ratio = None, None
        for h in range(8, 6000, 4):
            n = params("trm_mixer_nomix", in_features, num_classes, out_features,
                       hidden_dim=h, latent_dim=latent, n_steps=8, deep_supervision=True,
                       use_rmsnorm=True, token_hidden=8)
            r = n / ref
            if best_ratio is None or abs(r - 1.0) < abs(best_ratio - 1.0):
                best_h, best_ratio, best_n = h, r, n
        print(f"{ds:8s} H={H:3d}  ref(trm_flat)={ref:7d}  "
              f"trm_mixer h={mixer_h} ratio={mixer/ref:.3f}  ->  "
              f"nomix h={best_h} params={best_n} ratio={best_ratio:.4f}")
