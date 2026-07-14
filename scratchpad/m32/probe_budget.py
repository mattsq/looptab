"""M32: solve widths for the three ingredient-decomposition arms to match trm_flat's budget.

M31 split the forecasting mixer win into Δ(mixer−nomix)[mixing] + Δ(nomix−ff)[shared-readout], and
found the win is the shared-readout / channel-independent parameterization — a bundle of THREE things
nomix has and ff lacks (shared readout, channel-independence, per-variable weight-sharing). M32 adds
three controls, each flipping ONE of those from nomix, budget-matched to trm_flat exactly as M31's
nomix was:

  trm_mixer_unsharedro        mix ON,  UNSHARED per-cell readout   (readout isolation, CD body)
  trm_mixer_nomix_unsharedro  mix OFF, UNSHARED per-cell readout   (readout isolation, CI body)
  trm_mixer_nomix_distinctw   mix OFF, per-cell DISTINCT weights   (weight-sharing isolation)

The unshared readout costs M×(latent·H+H) and distinct weights cost M× the channel MLP, so at fixed
budget these arms are FORCED narrow (the "cheap parameterization frees width" mechanism under test).
We solve a single width w (hidden=latent=w, token_hidden=8 for the mix arm) to trm_flat's budget — the
same nearest-|ratio−1| search FFMatched / UntiedStackMatched use. Uses the real get_model builder so
the printed widths are exactly what run.py will construct; prints trm_mixer / nomix realized ratios as
a cross-check that the locked M30/M31 widths reproduce.
"""

from looptab.registry import get_model

LOOKBACK = 96
# (dataset -> M vars, trm_mixer latent, and the LOCKED M31 trm_mixer / trm_mixer_nomix hidden per H).
DATASETS = {
    "etth1": dict(M=7, latent=96,
                  mixer_hidden={192: 424, 336: 512, 720: 624},
                  nomix_hidden={192: 428, 336: 516, 720: 624}),
    "weather": dict(M=21, latent=192,
                    mixer_hidden={192: 912, 336: 1200, 720: 1592},
                    nomix_hidden={192: 916, 336: 1196, 720: 1620}),
}
HORIZONS = [192, 336, 720]
# New arms solved with a single width w (hidden=latent=w). token_hidden only matters for the mix arm.
NEW_ARMS = ["trm_mixer_unsharedro", "trm_mixer_nomix_unsharedro", "trm_mixer_nomix_distinctw"]


def params(name, in_features, num_classes, out_features, **kw):
    m = get_model(name, in_features=in_features, num_classes=num_classes,
                  out_features=out_features, **kw)
    return m.count_params()


for ds, spec in DATASETS.items():
    M, latent = spec["M"], spec["latent"]
    in_features = M * LOOKBACK
    for H in HORIZONS:
        num_classes, out_features = H, M
        ref = params("trm", in_features, num_classes, out_features,
                     hidden_dim=64, latent_dim=64, n_steps=8, deep_supervision=True,
                     use_rmsnorm=True)
        mixer = params("trm_mixer", in_features, num_classes, out_features,
                       hidden_dim=spec["mixer_hidden"][H], latent_dim=latent, n_steps=8,
                       deep_supervision=True, use_rmsnorm=True, token_hidden=8)
        nomix = params("trm_mixer_nomix", in_features, num_classes, out_features,
                       hidden_dim=spec["nomix_hidden"][H], latent_dim=latent, n_steps=8,
                       deep_supervision=True, use_rmsnorm=True, token_hidden=8)
        print(f"\n{ds:8s} H={H:3d}  ref(trm_flat)={ref:8d}  "
              f"trm_mixer ratio={mixer/ref:.4f}  trm_mixer_nomix ratio={nomix/ref:.4f}")
        for name in NEW_ARMS:
            best = None
            for w in range(4, 4000, 2):
                n = params(name, in_features, num_classes, out_features, hidden_dim=w, latent_dim=w,
                           n_steps=8, deep_supervision=True, use_rmsnorm=True, token_hidden=8)
                r = n / ref
                if best is None or abs(r - 1.0) < abs(best[2] - 1.0):
                    best = (w, n, r)
                if n > 1.15 * ref:
                    break
            w, n, r = best
            print(f"    {name:28s} w={w:5d} (hidden=latent) params={n:8d} ratio={r:.4f}")
