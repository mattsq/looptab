"""M34: solve control-arm widths (7-arm M31/M32 decomposition) for the forecasting-breadth
milestone, at H=24 (the M26 baseline horizon, never decomposed before -- M31/M32 only covered
H in {192,336,720}) and for two new high-channel-count datasets (electricity M=321, traffic
M=862) that give the channel-count dose-response curve CLAUDE.md flags as open (§11.2 #14).

etth2/ettm1/ettm2 have IDENTICAL geometry to etth1 (M=7, same lookback/horizon), so their
widths are dataset-content-independent -- only shape matters. trm_mixer/trm_flat/ff_matched
reuse the M26 etth1-h24 config's exact widths unchanged; this script only needs to solve the
four NEW M31/M32 control arms (nomix, unsharedro, nomix_unsharedro, nomix_distinctw) to
trm_flat's budget, which was never done at H=24.

Uses the real get_model builder so printed widths are exactly what run.py will construct.
"""

from looptab.registry import get_model

LOOKBACK = 96
HORIZON = 24
# M26's etth1-h24 baseline widths (dataset-content-independent -- shared by etth2/ettm1/ettm2).
FLAT_HIDDEN = 64
MIXER_HIDDEN = 224
MIXER_LATENT = 96
TOKEN_HIDDEN = 8

# (dataset family -> M vars). electricity/traffic are new high-channel-count datasets. weather
# added post-hoc (M=21) for the recipe-isolation follow-up's third channel-count data point --
# M31/M32 only solved weather widths at H in {192,336,720}, never H=24.
GEOMETRIES = {
    "ett_family (etth1/etth2/ettm1/ettm2)": 7,
    "weather": 21,
    "electricity": 321,
    "traffic": 862,
}


def params(name, in_features, num_classes, out_features, **kw):
    m = get_model(name, in_features=in_features, num_classes=num_classes,
                  out_features=out_features, **kw)
    return m.count_params()


for label, M in GEOMETRIES.items():
    in_features = M * LOOKBACK
    num_classes, out_features = HORIZON, M
    ref = params("trm", in_features, num_classes, out_features,
                 hidden_dim=FLAT_HIDDEN, latent_dim=FLAT_HIDDEN, n_steps=8,
                 deep_supervision=True, use_rmsnorm=True)
    # For the ETT family, cross-check the LOCKED M26 mixer width still lands near-budget; for the
    # new high-channel geometries, search trm_mixer's own width too (hidden=latent=w, like the
    # other arms -- a single-width search is simpler than the historical hidden!=latent split and
    # budget-fairness only needs the TOTAL to match, not that exact split).
    mixer_ratio = None
    if M == 7:
        mixer = params("trm_mixer", in_features, num_classes, out_features,
                       hidden_dim=MIXER_HIDDEN, latent_dim=MIXER_LATENT, n_steps=8,
                       deep_supervision=True, use_rmsnorm=True, token_hidden=TOKEN_HIDDEN)
        mixer_ratio = mixer / ref
    print(f"\n{label}  M={M}  in_features={in_features}  ref(trm_flat,hidden={FLAT_HIDDEN})={ref:,d}"
          + (f"  trm_mixer(hidden={MIXER_HIDDEN}) ratio={mixer_ratio:.4f}" if mixer_ratio else ""))

    names = ["trm_mixer_nomix", "trm_mixer_unsharedro", "trm_mixer_nomix_unsharedro",
             "trm_mixer_nomix_distinctw"]
    if M != 7:
        names = ["trm_mixer"] + names
    for name in names:
        best = None
        w = 4
        while w < 20000:
            n = params(name, in_features, num_classes, out_features, hidden_dim=w, latent_dim=w,
                       n_steps=8, deep_supervision=True, use_rmsnorm=True, token_hidden=TOKEN_HIDDEN)
            r = n / ref
            if best is None or abs(r - 1.0) < abs(best[2] - 1.0):
                best = (w, n, r)
            if n > 1.15 * ref:
                break
            # Coarser step for the large-M search space -- exact match isn't needed, +-5% is.
            w += 2 if w < 200 else (10 if w < 2000 else 50)
        w, n, r = best
        print(f"    {name:28s} w={w:6d} (hidden=latent) params={n:10,d} ratio={r:.4f}")
