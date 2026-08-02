"""Width probe for M34b: match every arm's param count to the flat `trm` reference at
in_features = w (data) + w (one noise channel per cell), out_features = w."""
import sys
sys.path.insert(0, "src")
from looptab.registry import get_model

def n(name, **kw):
    return get_model(name, **kw).count_params()

for w in (24,):
    d = w + w  # data + per-cell noise channel
    ref = n("trm", in_features=d, num_classes=2, hidden_dim=64, latent_dim=64, n_steps=8,
            out_features=w, use_rmsnorm=True)
    print(f"w={w} in={d}  trm_flat ref = {ref}")
    for name, kw in (
        ("ff_matched", {}),
        ("trm_mixer", {"token_hidden": 48, "use_rmsnorm": True}),
        ("untied_mixer_matched", {"token_hidden": 48, "use_rmsnorm": True}),
    ):
        best = None
        for h in range(8, 400):
            for lat in ({64} if name == "ff_matched" else {h, 64}):
                try:
                    p = n(name, in_features=d, num_classes=2, hidden_dim=h, latent_dim=lat,
                          n_steps=8, out_features=w, **kw)
                except Exception:
                    continue
                r = p / ref
                if best is None or abs(r - 1) < abs(best[3] - 1):
                    best = (h, lat, p, r)
        print(f"  {name:24s} hidden={best[0]:4d} latent={best[1]:4d} params={best[2]:7d} ratio={best[3]:.4f}")
