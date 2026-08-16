"""M34: solve control-arm widths for the SYNTHETIC scale-up (direction A) -- pushing the two
best-understood mixer-win tasks past M33's widest tested points, now that cuda_graph/compile/amp
make it affordable:
  converge : w in {24,32,48} tested (M9/M33) -> push to {64, 96}
  sudoku   : size=6 tested (M23/M33) -> push to size=9 (classic 9x9), n_givens in {30,26,24}
             (verified tractable to generate: ~43/123/360 ms/puzzle respectively)

Same 7-arm M31/M32/M33 decomposition recipe. Mirrors scratchpad/m33/probe_budget.py but widens
the nomix search cap (large-w sudoku size=9 needs it) and searches trm_mixer's own width for the
new points too (single-width hidden=latent=w search, like scratchpad/m34/probe_budget.py) instead
of hand-picking it, since there's no prior locked value to reuse at these new sizes.
"""

from looptab.data.dataset import make_splits
from looptab.registry import get_model

CONFIGS = {
    "converge_w64": dict(task="converge", params=dict(rule=78, distractors=0, w=64), flat_hidden=64),
    "converge_w96": dict(task="converge", params=dict(rule=78, distractors=0, w=96), flat_hidden=96),
    "sudoku9_g30": dict(task="sudoku", params=dict(size=9, n_givens=30), flat_hidden=128),
    "sudoku9_g26": dict(task="sudoku", params=dict(size=9, n_givens=26), flat_hidden=128),
    "sudoku9_g24": dict(task="sudoku", params=dict(size=9, n_givens=24), flat_hidden=128),
}
TOKEN_HIDDEN = {"converge_w64": 48, "converge_w96": 48, "sudoku9_g30": 64, "sudoku9_g26": 64,
                 "sudoku9_g24": 64}
NEW_ARMS = ["trm_mixer", "trm_mixer_nomix", "trm_mixer_unsharedro", "trm_mixer_nomix_unsharedro",
            "trm_mixer_nomix_distinctw"]


def geometry(task, params):
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
    th = TOKEN_HIDDEN[cfgname]
    geo = geometry(spec["task"], spec["params"])
    in_f, nc, of = geo
    ref = params_of("trm", geo, hidden_dim=spec["flat_hidden"], latent_dim=spec["flat_hidden"],
                    n_steps=8, deep_supervision=True, use_rmsnorm=True)
    print(f"\n{cfgname:16s} params={spec['params']}  in={in_f} nc={nc} cells={of}  "
          f"ref(trm_flat,hidden={spec['flat_hidden']})={ref:,d}")
    for name in NEW_ARMS:
        best = None
        w = 4
        while w < 8000:
            n = params_of(name, geo, hidden_dim=w, latent_dim=w, n_steps=8,
                          deep_supervision=True, use_rmsnorm=True, token_hidden=th)
            r = n / ref
            if best is None or abs(r - 1.0) < abs(best[2] - 1.0):
                best = (w, n, r)
            if n > 1.2 * ref:
                break
            w += 2 if w < 300 else (10 if w < 2000 else 50)
        w, n, r = best
        print(f"    {name:28s} w={w:6d} params={n:10,d} ratio={r:.4f}")
