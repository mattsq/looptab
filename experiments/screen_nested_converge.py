"""Structural screen for make_nested_converge: find a convergent + balanced + non-trivial +
genuinely two-timescale (inner depth>1, outer rounds>1) instance."""


import numpy as np

from looptab.data.generators import _inner_relax, ca_step


def characterize(inner_rule, outer_rule, n_blocks, block_w, n=2000, seeds=(1, 2, 3)):
    w = n_blocks * block_w
    max_rounds = 4 * n_blocks
    max_inner = 4 * block_w

    def round_(s):
        return _inner_relax(ca_step(s, outer_rule), n_blocks, block_w, inner_rule, max_inner)

    conv_rates, balances, mean_rounds, max_rounds_seen, mean_inner, trivial = [], [], [], [], [], []
    ok = True
    for sd in seeds:
        rng = np.random.default_rng(sd)
        b = rng.integers(0, 2, size=(n, w))
        s = b.copy()
        depth = np.full(n, -1, dtype=np.int64)
        # also track inner steps used in the FIRST round (proxy for inner timescale)
        inner_steps_first = None
        for step in range(max_rounds):
            # measure inner depth on this state for round 0
            if step == 0:
                outer = ca_step(s, outer_rule)
                blk = outer.reshape(n, n_blocks, block_w)
                isteps = np.zeros(n, dtype=np.int64)
                cur = blk.copy()
                for it in range(max_inner):
                    nxt = ca_step(cur, inner_rule)
                    changed = (nxt != cur).any(axis=(1, 2))
                    isteps[changed] = it + 1
                    if np.array_equal(nxt, cur):
                        break
                    cur = nxt
                inner_steps_first = isteps
            nxt = round_(s)
            newly = (nxt == s).all(axis=1) & (depth < 0)
            depth[newly] = step
            if (depth >= 0).all():
                break
            s = nxt
        fixed = depth >= 0
        cr = fixed.mean()
        conv_rates.append(cr)
        if fixed.sum() < 50:
            ok = False
            continue
        sinf = s[fixed]
        balances.append(sinf.mean())
        d = depth[fixed]
        mean_rounds.append(d.mean())
        max_rounds_seen.append(d.max())
        mean_inner.append(inner_steps_first[fixed].mean())
        # triviality: fraction of rows whose target is all-0 or all-1
        triv = ((sinf.sum(1) == 0) | (sinf.sum(1) == w)).mean()
        trivial.append(triv)
    if not ok or not balances:
        return None
    return dict(
        inner=inner_rule, outer=outer_rule, nb=n_blocks, bw=block_w, w=w,
        conv=np.mean(conv_rates), bal=np.mean(balances),
        rounds=np.mean(mean_rounds), maxr=np.max(max_rounds_seen),
        inner_d=np.mean(mean_inner), triv=np.mean(trivial),
    )


# ff-hard converging ECAs (orbits 0/1) + majority 232 as a strong coupler candidate.
hard = [13, 69, 79, 93, 78, 92, 141, 197]
candidates = []
# n_blocks * block_w around 24-32 (the M9-M12 width regime). block_w >= 5 to avoid tiny-ring cycles.
configs = [(3, 8), (4, 6), (4, 8), (2, 12), (3, 10), (6, 4)]
for nb, bw in configs:
    for inner in hard + [232]:
        for outer in hard + [232]:
            r = characterize(inner, outer, nb, bw, n=1500, seeds=(1, 2, 3))
            if r is None:
                continue
            # want: high convergence, balanced (0.3-0.7), non-trivial (<0.2), two-timescale
            if r["conv"] > 0.6 and 0.25 < r["bal"] < 0.75 and r["triv"] < 0.25 \
               and r["rounds"] > 1.3 and r["inner_d"] > 1.3:
                candidates.append(r)

candidates.sort(key=lambda r: -(r["conv"] * min(r["rounds"], 6) * min(r["inner_d"], 6)))
print(f"{'inner':>5} {'outer':>5} {'nb':>3} {'bw':>3} {'w':>3} {'conv':>5} {'bal':>5} "
      f"{'rounds':>6} {'maxr':>4} {'innerD':>6} {'triv':>5}")
for r in candidates[:25]:
    print(f"{r['inner']:5d} {r['outer']:5d} {r['nb']:3d} {r['bw']:3d} {r['w']:3d} "
          f"{r['conv']:.3f} {r['bal']:.3f} {r['rounds']:6.2f} {r['maxr']:4d} "
          f"{r['inner_d']:6.2f} {r['triv']:.3f}")
print(f"\n{len(candidates)} viable candidates")
