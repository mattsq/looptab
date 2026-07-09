# M6a — DONE. The both-axes probe (multi_parity). §9 gate is empirically UNSATISFIABLE by the generalist; loop is depth-positive, NOT a robust generalist (the "never-worst" property is falsified).

The §11.3(i) lever, run as an experiment rather than settled by fiat. After M0–M5 the §9
gate ("loop beats BOTH controls on A AND B") was unmet for a *structural* reason: each
canonical task needs exactly ONE axis (A→depth, B→width), so the single-axis control owning
that axis always TIES the loop. M6a builds the one task that needs **both depth and width at a
fixed tiny budget** — exactly where a generalist *should* beat both specialists — and asks
empirically whether a loop-beats-both cell exists at all.

**Task = `multi_parity`** (new generator, determinism-tested): predict `w` **independent**
k-parities in parallel from the same `d` bits. Depth axis = each output is order-`k` (shallow
`ff_matched` should fail at k≥4, per M4); width axis = `w` parallel computations (narrow
`untied_matched` blocks should bottleneck). NOT Task C — the `w` parities are independent, no
sub-problem feeds another; `w=1` reduces exactly to Task A (sanity anchor, asserted in tests).
Grid **k∈{3,4} × w∈{1,4,8}** at d=20, 5 arms (4 required + `untied_stack` ceiling), 10 seeds,
hyperparameters inherited verbatim from `m4_parity_grid.yaml` (no per-cell tuning, §8). New
code: `make_multi_parity` (+6 tests), one `make_splits` branch, the `TaskConfig.name` literal,
one config. 80 tests, ruff clean. Tracked:
`results/m6a_multi_parity_grid_20260622T080206_{curve,deltas,params}.csv` (+ JSON).
**Budget parity CLEAN** — all matched arms within ±0.7% in every cell (no width-quantization
breach; the answer rides on no confound).

**Per-arm test accuracy (token-acc, 10 seeds; us = untied_stack ceiling) and the two headline Δs:**

| k | w | baseline | loop (nods) | ff_matched | untied_matched | us (ceiling) | Δ(loop−ff) | Δ(loop−um) |
|---|---|---|---|---|---|---|---|---|
| 3 | 1 | .512 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 (tie) | 0.000 (tie) |
| 3 | 4 | .506 | 0.852 | **1.000** | 0.901 | 0.950 | **−0.148** (0/9, p=.004) | −0.049 (4/5, ns) |
| 3 | 8 | .503 | 0.793 | **0.982** | 0.733 | 0.843 | **−0.189** (0/10, p=.002) | +0.059 (7/3, ns) |
| 4 | 1 | .513 | 1.000 | 0.772 | 1.000 | 1.000 | **+0.228** (6/0, p=.031) | 0.000 (tie) |
| 4 | 4 | .508 | 0.827 | 0.557 | 0.810 | 0.931 | **+0.269** (10/0, p=.002) | +0.016 (7/3, ns) |
| 4 | 8 | .504 | 0.715 | 0.533 | 0.680 | 0.775 | **+0.182** (10/0, p=.002) | +0.035 (7/3, ns) |

**Reading (per §2/§8 — the pre-registered honesty clause fires; outcome (b)).**

1. **ZERO loop-beats-both cells.** The pre-registered condition (Δ(loop−ff) > 0 AND Δ(loop−um) > 0,
   *both* sign-test p<.05) is met in **no cell**. Where Δ(loop−ff) is strongly significant (k=4,
   all w: +0.18…+0.27, 10/0, p=.002), Δ(loop−um) is +0.016…+0.035, **ns** (7/3, p=.34) — the loop
   ties the deep control even with the width axis maximally stressed (w=8). Stressing both axes at
   once did **not** create the regime where tying pays; the two deep arms (loop, um) degrade
   *together* as w grows.
2. **`multi_parity` UNIFIES Task A and Task B into one task; k is the axis dial.** k=4
   (depth-demanding) reproduces the **Task A** pattern — loop beats shallow ff (depth), ties deep
   um (tying neutral) — and now **extends it to multi-output** (w=4, w=8). k=3 (easy parity, depth
   *not* needed) reproduces the **Task B** pattern — the wide shallow `ff_matched` is the **best**
   arm and the loop is **significantly beaten by it** (−0.148/−0.189, p≤.004). So at every tested
   operating point one single-axis specialist matches-or-beats the loop on the axis that matters
   and the other specialist is irrelevant. **In no tested cell does the generalist beat both**, and
   the k-dial argument *explains why*. **Caveat (adversarial review, do not overclaim):** this is
   "unsatisfied in every tested cell," NOT a proof of impossibility. In the k=4 cells the loop does
   edge `untied_matched` in the *predicted* direction (+0.016 w=4, +0.035 w=8) but
   **non-significantly** (7/3 seeds, p=.34) — under-powered, not a demonstrated tie-in-principle;
   and the grid is coarse (`w` 1→4→8, one d, one budget, one model size). Higher seeds on the
   k=4/w≥4 cells (or finer/larger `w`) would settle whether the gate is truly unsatisfiable or
   merely unmet here. The defensible reading: the gate is **unsatisfied in every tested cell with a
   structural reason (the k-dial)** — strengthening, not proving, the M2-confirm suspicion that the
   wording should change.
3. **The "tying buys width" half of the M2 synthesis does NOT replicate on the parity family.**
   Δ(loop−um) is ns in every multi-output cell — adding width pressure to parity does **not** make
   the fair untied stack fail relative to the loop, unlike CA (M2-confirm, where um was worst in
   6/6). So the CA tying advantage was **CA-specific**, not a general "tied recurrence buys width";
   on parity tying is neutral whether w=1 (M4) or w=8 (M6a). Both halves of the M2 synthesis are
   now retracted on parity (depth half already softened in M3a; width half here).
4. **The loop's last defensible property — "never the worst param-matched arm" (M2-confirm/M4) —
   is FALSIFIED.** At k=3,w∈{4,8} the loop is *significantly* beaten by `ff_matched` (a mandatory
   param-matched control), and at k=3,w=4 it is nominally the worst of the three param-matched arms
   (0.852 < um 0.901 < ff 1.000). Once depth is unneeded and width dominates, the loop is a
   middling generalist, not Pareto-safe. The honest residual claim is narrower than "robust":
   **the loop owns the *depth* axis** (beats shallow ff when interaction order is high; replicated
   M0/M2/M4 and extended to multi-output here) and is **tying-neutral** — it is depth-positive, not
   a robust all-rounder.
5. **DS inert, ceiling behaves.** |Δ(trm_ds − trm_nods)| ≤ 0.036, never significant (consistent
   M0–M5). The 4× `untied_stack` ceiling is the best arm in the hard k=4,w≥4 cells (capacity helps,
   as expected) — a labelled reference, not a control.

**Net (§9).** The user-chosen "resolve empirically first" path is resolved as far as 6 cells can:
a task built to need both axes yields **zero loop-beats-both cells in every tested cell**, because
its difficulty collapses onto a single dial (k) that hands the relevant specialist a tie (k=4) or a
win (k=3). This is **strong (not conclusive) evidence** the literal §9 gate is unsatisfiable by a
single-loop generalist judged against single-axis specialists — enough to motivate **relaxing the
wording**, with the under-powered k=4 width cells (finding above) the place to push if a stricter
proof is wanted. M6a additionally shows the natural fallback ("loop never significantly worse across
both tasks") is **definitively unmet** (k=3 wide, p≤.004 — this one IS conclusive). The earned,
defensible statement of the loop's value is: **depth-positive on high-order parity, tying-neutral,
no depth-extrapolation (M1/M3b/M7), not a robust generalist.** Do **not** build Task C on this.

**Caveats / open gaps.** (i) Token-acc is the meaningful metric at w=8 (whole-row EM is low for
all arms, ~0.05, over 8 outputs). (ii) One architecture size; the regime is below the
overfit/sample walls of M4/M5 (no all-arms-at-chance cell here, so every cell carries a verdict).
(iii) The remaining §11.3 levers (broaden M3b across rules/widths; a PonderNet/ACT halting
objective vs the extrapolation null) are untouched and now lower-value, since M6a settles the
top lever (i) negatively.

---
