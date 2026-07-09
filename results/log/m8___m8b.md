# M8 / M8b — DONE. Variable-compute FIXED-POINT task (converging CA). Adaptive compute FAILS (falsifies the M7 hypothesis); but the FIRST (replicated) loop-beats-both surfaces — on whole-row exact-match.

The adaptive-computation angle. Every prior task has a **fixed** required depth, so a fixed-depth
`untied_matched` always matches the loop and tying is neutral (M2…M6a). M8 builds the one regime never
tested: required compute **varies per instance and can exceed any fixed depth** — an
iterate-to-convergence / fixed-point task, where only the loop can spend variable compute (unroll more
at test). Two payoffs: (a) the first real shot at the loop beating BOTH fixed-compute controls via
adaptive computation; (b) a direct test of the M7 hypothesis (on a CONVERGENT target, over-unrolling
should HOLD and progressive loss should fire).

New task `converge` (generator `make_converge`, determinism + fixed-point + trajectory tests): map s0
to the CA's **fixed point** s_inf (iterate a *converging* rule until `ca_step(s)==s`). Verified
non-degenerate (rule 92, w=32: majority baseline 0.562; ~99% of rows need >1 step; convergence depth
varies per instance — ~45% need >4 steps, ~13% >6, max ~19). `make_trajectory_dataset` now dispatches
to `make_converge`; one `make_splits` branch; `TaskConfig.name` literal extended. 92 tests, ruff clean.
Tracked: `results/m8_converge_adaptive_20260622T102356_*` (rule 92, w=32, the R′ over-unroll sweep),
`results/m8b_converge_grid_20260622T120355_*` (replication grid rule∈{13,78,92}×w∈{24,32}), and
`results/m8c_converge_fair_*` (the FAIR-supervision re-analysis that isolates tying from the step-aligned
DS confound flagged in adversarial review — adds `untied_stepDS` and the clean `nods−untied` delta).
94 tests after the M8c hardening (assert the fixed point; parametrize the test over rules 13/78/92).

**M8 — over-unrolling DECAYS even on a convergent target (the adaptive-compute headline FAILS):**

| R′ (test unroll) | trm_stepDS | trm_prog_step | trm_nods | ff_matched | untied_matched |
|---|---|---|---|---|---|
| 6 (trained) | 0.922 | 0.919 | 0.896 | 0.920 (flat) | 0.834 (flat) |
| 8  | 0.909 | 0.905 | 0.884 | 0.920 | 0.834 |
| 12 | 0.885 | 0.878 | 0.862 | 0.920 | 0.834 |
| 16 | 0.871 | 0.860 | 0.847 | 0.920 | 0.834 |
| 24 | 0.857 | 0.842 | 0.828 | 0.920 | 0.834 |

**M8b — the trained-depth EXACT-MATCH picture, replicated across the grid (10 seeds; maj baseline 0.562):**

| cell | stepDS EM | ff EM | untied EM | Δ(stepDS−ff) EM | Δ(stepDS−untied) EM | Δ(stepDS−ff) acc |
|---|---|---|---|---|---|---|
| rule13,w24 | 0.403 | 0.307 | 0.100 | +0.096 (8/2, p=.11 ns) | +0.303 (10/0, p=.002) | +0.003 ns |
| rule13,w32 | 0.102 | 0.101 | 0.038 | +0.001 (ns) | +0.064 (10/0, p=.002) | −0.007 (1/9, p=.02 *worse*) |
| rule78,w24 | 0.427 | 0.311 | 0.126 | **+0.116 (10/0, p=.002)** | +0.301 (10/0, p=.002) | +0.005 (10/0, p=.002) |
| rule78,w32 | 0.169 | 0.121 | 0.037 | **+0.048 (9/1, p=.021)** | +0.132 (10/0, p=.002) | +0.002 (ns) |
| rule92,w24 | 0.418 | 0.331 | 0.115 | **+0.087 (9/1, p=.021)** | +0.303 (10/0, p=.002) | +0.003 (ns) |
| rule92,w32 | 0.162 | 0.112 | 0.037 | **+0.050 (10/0, p=.002)** | +0.125 (10/0, p=.002) | +0.002 (ns) |

**Reading (per §2/§8).**

1. **Adaptive computation FAILS — and it FALSIFIES the M7 non-convergence hypothesis.** Over-unrolling
   the loop **decays** (stepDS 0.922→0.857 as R′ 6→24) **even though the target is a genuine fixed
   point**. The loop does not hold the fixed point and cannot turn extra test-time steps into solving
   the deep tail. Critically, M7 had *hypothesized* the rule-30 over-unroll decay was caused by CA
   being non-convergent — M8 shows the decay happens on a CONVERGENT target too, so it is **intrinsic
   to how the loop bakes in its trained depth, NOT a property of task convergence.** The M7
   rationalization is dead (correctly, it was labelled untested). Adaptive compute is not a free lunch
   the loop can exploit here.
2. **A robust, properly-isolated TYING-POSITIVE: the weight-tied loop beats a FAIR untied stack on
   whole-row coherence, 6/6 cells, at EQUAL supervision (M8c).** [M8b first reported this through
   `trm_stepDS` vs `untied_matched` — a §8 confound, since stepDS got step-aligned DS (6 targets/batch)
   and untied got only final loss; flagged by adversarial review.] **M8c** (`m8c_converge_fair.yaml`)
   computes the clean isolation: `Δ(trm_nods − untied_matched)` (both final-loss) AND
   `Δ(trm_stepDS − untied_stepDS)` (both step-aligned — `untied_stepDS` = the fair untied stack *given*
   the same step-aligned DS). Both are positive **10/0, p=.002 in ALL 6 cells**: nods−untied EM +0.05…
   **+0.37** (token-acc +0.06…+0.09), stepDS−untied_stepDS EM +0.07…+0.30. So tying buys whole-row
   coherence at equal supervision, robustly — the cleanest tying-positive in the project (parity was
   tying-neutral, M4/M6a). It is **tying, not depth**: `untied_matched` is *deep* (6 blocks) yet WORST
   on EM everywhere — depth alone doesn't buy coherence, the weight-tied reuse does.
3. **The loop beats BOTH controls only in 3/6 cells (all w=24), and only on EM — the w=32 "ff win" was
   SUPERVISION, not architecture.** The clean loop-beats-both is `trm_nods` (equal supervision) >
   *both* ff and untied: it holds at **w=24 for all 3 rules** (nods−ff EM +0.11…+0.17 p≤.021;
   nods−untied EM +0.32…+0.37 p=.002) but **fails at w=32**, where nods−ff is ns/negative on EM and
   *significantly negative on token-acc* (−0.015…−0.024, p=.002 — the plain loop LOSES to wide shallow
   ff at w=32). M8b's "4/6 via stepDS" over-counted: at w=32 only the step-DS arm clears ff, i.e. that
   half was carried by richer supervision the controls weren't given, not by the architecture. Honest
   count: **clean loop-beats-both = 3/6 (w=24); robust tying-over-untied = 6/6.**
4. **Progressive loss still adds nothing** (Δ(prog_step − stepDS) ns in all 6 cells, M8b) — consistent
   with M7/M7b: the progressive detach is inert wherever measured.

**Net (corrected after M8c).** A layered result. The headline angle (adaptive computation) is a **clean
negative** that **kills the M7 hypothesis** (over-unroll decay is intrinsic, not convergence-related).
But pursuing it surfaced a genuine, properly-isolated **tying-positive**: the weight-tied loop beats a
*fair untied stack* on whole-row coherence in **all 6 cells at equal supervision** (the cleanest such
result in the project; parity was tying-neutral). The stronger "loop beats **both** controls" claim is
real but **narrower than M8b stated** once the supervision confound is removed: clean (plain loop, equal
supervision) it holds in **3/6 cells, all w=24, on exact-match only**. So there *is* a regime
(fixed-point multi-output, whole-row metric, w=24) where the loop beats both param-matched controls — a
concrete counterexample to "the loop never beats both anywhere" — but it does not generalize to w=32
(where the plain loop loses to wide shallow ff) and is EM-only. It does not satisfy the literal §9 gate
(names Tasks A/B).

**Caveats (adversarial-proofed; S1/S2 resolved by M8c).** (i) **EM-only**; on token-acc the loop ties
ff at w=24 and *loses* to ff at w=32 — not a both-metric dominance. (ii) **Width-dependent**: clean
loop-beats-both is w=24 only (3/6); at w=32 the ff-axis win needs step-aligned DS the controls weren't
given. (iii) The robust claim is **tying > fair untied (6/6)**, isolated at equal supervision two ways
(M8c) — *that* is the defensible architectural finding; "beats both" is the narrower corollary.
(iv) **`untied_matched` is +2.5%/+3.1% OVER budget** (width-quantization, flagged `within_tol=False`) —
but it loses, so over-budget is conservative; the loop>ff comparisons are budget-clean (ff ≤0.2%).
(v) One model size, one operator family (majority-type converging ECAs), n_train=4000. (vi) The
mechanism is whole-row coherence from recurrence/tying, **not** adaptive computation — do not conflate.

---
