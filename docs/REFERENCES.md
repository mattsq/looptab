# References and mechanism pointers


- **TRM** — Jolicoeur-Martineau 2025, arXiv 2510.04871. Base the recurrent core on this:
  single tiny 2-layer net, iterative answer refinement via a latent. Simpler/stronger than HRM.
- **HRM** — Wang et al. 2025, arXiv 2506.21734. The H/L hierarchy + 1-step-gradient + ACT.
- **ARC-Prize HRM autopsy** — arcprize.org/blog/hrm-analysis. *Why we test the loop before
  the hierarchy.*
- **Deep Equilibrium Models** — Bai, Kolter, Koltun 2019, arXiv 1909.01377. Fixed-point /
  1-step gradient that HRM's training rests on.
- **Universal Transformers** — Dehghani et al. 2018, arXiv 1807.03819. Weight-tied depth
  recurrence + adaptive compute, the archetype.
- **ACT** (Graves 2016, arXiv 1603.08983) / **PonderNet** (Banino et al. 2021,
  arXiv 2107.05407). Halting machinery for the later halting knob.
- **"Can you learn an algorithm?"** — Schwarzschild et al. 2021, arXiv 2108.06011. The
  easy-to-hard extrapolation logic behind Task B.
- **Grinsztajn et al. 2022** — arXiv 2207.08815. The three tabular failure modes Task A is
  built to probe.
- **Gorishniy "Revisiting DL Models for Tabular Data"** — 2021, arXiv 2106.11959. Baseline
  architectures (FT-Transformer / tuned MLP) for when real tabular arrives.

### Depth-extrapolation / "transferable step operator" mechanisms (added for the post-M6a mechanism hunt)

These target our central null — over-unrolling decays to baseline and OOD depth collapses (M1/M3b);
"the loop does not settle a stable fixed point." All are tiny-model, synthetic-algorithmic-task
results (no downloads), so they fit §5 tiny-first and the §4 control contract.

- **Deep Thinking nets ("End-to-end Algorithm Synthesis…: Logical Extrapolation Without
  Overthinking")** — Bansal, Schwarzschild, Borgnia, Emam, Huang, Goldblum, Goldstein 2022,
  arXiv 2202.05826 (NeurIPS'22). *The direct fix for our exact failure.* Two ingredients:
  **(1) recall** — concatenate the input to the recurrent module's input at *every* step so it
  can't be forgotten (our TRM already does a form via `cat[X, z, a]`); **(2) progressive loss** —
  per batch pick random `n, k` with `n+k < T_max`, run `n` steps with **gradients detached**, then
  `k` steps **with** gradients, and apply the loss on that output (combined with the usual
  max-iteration loss). This penalizes iteration-count-specific behavior and pushes the loop toward a
  *repeatable* step operator / steady state — i.e. it directly attacks "overthinking." Code:
  github.com/aks2203/deep-thinking. **Top candidate for the next milestone.**
- **Path Independence in equilibrium models** — Anil, Pokle et al. 2022, arXiv 2211.09961
  (NeurIPS'22). Shows upward (harder-than-trained) generalization *correlates with* path
  independence (convergence to the same attractor regardless of init/over-unroll); interventions
  that promote it improve extrapolation, those that penalize it hurt. Gives a **measurable
  diagnostic** (path independence on OOD samples ⇒ accuracy) and a regularizer to try.
- **Looped Transformers for Length Generalization** — Fan, Du, Ramchandran, Lee 2024,
  arXiv 2409.15647. Weight-tied looping + an **adaptive step count** length-generalizes on
  algorithmic (n-RASP-L) tasks where required depth grows with input length — the architectural
  cousin of our Task B depth-extrapolation claim; their adaptive-halting tie-in is the alternative
  to progressive loss.
- **Rethinking Deep Thinking (Lipschitz-constrained stable algorithm learning)** — 2024,
  arXiv 2410.23451. Stabilizes the deep-thinking recurrence with Lipschitz constraints to curb
  overthinking — a follow-up lever if progressive loss alone is unstable at our scale.
- **Recurrent-depth latent reasoning ("Scaling up Test-Time Compute…")** — Geiping et al. 2025,
  arXiv 2502.05171 (NeurIPS'25 spotlight). Large-scale evidence that iterating a recurrent block to
  arbitrary test-time depth scales reasoning; the per-token adaptive-compute view. Context for why
  a transferable operator matters; out of our tiny-first scope to replicate, in scope to borrow from.
  Also the source of the **latent "orbits"** observation our M21 introspection echoes (a shared
  init-independent limit set that never settles to a fixed point).

### Introspection / stability diagnostics (the M21 toolkit + the architectural-refinement levers)

These ground the M21 measurement-only suite and the (unbuilt, evidence-gated) `trm_stable` follow-up.

- **Stabilizing Equilibrium Models by Jacobian Regularization** — Bai, Koltun, Kolter 2021,
  arXiv 2106.14342. Jacobian spectral radius ρ(J) is the stability condition for a fixed-point
  iteration (ρ<1 ⇒ contraction); regularizing it (Hutchinson trace estimator) stabilizes DEQ
  training. The headline M21 metric and the first refinement lever.
- **Path Independent Equilibrium Models** — Anil, Pokle et al. 2022, arXiv 2211.09961. Already cited
  above for the diagnostic; *also* the intervention lever — training that promotes path independence
  improves upward generalization, training that penalizes it hurts. The M21 `za_alignment` is its
  Asymptotic-Alignment score.
- The Jacobian-spectral / Lipschitz refinement argument rests ENTIRELY on the three verifiable refs
  above (2106.14342, 2410.23451, 2211.09961) — do not let it depend on any single citation.
  A newer looped-LM template, **"STARS / Stabilizing Recurrent Dynamics…" (arXiv 2605.26733)**, was
  found via web search (Jacobian-Spectral-Radius Regularization + random-loop sampling; pre-norm
  "grow without bound" vs post-norm "settle into poor states"); it is an OPTIONAL pointer whose ID is
  a 2026 arXiv number this repo cannot verify offline — cite it as supplementary, not load-bearing.
