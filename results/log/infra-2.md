# Infra-2 — GPU acceleration track (device-resident data, AMP/compile, CUDA graphs, fused mixer kernel). No scientific change.

Not a milestone — a second perf pass (after `infra.md`'s CPU-side pass), this time on the
GPU path. **No conclusion in `results/` or CLAUDE.md §11.2 changes**: every numeric claim in
this file is either bit-identical to the eager fp32 baseline or, where it isn't (AMP/compile),
was already re-verified against the headline Δs it could have disturbed (see §5 below) and
found to preserve sign and stay well inside seed noise. This file exists to give the perf work
its own citable narrative, matching CLAUDE.md §11.1's GPU section and §11.2 #16–#17, which
already carry the detailed numbers — this is the consolidated record `results/LOG.md` was
missing.

## 1. Device-resident dataset (`make_loaders(..., device=...)`)

`train.device: cuda` previously copied each batch host→device inside the training loop. The
loader now parks the whole dataset on the GPU once and gathers batches there; the batch
permutation is still drawn on the CPU generator, so batches are **bit-identical** to the CPU
path (verified in `tests/test_dataset.py` and by reproducing committed run output exactly).
Host→device copies were only ~0.9% of runtime, so this is a modest, honestly-sized **~1.02x**
on fp32 (~1.10x under AMP) — free, but it stops the copy becoming the floor if compute gets
cheaper elsewhere in this file.

`parallel_workers` still helps on GPU (a microbenchmark that spawned a fresh process per model
suggested otherwise — 0.40–0.51x — but that was CUDA-context-startup-per-process, an artifact
of the benchmark, not the runner; the real `ProcessPoolExecutor` keeps workers alive across
seeds). Clean A/B on the M23 mixer sweep: `parallel_workers: 3` = 8m26s vs `1` = 9m37s (1.14x).

## 2. `train.amp` (fp16 autocast + GradScaler)

Opt-in, off by default, CUDA-only, standard-train-path only (raises on curriculum/ACT/N_sup/
contraction rather than silently skipping). Training only — **eval always runs fp32**, so
metrics stay comparable across arms. Not bit-identical, so it must be applied uniformly across
every arm in an experiment (a uniform shift cancels in the reported Δ; a per-arm one would not).

**Not a free win — the size of the GEMM decides the sign of the speedup, not the model.**
Measured across three regimes:
- Sudoku mixer sweep (256×36 rows/GEMM): **1.18x** (7m08s vs 8m26s), Δs agree to ±0.0001.
- ETTh1 forecasting (128×7 rows/GEMM): **0.74x — 35% SLOWER** (6m22s→8m49s), numerics still fine
  (largest Δ shift 0.0042 vs ±0.027–0.074 seed stds), every arm neutral-or-slower including the
  widest (`trm_mixer` 0.86x, `trm_decoupled` 0.81x) — so it's row-count, not arm width.
- `yeast` multilabel/F1 (small per-cell width, 10-fold CV): **0.65x**, also slower; one
  sign-test call (`Δ(trm−ff_matched)` accuracy) moves from raw p=0.021 to a near-tie-flagged
  robust p=0.070 under AMP's fp16 rounding, without reversing.

Working hypothesis: below some `batch × n_cells` row count the fp32↔fp16 cast traffic costs
more than Turing's tensor cores save (Sudoku's 9216 rows wins, ETTh1's 896 and yeast's small
width lose). Two-and-a-bit data points — treat this as a hypothesis, not a rule. **The
actionable takeaway is procedural: time `amp` on your own config before reaching for it.**

## 3. `train.compile` (`torch.compile` per arm)

Needs Dynamo-capable torch + Triton (no Windows wheels for stock Triton — `triton-windows`
works). Costs a per-process compile warmup, amortised by Inductor's on-disk cache across a
sweep. On the Sudoku mixer sweep, `compile+amp` on torch 2.13 reached **1.74x** (4m50s vs
8m26s reference) with Δs agreeing to ±0.0001/±0.002. On ETTh1 forecasting it's a wash
(1.04x compile-only, 1.01x compile+amp) — consistent with §2's small-GEMM hypothesis:
neither optimization has enough work per kernel to pay for its own overhead there.

Also fixed in this window: a `UnicodeEncodeError` on Windows (`sys.stdout` falling back to
cp1252 under redirection, nondeterministically) was silently dropping every remaining Δ line
after the first non-ASCII character (`Δ`, `±`, `−`) mid-sweep. `run.py::main()` now
force-reconfigures stdout/stderr to UTF-8 unconditionally at entry; `tests/test_run.py`
unaffected (38/38 pass).

## 4. `train.cuda_graph` (opt-in, CUDA-only)

Captures the whole training step — zero-grad, forward, backward, optimizer step
(`capturable=True` AdamW) — into a CUDA graph and replays it. Motivated by profiling ETTh1's
`trm_mixer` step: only ~4.5ms of a 21ms eager step was GPU compute, the rest was ~590
kernel-launch dispatches. This reframes §2/§3's "small-GEMM configs don't benefit from
AMP/compile" finding rather than contradicting it — those configs are **launch-bound, not
compute-bound**, and CUDA graphs are the lever that targets launch overhead specifically.

- Isolated: **~4.9–5.3x** on ETTh1's shape.
- End-to-end through the real runner (6 arms, 3 seeds, `python -m looptab.run`): **165s→35s
  (4.71x)**, Δs preserved within seed noise.
- Reverses nothing about large-GEMM configs: an earlier speed-knob search had already rejected
  CUDA graphs as a 1.00x no-op at Sudoku scale (large GEMMs, already compute-bound). Both
  findings are correct for their own shape — measure per config.
- `torch.compile(mode="reduce-overhead")` captures most of the same win (~6x vs default mode's
  ~2x) because it also uses CUDA graphs internally — the lever is the graph capture, not Triton
  fusion.
- Mutually exclusive with `amp` for now; standard train path only (same guard pattern as `amp`
  for curriculum/ACT/N_sup/contraction).

## 5. Fused mixer CUDA kernel (`trm_mixer_fused` / `models/csrc/fused_token_mix.cu`)

`trm_mixer` with the token-mixing step dispatched to a hand-written kernel instead of eager
transpose+Sequential+transpose. **Numerically exact** (gradcheck-verified), not a precision
tradeoff — but compiled for a closed set of `(n_cells, token_hidden)` shapes (Sudoku 36/64,
ETTh1 7/8, weather 21/8, plus 6/6 for tests); an unsupported shape raises loudly at
construction. CUDA-only, JIT-compiled lazily on first use (importing the package never needs
the toolchain).

On ETTh1's shape, once both are wrapped in a CUDA graph, the fused kernel edges out
`torch.compile`'s best mode by single digits, not an order of magnitude: eager 21ms →
`compile` default 10.6ms (2.0x) → `compile reduce-overhead` / eager+manual-graph ~3.2–4.0ms
(~5–6x) → fused-kernel+graph ~3.18ms (consistently fastest across 4 repeated runs, but a small
margin over `compile`'s best mode — the headline win is §4's CUDA graph, this is a modest
addition on top).

**Correctness lesson, generalizable beyond this kernel.** The first integration attempt
trained "successfully" under `cuda_graph` — no crash, no NaN, parameters visibly updating —
but the loss oscillated instead of converging. Root cause: the kernel launched on the implicit
legacy default stream (bare `<<<grid,block>>>`) instead of `at::cuda::getCurrentCUDAStream()`.
Invisible in eager use (PyTorch's current stream is usually the default stream there), but
`torch.cuda.graph()` capture redirects "current stream" to a dedicated capture stream, so the
kernel raced against the rest of the captured graph instead of being ordered against it — a
plausible-but-wrong gradient, not a crash. **Any custom CUDA extension intended to be
graph-capturable must launch on the current PyTorch stream, not the implicit default one.**
`tests/test_fused_mixer.py::test_fused_kernel_composes_with_cuda_graph` guards against
regressing this — it asserts genuine convergence under `cuda_graph`, not just "ran without
error."

Also incompatible with `amp`: the extension hardcodes fp32 with no autocast registration, so
under `train.amp=true` it would silently train at a different precision than the other arms'
autocast `nn.Linear` ops — a per-arm precision difference smuggled into the reported Δ.
`run.py` now rejects `trm_mixer_fused` + `amp=true` loudly (same guard pattern as the other
`amp` incompatibilities).

## Net effect

Training on small-cell, launch-bound configs (ETTh1-shaped forecasting) is now **~4.7x**
faster end-to-end via `cuda_graph`. Training on large-GEMM, compute-bound configs
(Sudoku-shaped mixer sweeps) is now **~1.7x** faster via `compile+amp`. Neither knob is a
universal win — §2/§3/§4 each found the opposite sign on the other one's best-case shape — so
none of them are on by default, and every arm-comparing experiment must apply them uniformly
or not at all. This is what M34 (forecasting dataset breadth) and the synthetic scale-up
milestone lean on to make previously CPU-prohibitive sweeps affordable.

## 6. M34 high-channel follow-up (8GB / Windows-WDDM)

M34 widened the operating range from 7--21 channels to electricity (321) and traffic
(862). That exposed three bottlenecks which the small forecasting configs could not reveal:

- **Activation memory, not launch latency, dominates the large-channel mixer.** Full fp32 CUDA
  graphs are actively harmful here because their private pool competes with activations on an
  8GB card (electricity batch 128: eager ~6.3s/step, graph ~18.5s/step). AMP is the correct
  precision on this Turing GPU, but traffic's h1760 mixer still cannot fit batch 128. The standard
  train path now supports exact effective-batch gradient accumulation via `microbatch_size`, with
  an optional per-arm override: M34 retains batch 128 / one optimizer update while using 64-row
  microbatches for electricity and 16-row microbatches for the largest traffic arm. Measured
  effective-batch training is roughly ~0.52s for electricity and ~2.5s for traffic, versus an OOM
  or allocator-spill cliff on the unsliced batch. `cuda_graph` + microbatching is rejected for now;
  graph capture gave no material gain on these compute-bound microbatches.
- **Dataset residency is now capacity-aware.** `make_loaders` still parks small tables on the GPU,
  but keeps combined train/test tensors above 256MiB on CPU. This preserves the old bit-identical
  device-resident fast path while protecting activation headroom for electricity (~1.0GiB dataset)
  and traffic (~2.5GiB). Batches use the existing explicit `.to(device)` staging path.
- **Forecast windows are selected before materialization.** The old implementation constructed
  every possible window and sliced afterward; on traffic that meant a ~7GiB temporary for a run
  which retains only 7,000 rows. It now gathers only the requested chronological starts, preserving
  shapes, values, fold mapping, standardization and no-leakage semantics. Small split construction
  is ~0.30s for traffic and ~0.15s for electricity (then faster from a hash-verified series cache).
- **Allocator cache is cleared only at runner phase boundaries.** Cached AMP training workspaces
  could coexist with the differently-shaped fp32 evaluation batch and force WDDM shared-memory
  spill. On the fixed one-epoch traffic smoke run, train+two-arm evaluation fell from **73s to
  9.4s (7.8x)** after cache boundaries; `empty_cache` is never called per batch and cannot discard
  live parameters/datasets.
- **Sudoku uniqueness checks keep their recursive hot state in flat Python lists with cached
  topology and integer `bit_count`.** Puzzle bytes/tie-breaking are unchanged, while 9x9 generation
  is roughly **3--4x faster** across the M34 givens range.

The resulting dispatch rule is shape-specific: keep full-step fp32 CUDA graphs for launch-bound
ETT-scale jobs; use AMP + per-arm microbatching (without graphs) for high-channel electricity/
traffic. The optimizer batch, update count, model/control budget and evaluation precision remain
unchanged.
