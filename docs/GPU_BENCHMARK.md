# GPU benchmark checklist (M35 speed substrate)

The M35 knobs were built and correctness-tested in a **CPU-only** container. Everything here is
about turning "built" into "measured" on the RTX 2070, so §11.2 can gain a real entry instead of
the hypotheses currently parked in §11.3.

Read this first: **the repo has twice learned that these knobs flip sign by config shape**
(§11.2 #16 — CUDA graphs ~5x on small cells but a 1.00x no-op at sudoku scale; #18 — the same
graphs ~3x SLOWER at M=321). So the deliverable is not one number. It is a small table of
*which mode wins at which shape*, with the losers recorded too.

## 0. Preconditions

```powershell
uv sync --extra gpu
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest -q          # expect the CUDA-marked tests to RUN now, not skip
```

The CUDA-only tests are the first real signal — they are skipped on CPU, so this is the first time
they execute. In particular these three must pass before any timing matters:

| test | what a failure means |
|---|---|
| `TestCudaGraphOnGPU::test_amp_plus_cuda_graph_trains` | the static-loss-scale AMP capture is wrong; do not trust any amp+graph timing |
| `TestDeepSupervisionGraphsOnGPU::test_ds_graph_trains_and_matches_eager_ballpark` | the DS pass capture diverges from eager — the #17 failure class (plausible-but-wrong gradients, no crash) |
| `TestCudaGraphOnGPU::test_tail_graph_covers_the_ragged_batch` | mid-training capture is corrupting optimizer state |

`test_amp_static_scale_overflow_raises_actionably` should also pass; it deliberately overflows fp16
to prove the epoch-boundary check catches what `GradScaler` normally would.

## 1. Timing protocol (applies to every row below)

- Time the **whole runner**, not an isolated microbenchmark. §11.1 already records one case where a
  microbenchmark said GPU workers were a 0.40x pessimisation and the real A/B said 1.14x.
- Run each mode **twice** and keep the second (CUDA context / Inductor cache warmup).
- Compare **Δs, not just wall clock**. A speed knob that shifts a Δ beyond seed noise is a
  correctness problem, not a speed result. The existing bar: Δ agreement to ~±0.001 against seed
  stds of ±0.003–0.07.
- Record the losers. "cuda_graph was 2.8x slower here" is exactly as useful as a win, and is what
  stops the next agent re-deriving it.

## 2. The matrix worth running

### 2a. DS-family capture — the main new lever (expected: biggest win)

`n_sup` passes per batch multiply the launch count, which is what #16 says graphs eliminate. This
is the change most likely to matter.

```powershell
# baseline, then each knob, uniformly across arms
uv run python -m looptab.run --config configs/experiments/m18e_compute_matched.yaml
# then re-run with train.cuda_graph: true, then amp: true, then both
```

| config | mode | wall clock | Δ(headline) | notes |
|---|---|---|---|---|
| `m18e_compute_matched` | eager (reference) | | | |
| | `cuda_graph` | | | expect the largest gain |
| | `amp` | | | small GEMMs → may lose, per §11.3 |
| | `amp`+`cuda_graph` | | | inherits amp's penalty |

Also worth one row on an ACT config (`m23_sudoku_act_sweep`) since `train_act` captures the halting
BCE inside the graph and sudoku is the compute-bound shape where #16 found graphs inert — a
plausible null, and worth having on record as one.

### 2b. Sync removal — measure it in isolation

This one is bit-identical, so it is a pure timing question and needs no Δ check. Compare the M35
tree against the pre-M35 commit on the SAME config (an `n_sup` config maximizes the effect, since
the removed `.item()` was per pass). If this shows ~0 on a compute-bound config, that is the
expected and reportable answer.

### 2c. `speed: auto` — does the autotuner pick what a manual sweep says wins?

The autotuner is only trustworthy if its pick matches the answer from 2a. Run:

```powershell
# train: { speed: auto }
uv run python -m looptab.run --config configs/experiments/m18e_compute_matched.yaml
```

Check the `[speed:auto]` lines and the `speed_modes` field in the run record against your 2a table.

**Verify the re-seed while you are here.** Probing trains throwaway models, which consumes the
global RNG that `InMemoryLoader` draws each epoch's shuffle from; `run_point` re-seeds and rebuilds
the arm afterwards so the real run sees the stream it would have seen anyway. The CPU suite proves
the re-seed restores the stream, but only a GPU run exercises the path end to end: **an `auto` run
that lands on `eager` must reproduce the `manual` eager run's metrics exactly.** If it does not, the
re-seed is not covering something and every autotuned number is suspect.

Two further failure modes to watch for:

- **it picks a mode that 2a says is slower** → the probe is too short; raise `speed_probe_steps`.
- **different arms get different modes** → expected and allowed, but it means the run's Δ mixes
  precisions. Pin the winner manually before treating such a run as a result.

Also time the probe overhead itself: `auto` pays four short training runs per arm up front. On a
short config that could exceed what it saves.

### 2d. High-channel regression check — does #18 still hold?

The point here is to confirm the new knobs do **not** quietly make electricity/traffic worse, since
`speed: auto` will now happily try graphs there.

```powershell
uv run python -m looptab.run --config configs/experiments/m34_electricity_h24.yaml   # AMP+microbatch recipe, as committed
```

Expected: `cuda_graph` remains harmful (that is #18), `speed: auto` should therefore *reject* it —
and if it does not, the probe is not modelling the memory pressure, which is worth writing down as
a limitation of the autotuner rather than a reason to distrust #18.

### 2e. `cuda_graph_tail` — correctness-adjacent, not a speed knob

It stops dropping the ragged final batch, so it **changes which rows are trained on**. Worth one
run to confirm the second capture does not blow the memory budget, and to see how much the dropped
tail was actually worth (on `_small_loader`-shaped runs it is 8 of 200 rows; on real configs it is
usually negligible, which is the useful thing to establish).

## 3. What to write up

Add one `results/log/m35.md` narrative plus a §11.2 entry stating, per shape:

- which mode won, by how much, and **which lost** (with numbers);
- whether every Δ reproduced within seed noise (if not, stop and treat it as a bug);
- whether `speed: auto`'s pick agreed with the manual sweep;
- the dispatch rule a future agent should follow, in one line, in the style of #18's
  "CUDA graphs for small-cell launch-bound jobs, AMP+microbatching for high-channel jobs".

If a knob turns out to be a no-op or a harm on every shape you test, say so plainly and add it to
§11.4's closed-levers list. A clean negative here is worth as much as a win — that is the same
standard §8 sets for the research results.
