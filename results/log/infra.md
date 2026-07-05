# Infra — Training/eval performance (no scientific change). Bit-identical, ~2.5× faster.

Not a milestone — a perf pass on the model/training/eval path. **All run outputs are byte-for-byte
unchanged** (verified: parity single-output and iterated multi-output cells reproduce prior
accuracies and exact-match exactly; 67/67 tests pass; ruff clean).

Four bottlenecks resolved:

1. **Data path dominated wall-clock.** For the tiny models here the per-sample
   `Dataset.__getitem__` + default-collate path of `torch.utils.data.DataLoader` cost more than
   the matmuls. Replaced with `InMemoryLoader` (`src/looptab/data/dataset.py`): the RAM-resident
   dataset is stacked into tensors once and batched by slicing a permutation. Determinism is
   preserved **bit-for-bit** by reproducing `DataLoader`'s exact per-epoch global-RNG protocol —
   the `_BaseDataLoaderIter` worker `_base_seed` draw *and then* `RandomSampler`'s seed draw → fresh
   `Generator` → `randperm` — so both the consumed RNG state and the batch composition match the
   loader it replaces (checked against a real `DataLoader` over multiple epochs).
2. **Redundant eval forward pass.** On multi-output (Task B) cells, `accuracy` and `exact_match`
   each ran their own forward over the test set (and once per R' in the extrapolation harness).
   Added `evaluate` (`src/looptab/eval/metrics.py`) which derives both from a single `_predict`;
   `run_point` and `run_extrapolation_point` now use it. Same predictions, half the eval passes.
3. **CPU thread oversubscription.** The tiny models' matmuls fall below torch's parallelization
   threshold, so torch's default intra-op thread count (= core count) adds only dispatch overhead.
   Measured (4-core box): threads 1≈2 < 4 < **8 ≈ 3× slower than 1**. On many-core cloud machines
   the default is far worse (torch defaults to the full core count). Added `TrainConfig.num_threads`
   (default **1**), applied once in `run.main()` via `torch.set_num_threads`. Verified bit-identical
   across thread counts (full-precision, both single- and multi-output) — the small kernels don't
   reorder reductions — so this is a pure speed/portability win. `num_threads: null` restores torch's
   default for when models grow.

Measured: a representative `run_point` (2 arms × 30 epochs, n_train=4000) went 7.19s → 2.85s (~2.5×)
on CPU from (1)+(2); thread pinning takes the warm loop a further ~2.83s → 2.43s here and avoids the
~3×+ oversubscription penalty on big-core boxes. Multi-output runs gain additionally from the
single-pass eval. No config, metric, or conclusion changes — this only makes re-running cheaper.

4. **Serial seed loop left cores idle.** With per-run work pinned to 1 thread (item 3), a
   multi-core CPU sat mostly idle. The per-axis-point seed loop now runs across a process pool
   (`ExperimentConfig.parallel_workers`, default **1** = unchanged serial; `run._compute_seeds`),
   each worker pinned to `train.num_threads` so workers × threads never oversubscribe. Seeds are
   pure functions of their seed and self-reseed, so it is **bit-identical** to serial (verified:
   `parallel_workers=4` reproduces serial accuracies exactly; guarded by
   `test_parallel_seeds_bit_identical_to_serial`). Measured **4.12× on a 4-core box** for a
   4-seed run; scales with cores/seeds. Also switched eval to `torch.inference_mode` (a
   strictly-faster `no_grad`; numerically identical).

Measured: a representative `run_point` (2 arms × 30 epochs, n_train=4000) went 7.19s → 2.85s (~2.5×)
on CPU from (1)+(2); thread pinning (3) takes the warm loop a further ~2.83s → 2.43s and avoids the
~3×+ oversubscription penalty on big-core boxes; seed-parallelism (4) adds ~Ncores× on multi-seed
runs (4.12× measured on 4 cores). Multi-output runs gain additionally from the single-pass eval. No
config, metric, or conclusion changes — this only makes re-running cheaper. **Set `parallel_workers`
to the core count on any ≥5-seed sweep/grid to use the cores; it stays off (1) by default.**

**Model-level changes investigated and REJECTED (negative result, §8).** A pass looking for
faster *model math* found nothing worth landing — the TRM core is tiny and already minimal, so its
cost is the irreducible matmul forward/backward, not removable Python overhead. Measured on
representative configs (d∈{20,40,80}, steps 4–8, threads=1):
  - *Precompute the constant `X` projection out of the weight-tied loop* (mathematically the same
    reassociation of the first linear): **1.01–1.05×**, and **not** bit-identical (maxdiff ~1e-7
    from FP reassociation → would force re-baselining every committed result). Reject.
  - *Batch deep supervision into one `cross_entropy` over stacked per-step logits*: **0.98–0.99×
    (slightly slower** — the `stack`+`expand` cost cancels the fewer-call saving), and not
    bit-identical. Reject.
  - *Functional forward* (`F.linear`/`F.gelu` instead of `Module.__call__`, skipping hook checks):
    **bit-identical (maxdiff 0.0)** but only **1.01–1.04×** — not worth the readability cost of
    reaching into `update_net` internals on the canonical model. Reject.
So the model is left as-is; the wins all live at the harness level (1)–(4). Don't re-litigate these
without first changing the regime (much larger models, or accepting a numerics re-baseline).

---
