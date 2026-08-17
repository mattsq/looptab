"""Pydantic config models. A single config + seed fully determines a run.

An experiment is a list of `arms` (each an independent model spec with a label)
trained on a task, optionally swept over one task parameter (e.g. parity `k`).
Reporting `deltas` between named arms is what satisfies the prime directive: every
result is a Δ between a recurrent arm and a matched control — and, critically, the
deep-supervision ablation is *its own arm* so the loop and deep supervision are never
confounded (CLAUDE.md §4/§8).
"""

from __future__ import annotations

import itertools
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskConfig(BaseModel):
    name: Literal[
        "linear", "parity", "multi_parity", "iterated", "converge", "hopfield", "mixed_converge",
        "nested_converge", "disruption", "multilabel", "sudoku", "etth1", "weather",
        "etth2", "ettm1", "ettm2", "electricity", "traffic",
    ]
    # "classification" (default; all M0–M25 tasks) trains with cross-entropy and reports
    # accuracy/EM/F1. "regression" (M26 forecasting) trains with MSE and reports MSE/MAE/R².
    objective: Literal["classification", "regression"] = "classification"
    params: dict = Field(default_factory=dict)
    n_train: int = 4000
    n_test: int = 1000
    # Base seeds. The runner offsets these per outer seed so that the variance
    # band reflects *function-level* variation too (a new task_seed per seed),
    # while train/test still share the same task_seed within a seed (CLAUDE.md §3).
    task_seed: int = 0
    train_sample_seed: int = 1
    test_sample_seed: int = 2


class ModelConfig(BaseModel):
    name: str
    label: Optional[str] = None  # name used in Δ reporting; defaults to `name`
    hidden_dim: int = 64
    latent_dim: int = 64
    n_steps: int = 4
    # Optional per-arm override for TrainConfig.microbatch_size.  Activation memory varies by
    # orders of magnitude across budget-matched parameterizations (M34 traffic: h1760 mixer vs
    # h44 distinct-weight control), so forcing the most constrained microbatch on every arm wastes
    # launches.  The effective optimizer batch remains TrainConfig.batch_size in every arm.
    microbatch_size: Optional[int] = None
    # `deep_supervision` toggles whether the TRM loop emits per-step readouts.
    # `deep_supervision_weight` is the per-arm training weight on those readouts.
    # Decoupling these (per arm) is what lets us ablate deep supervision separately
    # from the loop: a TRM arm with DS off isolates the loop alone.
    deep_supervision: bool = True
    deep_supervision_weight: float = 1.0
    # `ds_mode` selects what the per-step readouts are supervised against (M3b):
    #   "final"        — every step is pinned to the final state s_T (the M0–M3a default;
    #                    this is the DS that has been neutral-to-negative everywhere).
    #   "step_aligned" — loop step i is supervised against the intermediate CA state s_i
    #                    (requires a trajectory target and n_steps == T per batch). This is
    #                    the version that *should* fire if the loop learns a step operator.
    #   "progressive_final" / "progressive_step" — Deep Thinking progressive loss (M7, Bansal
    #                    2022): per batch run (T−k) steps with gradients DETACHED, then k steps
    #                    with gradient, supervising the k grad steps against the final state
    #                    (_final) or step-aligned against s_{T−k+1..T} (_step). Forces an
    #                    iteration-count-independent / repeatable step operator → the depth-
    #                    extrapolation lever. Requires a trajectory target; loop arms only.
    ds_mode: Literal["final", "step_aligned", "progressive_final", "progressive_step"] = "final"
    # M7: mix weight between the progressive term and the standard full-T term in the
    # progressive loss: L = alpha·L_progressive + (1−alpha)·L_full. Deep Thinking keeps both
    # (the full term anchors the model so it doesn't collapse). Only used by progressive ds_modes.
    progressive_alpha: float = 0.5

    # --- M18: TRM-faithful ingredients (all default to OFF = bit-identical to pre-M18) -------
    # 2024–26 work on looped models (TRM ablations, HRM autopsy) flags four ingredients the
    # repo lacked. Each is opt-in per arm so a single "trm_faithful" arm bundles them and the
    # Δ vs plain `trm` attributes their joint effect (CLAUDE.md §4/§8; bundle-first per the
    # plan, ablate if it moves).
    #   use_rmsnorm — RMSNorm on the latent each update (ingredient 3; stability for the loop).
    #   n_latent    — z-updates per answer update (ingredient 4; TRM uses 6). 1 = original 1:1.
    #   n_sup       — detached deep-supervision passes carrying (z,a) across them (ingredient 1;
    #                 the autopsy's active ingredient; the repo's old DS is a *different* thing).
    #                 1 = a single supervised forward (one pass; a distinct routine from `train`,
    #                 not asserted bit-identical to it). >1 routes to
    #                 train_deep_supervision (standard-train path only; errors under curriculum).
    #   ema_decay   — EMA of weights folded in for eval (ingredient 2; TRM uses 0.999). None=off.
    use_rmsnorm: bool = False
    n_latent: int = 1
    n_sup: int = 1
    ema_decay: Optional[float] = None
    # M23 — ACT adaptive-computation halting (the §4/§12 unbuilt TRM ingredient; OFF by default ⇒
    # bit-identical). When ``use_act`` (TRM only), the arm trains via ``train_act`` with ``n_sup``
    # the max segment count and a learned halt head (BCE to per-example correctness), evaluated
    # adaptively (``evaluate_act``: each example halts when solved, the rest get more segments).
    # ``halt_weight`` scales the halting-head loss relative to the task loss.
    use_act: bool = False
    halt_weight: float = 0.5
    # M23 re-test — `trm_mixer` only: width of the token-mixing MLP over the cell axis (the
    # cross-cell propagation operator). None ⇒ = n_cells. Ignored by every other arm.
    token_hidden: Optional[int] = None
    # `n_sup_carry` (review fix B1): with n_sup>1, whether the detached (z,a) is carried across
    # passes (True = the real deep-supervision mechanism) or each pass restarts fresh (False = the
    # COMPUTE-MATCHED control: same optimizer-step count, no carry). Δ(carry − no-carry) isolates
    # the carry from the raw 4× step-count. Only used when n_sup>1.
    n_sup_carry: bool = True

    # --- M27: contraction regularization (the M21 `trm_stable` lever; `trm` only) --------------
    # M21 MEASURED that the trained loop never settles a fixed point (Jacobian ρ>1, frac_expanding
    # =1.0, residual ~1.2) EVEN where it wins, and over-unrolling decays. These two OFF-by-default
    # penalties push the one-step latent map F(z)=update(cat[X,z,readout(z)]) toward contraction so
    # we can test whether a CONTRACTIVE loop finally extrapolates / uses test-time compute — judged
    # against BOTH the extrapolation metric AND the coherence Δ it costs (a trm-only, param-
    # identical arm, so budget parity vs the plain `trm` control is exact). Both default 0.0 ⇒ the
    # routine is never entered and every committed M0–M26 result is bit-identical.
    #   jac_reg_weight   — DEQ Jacobian penalty (Bai 2021 arXiv 2106.14342): weight on a Hutchinson
    #                      estimate mean‖J v‖² of the one-step map's amplification (drives ρ↓).
    #   fixed_point_weight — path-independence / fixed-point residual (Anil 2022 arXiv 2211.09961):
    #                      weight on the normalized step residual ‖z_{t+1}−z_t‖/‖z_t‖ (→0).
    #   n_reg_steps      — outer steps rolled (detached) to the linearization / residual point.
    jac_reg_weight: float = 0.0
    fixed_point_weight: float = 0.0
    n_reg_steps: int = 4

    def resolved_label(self) -> str:
        return self.label or self.name


class TrainConfig(BaseModel):
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    # Optional activation-memory bound while preserving `batch_size` as the EFFECTIVE optimizer
    # batch.  The standard train path slices each loader batch into microbatches, weights each
    # mean loss by microbatch_size/full_batch_size, accumulates gradients, then performs exactly
    # one optimizer/EMA update.  This is essential for M34's high-channel forecasting shapes on
    # the 8GB RTX 2070: traffic batch=128 does not fit even under fp16, while microbatch=16 does.
    # None = the pre-existing one-pass batch path.  Incompatible with cuda_graph for now.
    microbatch_size: Optional[int] = None
    device: str = "cpu"
    # CPU intra-op thread count (applied once by the runner). The models here are tiny, so
    # their matmuls fall below torch's parallelization threshold: extra threads add only
    # dispatch overhead. Measured on this regime, 1 thread is fastest and oversubscription
    # (threads ≫ work) is *catastrophic* — e.g. 8 threads ran ~3× slower than 1. This bites
    # hardest on many-core cloud boxes, where torch otherwise defaults to the full core count.
    # Bit-identical to other thread counts (the small kernels don't reorder reductions), so
    # pinning is a pure speed/portability win, not a numerical change. `None` leaves torch's
    # default untouched (set this if you ever scale the models past the tiny regime).
    num_threads: Optional[int] = 1
    # --- Speed knobs. BOTH default OFF and are bit-identical when off; both CHANGE NUMERICS
    # when on, so they are opt-in and a whole experiment must set them uniformly (every arm on
    # the same path) — a uniform shift cancels in the Δ the repo reports, a per-arm one does not.
    #
    # `amp`: fp16 mixed precision for TRAINING only (autocast + GradScaler); evaluation always
    # runs in fp32, so metrics stay directly comparable. CUDA-only (silently inert on CPU, which
    # has no fp16 tensor cores). Turing/Ampere+ give ~2x on the GEMMs; measured ~1.1-1.2x
    # end-to-end here. Supported on the standard train path AND (since the DS-family pass came
    # under capture) on `train_deep_supervision` / `train_act` / `train_stable`; only the
    # trajectory CURRICULUM routines still raise, because they resample the unroll depth per
    # batch. On `train_stable` the contraction penalties stay fp32 deliberately (the Jacobian
    # probe's whole point is small-magnitude fidelity) — see that routine's docstring.
    #
    # `compile`: wrap each arm in `torch.compile`. Measured ~1.9-2.2x with amp (the largest
    # single lever), but it needs a torch new enough for this interpreter (torch 2.2 + Python
    # 3.12 raises "Dynamo is not supported") AND Triton — which has no Windows wheels, so on
    # Windows it additionally needs the community `triton-windows` package. Costs a ~30s compile
    # warmup per process, which is heavy for short runs. Fails loudly with guidance if the
    # toolchain can't support it.
    amp: bool = False
    compile: bool = False
    # `cuda_graph`: capture the whole training step (zero-grad + forward + loss + backward +
    # optimizer step, + EMA update if enabled) into a CUDA graph and replay it every batch,
    # eliminating per-kernel-launch dispatch overhead. CUDA-only (silently inert on CPU).
    # Supported on the standard train path AND on the DS family (`train_deep_supervision` /
    # `train_act`), where the captured unit is one PASS/SEGMENT — those routines run a fixed
    # number of identically-shaped passes per batch, differing only in the carried (z, a) state,
    # so two graphs (fresh-init + carried-state) on one shared pool cover them. The trajectory
    # curriculum routines and `train_stable` still raise: per-batch depth resampling and the
    # `torch.func.jvp` probe have no fixed capturable shape.
    # Composes with `amp` (see `amp_static_loss_scale`).
    #
    # Why this can be a much bigger lever than `amp`/`compile` on SMALL configs: profiling a
    # tiny model (M26 ETTh1 forecasting: batch=128, n_cells=7) showed only ~4.5ms of a 21ms
    # eager step was actual GPU compute — the rest was ~590 kernel-launch dispatches/step. CUDA
    # graphs eliminate nearly all of that, measured ~4.9-5.3x end-to-end there (vs `compile`
    # default mode's ~2x) — on a workload this small, replaying the whole step as a single
    # pre-recorded launch beats kernel fusion. On a large-GEMM config (e.g. the M23 sudoku sweep)
    # the earlier speed-knob search found CUDA graphs a 1.00x no-op (§11.3), because the GPU was
    # already compute-bound there — this knob's payoff is shape-dependent, so measure per config.
    #
    # GPU-VERIFIED DANGER ZONE (results/log/m35.md): on a WIDE (hidden_dim ~1000+), HIGH-
    # in_features arm (e.g. a TRMMixer sized like electricity's h1110 widest arm, in_features in
    # the tens of thousands) DS-family capture (two graphs: fresh + carry) measured 2.76x SLOWER
    # than eager (7.58s/pass -> 20.93s/pass) and used MORE peak memory (11.37GB -> 11.99GB) on an
    # 8GB card, mirroring #18's standard-path finding, only worse. Eager ALONE already exceeded
    # the card's memory at this shape, so the fix there is amp + microbatch_size, never
    # cuda_graph, regardless of DS-family or standard path.
    #
    # Correctness caveats, both handled internally, not the caller's problem:
    #  - the last batch of an epoch is DROPPED if it's smaller than the graph's captured batch
    #    size (a CUDA graph needs a fixed shape) — the standard `drop_last` tradeoff, applied only
    #    under this flag.
    #  - a few warmup steps run for real (on a side stream, required before capture) and are then
    #    UNDONE (parameters/optimizer state/EMA shadow restored to their pre-warmup values) so
    #    training starts from the true initial weights, not from post-warmup ones.
    # Like `amp`/`compile`, NOT bit-identical when on (warmup can pick different cuDNN/cuBLAS
    # algorithms than a pure eager run) — opt-in, and an experiment must set it uniformly across
    # every arm.
    cuda_graph: bool = False
    # `cuda_graph_tail`: capture a SECOND graph for the ragged final batch instead of dropping it,
    # so graphed runs cover exactly the rows an eager run does. The two graphs share one memory
    # pool, so this does not double the graph memory footprint that §11.2 #18 found harmful at
    # high channel counts. Off by default because dropping the tail is what every committed
    # `cuda_graph` result did — turning this on changes which rows are trained on, so it is a
    # numerical change, not a pure speed knob. Ignored unless `cuda_graph` is on.
    cuda_graph_tail: bool = False
    # `amp_static_loss_scale`: the fixed fp16 loss scale used when `amp` and `cuda_graph` are BOTH
    # on. GradScaler's dynamic scale needs a host-side inf/nan check every step, which a replayed
    # graph cannot perform; a fixed scale (multiply the loss, divide the grads, both inside the
    # captured region) is the standard graph-compatible AMP recipe. Overflow cannot be caught
    # per-step, so it surfaces as a non-finite EPOCH loss and raises with instructions to lower
    # this value. Unused when the two flags are not combined (plain `amp` keeps dynamic scaling).
    # Default 8192 (2**13), not GradScaler's usual 65536 init: on real GPU testing, 65536
    # overflowed fp16 gradients on the deep-supervision multi-pass captured path (n_sup=3,
    # carry=True) at epoch 1, while the single-pass standard path was fine at that value —
    # 65536..32768 is right at that path's overflow edge (binary-searched: 32768 trains cleanly,
    # 65536 overflows), so 8192 keeps an 8x margin below the observed failure point rather than
    # trusting a value that was never actually run on a GPU.
    amp_static_loss_scale: float = 8192.0
    # `speed`: per-arm automatic selection among the knob combinations above. "manual" (default)
    # honours the explicit amp/cuda_graph flags exactly as before. "auto" times a handful of real
    # training steps per arm, on a THROWAWAY copy of the model and optimizer, for each candidate
    # mode this config admits, then trains with the fastest — operationalizing the repo's own
    # repeatedly-learned lesson that every one of these knobs flips sign by config shape (§11.2
    # #16/#18: CUDA graphs are ~5x on small-cell shapes and ~3x SLOWER at high channel counts;
    # amp is 1.18x on sudoku and 0.74x on ETTh1). The chosen mode is recorded per arm in the run
    # record, so a result always says how it was produced. CUDA-only; on CPU "auto" resolves to
    # eager. NOTE: because different arms may land on different modes, "auto" deliberately breaks
    # the "set knobs uniformly across arms" contract — the modes are numerically equivalent up to
    # fp16 rounding, but a Δ measured under mixed modes carries a per-arm precision difference.
    # Use it for exploration/timing; pin the winning mode manually for a result you will publish.
    # `speed_probe_steps` sets how many timed steps each candidate gets (after warmup).
    #
    # ★ GPU-VERIFIED FINDING (results/log/m35.md, CLAUDE.md §11.2 #21): "auto" is NOT currently
    # recommended for a real run. Two concrete problems, not hypothetical: (1) the autotuner only
    # probes through the standard `train()` path, so it silently gives NO autotuning to DS-family
    # arms (`n_sup>1` / `use_act` / curriculum / `train_stable`) — falls back to the configured
    # flags (default: plain eager) — which is exactly backwards, since DS-family arms are where
    # `cuda_graph` gives its biggest measured win (2.25x on `m18e_compute_matched`). (2) at real
    # config scale (`electricity`, M=321) the probe's own cost dwarfed what the chosen modes
    # saved: total wall clock was ~5x SLOWER than just pinning `amp: true` manually, even though
    # every autotuned choice was individually sensible. Prefer manually setting `amp`/`cuda_graph`
    # per §11.2 #16/#18/#21's measured dispatch rules (small launch-bound cells → `cuda_graph`;
    # high channel count → `amp`+`microbatch_size`, no graph) over reaching for `auto`.
    speed: str = "manual"
    speed_probe_steps: int = 8

    @field_validator("speed")
    @classmethod
    def _check_speed(cls, v: str) -> str:
        if v not in ("manual", "auto"):
            raise ValueError(f"train.speed must be 'manual' or 'auto', got {v!r}")
        return v


class SweepConfig(BaseModel):
    """Sweep one task parameter over a list of values, in a single run.

    Produces the k-vs-accuracy curve (with variance bands) the M0 DoD asks for,
    from one config (CLAUDE.md §11).
    """

    param: str  # key in task.params, e.g. "k"
    values: list


class GridConfig(BaseModel):
    """Sweep several task parameters over a full Cartesian product, in a single run.

    Where `sweep` varies one axis to draw a curve, `grid` varies *several* axes to
    *replicate* a finding across configs — e.g. rerun the whole arm factorial across
    CA `rule` × `w` to check the M2 cross-task robustness isn't a one-config fluke
    (CLAUDE.md §11 / M2-confirm). Each grid cell still runs every arm at its own
    config; cells are independent configs, not an ablation, so §5.6 ("one knob per
    ablation") still holds *within* each cell.
    """

    params: dict[str, list]

    def points(self) -> list[dict]:
        """List of task-param override dicts, one per Cartesian-product cell."""
        keys = list(self.params.keys())
        return [
            dict(zip(keys, combo)) for combo in itertools.product(*(self.params[k] for k in keys))
        ]


class ExtrapolationConfig(BaseModel):
    """Depth-extrapolation sweep over task CA steps (T) and test unroll steps (R)."""

    T_values: list[int]
    R_values: list[int]


class CurriculumConfig(BaseModel):
    """Train across a range of CA depths instead of a single fixed T (M3b).

    Each batch samples a depth ``T ~ Uniform{T_min..T_max}``; the model is unrolled to T and
    supervised against the trajectory up to s_T. Seeing the step operator applied at varying
    depths is what should let a step-aligned loop learn a *transferable* operator (the M1
    extrapolation null + M3a optimization wall are the two stacked levers this targets). The
    trajectory dataset is generated once at length ``T_max``; per-batch depths slice into it.
    """

    param: str = "T"  # the task depth parameter the curriculum sweeps
    T_min: int = 1
    T_max: int = 8


class DiagnosticsConfig(BaseModel):
    """Opt-in latent / weight introspection pass (M21). OFF by default = bit-identical.

    When ``enabled``, the runner runs ``eval.introspection.run_introspection`` on each trained
    arm over the first test batch and writes a side-car ``*_diagnostics.csv``. It is a *measurement-
    only* layer: it reads the trained model with forward / autograd passes (the M7
    ``init_state``/``return_state`` API + forward hooks), never perturbing training or any committed
    metric (CLAUDE.md §5/§8). Diagnostics are per-arm descriptors; the deliverable reports them
    across all arms and both anchor regimes (the contrast is the finding, never a lone loop number).
    """

    enabled: bool = False
    # Over-unroll horizon = factor × the arm's trained n_steps (M1/M8 decay is read off the tail).
    overunroll_factor: int = 4
    # Random z0 inits for the path-independence / asymptotic-alignment probe (Anil 2022).
    n_random_inits: int = 5
    # Power-iteration steps for the Jacobian spectral-radius / operator-norm estimate.
    power_iter_steps: int = 20
    # Number of examples the (per-example) Jacobian spectrum is averaged over.
    jac_n_examples: int = 8


class ExperimentConfig(BaseModel):
    task: TaskConfig
    arms: list[ModelConfig]
    train: TrainConfig
    sweep: Optional[SweepConfig] = None
    grid: Optional[GridConfig] = None
    extrapolation: Optional[ExtrapolationConfig] = None
    curriculum: Optional[CurriculumConfig] = None
    diagnostics: Optional[DiagnosticsConfig] = None
    # Pairs of arm labels to diff: [[recurrent, control], ...]. If omitted, every
    # non-last arm is diffed against the last arm (assumed to be the control).
    deltas: Optional[list[list[str]]] = None
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    results_dir: str = "results"

    # Number of worker processes for the per-axis-point seed loop. Seeds are embarrassingly
    # parallel — each is a pure function of its seed and self-reseeds (CLAUDE.md §5.3) — so
    # running them across processes is **bit-identical** to serial, just faster on multi-core
    # CPUs (the only place parallelism helps here, since the tiny per-run work is pinned to
    # one torch thread). Default 1 = serial (unchanged behaviour); raise it (e.g. to the core
    # count) on a ≥5-seed run for a near-linear speedup. Each worker is pinned to
    # `train.num_threads` so workers × threads never oversubscribe.
    parallel_workers: int = 1

    # --- M3a: depth-at-fixed-budget sweep (CLAUDE.md §11 / LOG.md) -------------------
    # When set (e.g. "T"), every recurrent/untied arm's unroll depth is set to the swept
    # task value `task_params[param]` instead of its static `n_steps`. This is what makes
    # "match the loop's n_steps to the task's T" a config knob: as we sweep T, the loop
    # unrolls T steps and the untied stack grows to T blocks, all from one config. Without
    # it, depth would be pinned at the per-arm n_steps and the depth sweep would be a no-op.
    couple_n_steps_to_param: Optional[str] = None
    # Budget-parity audit (the M3a confound guard). `budget_reference` is the arm label
    # whose param count defines the fixed budget (the loop). Every other arm except those
    # in `budget_ceiling` must land within `budget_tol` of it, *per cell*; the runner logs
    # realized counts and flags any breach. `budget_ceiling` lists deliberately
    # non-param-matched arms (e.g. the ~n_steps× `untied_stack`) exempt from the check.
    budget_reference: Optional[str] = None
    budget_ceiling: list[str] = Field(default_factory=list)
    budget_tol: float = 0.02

    @model_validator(mode="after")
    def _check_axes(self) -> "ExperimentConfig":
        # `sweep` (1-D curve) and `grid` (N-D replication) are mutually exclusive: both
        # drive the same outer point-loop, so allowing both would silently ignore one.
        if self.sweep is not None and self.grid is not None:
            raise ValueError("Set at most one of `sweep` or `grid`, not both.")
        # The extrapolation harness keeps a single result set keyed by (T, R); pairing
        # it with a multi-cell grid would overwrite all but the last cell. The grid
        # replicates the *at-training-config* Δ; depth-extrapolation is a separate run.
        if self.grid is not None and self.extrapolation is not None:
            raise ValueError("`grid` and `extrapolation` cannot be combined in one run.")
        return self

    def axis_points(self) -> list[tuple[str, dict]]:
        """(label, task-param-overrides) for each point on the sweep/grid axis.

        One entry with no overrides when neither `sweep` nor `grid` is set.
        """
        if self.grid is not None:
            return [(", ".join(f"{k}={v}" for k, v in ov.items()), ov) for ov in self.grid.points()]
        if self.sweep is not None:
            return [(f"{self.sweep.param}={v}", {self.sweep.param: v}) for v in self.sweep.values]
        return [("single", {})]

    def resolved_deltas(self) -> list[list[str]]:
        if self.deltas is not None:
            return self.deltas
        labels = [a.resolved_label() for a in self.arms]
        if len(labels) < 2:
            return []
        control = labels[-1]
        return [[lbl, control] for lbl in labels[:-1]]
