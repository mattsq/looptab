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

from pydantic import BaseModel, Field, model_validator


class TaskConfig(BaseModel):
    name: Literal["linear", "parity", "iterated"]
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
    ds_mode: Literal["final", "step_aligned"] = "final"

    def resolved_label(self) -> str:
        return self.label or self.name


class TrainConfig(BaseModel):
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    device: str = "cpu"


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


class ExperimentConfig(BaseModel):
    task: TaskConfig
    arms: list[ModelConfig]
    train: TrainConfig
    sweep: Optional[SweepConfig] = None
    grid: Optional[GridConfig] = None
    extrapolation: Optional[ExtrapolationConfig] = None
    curriculum: Optional[CurriculumConfig] = None
    # Pairs of arm labels to diff: [[recurrent, control], ...]. If omitted, every
    # non-last arm is diffed against the last arm (assumed to be the control).
    deltas: Optional[list[list[str]]] = None
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    results_dir: str = "results"

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
