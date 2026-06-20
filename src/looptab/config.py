"""Pydantic config models. A single config + seed fully determines a run.

An experiment is a list of `arms` (each an independent model spec with a label)
trained on a task, optionally swept over one task parameter (e.g. parity `k`).
Reporting `deltas` between named arms is what satisfies the prime directive: every
result is a Δ between a recurrent arm and a matched control — and, critically, the
deep-supervision ablation is *its own arm* so the loop and deep supervision are never
confounded (CLAUDE.md §4/§8).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


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


class ExperimentConfig(BaseModel):
    task: TaskConfig
    arms: list[ModelConfig]
    train: TrainConfig
    sweep: Optional[SweepConfig] = None
    # Pairs of arm labels to diff: [[recurrent, control], ...]. If omitted, every
    # non-last arm is diffed against the last arm (assumed to be the control).
    deltas: Optional[list[list[str]]] = None
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    results_dir: str = "results"

    def resolved_deltas(self) -> list[list[str]]:
        if self.deltas is not None:
            return self.deltas
        labels = [a.resolved_label() for a in self.arms]
        if len(labels) < 2:
            return []
        control = labels[-1]
        return [[lbl, control] for lbl in labels[:-1]]
