"""Pydantic config models. A single config + seed fully determines a run."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional


class TaskConfig(BaseModel):
    name: Literal["linear", "parity", "iterated"]
    params: dict = Field(default_factory=dict)
    n_train: int = 4000
    n_test: int = 1000
    task_seed: int = 0
    train_sample_seed: int = 1
    test_sample_seed: int = 2


class ModelConfig(BaseModel):
    name: str
    hidden_dim: int = 64
    latent_dim: int = 64
    n_steps: int = 4
    deep_supervision: bool = True


class TrainConfig(BaseModel):
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    deep_supervision_weight: float = 1.0
    device: str = "cpu"


class ExperimentConfig(BaseModel):
    task: TaskConfig
    model: ModelConfig
    control: ModelConfig
    train: TrainConfig
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    results_dir: str = "results"
