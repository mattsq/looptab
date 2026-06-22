"""Integration test: train TRM and FFMatched on Task 0 for a few epochs."""

import pytest
import torch
from torch.utils.data import DataLoader

from looptab.data.dataset import TrajectoryDataset, make_loaders, make_trajectory_dataset
from looptab.data.generators import make_linear
from looptab.eval.metrics import accuracy, delta_report
from looptab.models.controls import FFMatched
from looptab.models.trm import TRM
from looptab.train.loop import train, train_curriculum, train_progressive


def _small_loader():
    X, y = make_linear(n=200, d=10, task_seed=0, sample_seed=1)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=64)


def test_train_trm_runs():
    m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    loader = _small_loader()
    losses = train(m, loader, epochs=5, lr=1e-3, device="cpu")
    assert len(losses) == 5
    assert all(isinstance(x, float) for x in losses)


def test_train_ff_runs():
    m = FFMatched(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    loader = _small_loader()
    losses = train(m, loader, epochs=5, lr=1e-3, device="cpu")
    assert len(losses) == 5


def test_accuracy_above_chance():
    """After 30 epochs on linear (easy), both models should beat 55% accuracy."""
    loader = _small_loader()
    for cls in [TRM, FFMatched]:
        m = cls(in_features=10, num_classes=2, hidden_dim=32, latent_dim=32, n_steps=4)
        train(m, loader, epochs=30, lr=1e-3, device="cpu")
        acc = accuracy(m, loader)
        assert acc > 0.55, f"{cls.__name__} acc={acc:.3f} not above chance"


def test_delta_report():
    rec = [0.8, 0.82, 0.79, 0.81, 0.83]
    ctl = [0.75, 0.76, 0.74, 0.77, 0.75]
    r = delta_report(rec, ctl)
    assert r["delta_mean"] > 0
    assert r["n_seeds"] == 5
    assert "delta_std" in r
    assert r["sign_test"]["n_pos"] == 5  # all five seeds favour recurrent


# --- M3b: curriculum + step-aligned deep supervision ---------------------------------------


def _traj_loader(T_max=5, w=8, n=128):
    ds = make_trajectory_dataset(
        task_cfg={"w": w, "rule": 30, "distractors": 2},
        task_seed=0,
        sample_seed=1,
        n=n,
        T_max=T_max,
    )
    assert isinstance(ds, TrajectoryDataset)
    loader, _ = make_loaders(ds, ds, batch_size=64)
    return loader, w


def test_train_curriculum_step_aligned_runs_and_learns():
    """Step-aligned DS over a depth curriculum runs and reduces the training loss."""
    loader, w = _traj_loader(T_max=5, w=8)
    m = TRM(
        in_features=10,  # w(8) + distractors(2)
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=5,
        deep_supervision=True,
        out_features=w,
    )
    losses = train_curriculum(
        m, loader, T_min=1, T_max=5, ds_mode="step_aligned", epochs=20, seed=0
    )
    assert len(losses) == 20
    assert losses[-1] < losses[0]  # the operator-supervised loss goes down


def test_train_curriculum_final_mode_runs():
    loader, w = _traj_loader(T_max=4, w=8)
    m = TRM(
        in_features=10,
        num_classes=2,
        hidden_dim=24,
        latent_dim=24,
        n_steps=4,
        deep_supervision=False,
        out_features=w,
    )
    losses = train_curriculum(m, loader, T_min=1, T_max=4, ds_mode="final", epochs=5, seed=0)
    assert len(losses) == 5


def test_step_aligned_requires_per_step_readouts():
    """ds_mode='step_aligned' on a model without per-step readouts is undefined => raises."""
    loader, w = _traj_loader(T_max=3, w=8)
    m = TRM(
        in_features=10,
        num_classes=2,
        hidden_dim=16,
        latent_dim=16,
        n_steps=3,
        deep_supervision=False,  # no per-step logits emitted
        out_features=w,
    )
    with pytest.raises(ValueError):
        train_curriculum(m, loader, T_min=3, T_max=3, ds_mode="step_aligned", epochs=1, seed=0)


def test_train_progressive_final_runs_and_learns():
    """M7: progressive loss (final target) runs and reduces the training loss."""
    loader, w = _traj_loader(T_max=5, w=8)
    m = TRM(
        in_features=10,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=5,
        deep_supervision=False,
        out_features=w,
    )
    losses = train_progressive(
        m, loader, T_min=1, T_max=5, ds_mode="progressive_final", alpha=0.5, epochs=20, seed=0
    )
    assert len(losses) == 20
    assert losses[-1] < losses[0]


def test_train_progressive_step_runs_and_learns():
    """M7: step-aligned progressive loss runs and reduces the training loss."""
    loader, w = _traj_loader(T_max=5, w=8)
    m = TRM(
        in_features=10,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=5,
        deep_supervision=True,  # progressive_step needs per-step readouts
        out_features=w,
    )
    losses = train_progressive(
        m, loader, T_min=1, T_max=5, ds_mode="progressive_step", alpha=0.5, epochs=20, seed=0
    )
    assert len(losses) == 20
    assert losses[-1] < losses[0]


def test_train_progressive_step_requires_per_step_readouts():
    loader, w = _traj_loader(T_max=3, w=8)
    m = TRM(
        in_features=10,
        num_classes=2,
        hidden_dim=16,
        latent_dim=16,
        n_steps=3,
        deep_supervision=False,
        out_features=w,
    )
    with pytest.raises(ValueError):
        train_progressive(
            m, loader, T_min=3, T_max=3, ds_mode="progressive_step", epochs=1, seed=0
        )


def test_train_progressive_rejects_non_progressive_mode():
    loader, w = _traj_loader(T_max=3, w=8)
    m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=3, out_features=w)
    with pytest.raises(ValueError):
        train_progressive(m, loader, T_min=1, T_max=3, ds_mode="final", epochs=1, seed=0)


def test_train_progressive_is_deterministic():
    """Same seed => identical training (the per-batch T,k schedule is reproducible)."""
    loader, w = _traj_loader(T_max=5, w=8)

    def run():
        torch.manual_seed(0)
        m = TRM(
            in_features=10,
            num_classes=2,
            hidden_dim=16,
            latent_dim=16,
            n_steps=5,
            deep_supervision=True,
            out_features=w,
        )
        return train_progressive(
            m, loader, T_min=1, T_max=5, ds_mode="progressive_step", epochs=8, seed=3
        )

    assert run() == run()


def test_curriculum_depth_schedule_is_deterministic():
    """Same seed => identical training (the per-batch T schedule is reproducible)."""
    loader, w = _traj_loader(T_max=5, w=8)

    def run():
        torch.manual_seed(0)
        m = TRM(
            in_features=10,
            num_classes=2,
            hidden_dim=16,
            latent_dim=16,
            n_steps=5,
            deep_supervision=True,
            out_features=w,
        )
        return train_curriculum(
            m, loader, T_min=1, T_max=5, ds_mode="step_aligned", epochs=8, seed=3
        )

    assert run() == run()
