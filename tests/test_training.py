"""Integration test: train TRM and FFMatched on Task 0 for a few epochs."""

import torch
from torch.utils.data import DataLoader

from looptab.data.generators import make_linear
from looptab.eval.metrics import accuracy, delta_report
from looptab.models.controls import FFMatched
from looptab.models.trm import TRM
from looptab.train.loop import train


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
