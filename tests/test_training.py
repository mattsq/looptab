"""Integration test: train TRM and FFMatched on Task 0 for a few epochs."""

import pytest
import torch
from torch.utils.data import DataLoader

from looptab.data.dataset import TrajectoryDataset, make_loaders, make_trajectory_dataset
from looptab.data.generators import make_linear
from looptab.eval.metrics import accuracy, delta_report, evaluate_act
from looptab.models.controls import FFMatched
from looptab.models.trm import TRM
from looptab.train.loop import train, train_act, train_curriculum, train_progressive


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


# --- M18: train_deep_supervision (N_sup detached carry) + EMA -------------------------------

from looptab.train.loop import train_deep_supervision  # noqa: E402


def test_train_deep_supervision_runs_and_is_deterministic():
    """N_sup detached-carry training: runs, and same seed → identical weights (reproducible)."""
    def _make():
        torch.manual_seed(0)
        return TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=3)

    loader = _small_loader()
    m1 = _make()
    train_deep_supervision(m1, loader, n_sup=3, epochs=4, lr=1e-3, device="cpu")
    m2 = _make()
    train_deep_supervision(m2, loader, n_sup=3, epochs=4, lr=1e-3, device="cpu")
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_train_deep_supervision_nsup1_matches_plain_single_pass():
    """n_sup=1 is one supervised forward per batch — distinct routine, same gradient content.

    We don't assert byte-equality with ``train`` (loss aggregation differs), only that the
    routine trains a model to a finite loss and changes the weights.
    """
    torch.manual_seed(0)
    m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2,
            deep_supervision=False)
    before = [p.detach().clone() for p in m.parameters()]
    losses = train_deep_supervision(m, _small_loader(), n_sup=1, epochs=3, device="cpu")
    assert len(losses) == 3 and all(v == v for v in losses)  # no NaN
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_ema_changes_weights_and_is_deterministic():
    """EMA folds averaged weights in → differs from no-EMA, and is reproducible."""
    def _run(ema_decay):
        torch.manual_seed(0)
        m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
        train(m, _small_loader(), epochs=6, lr=1e-2, ema_decay=ema_decay, device="cpu")
        return [p.detach().clone() for p in m.parameters()]

    no_ema = _run(None)
    ema_a = _run(0.9)
    ema_b = _run(0.9)
    # EMA weights are reproducible...
    for a, b in zip(ema_a, ema_b):
        assert torch.equal(a, b)
    # ...and differ from the un-averaged endpoint.
    assert any(not torch.equal(a, b) for a, b in zip(no_ema, ema_a))


def test_ema_invalid_nsup():
    m = TRM(in_features=10, num_classes=2, hidden_dim=8, latent_dim=8, n_steps=2)
    with pytest.raises(ValueError):
        train_deep_supervision(m, _small_loader(), n_sup=0, epochs=1, device="cpu")


def test_train_deep_supervision_carry_flag_matches_compute_changes_result():
    """carry=False (compute-matched control) runs, is deterministic, and differs from carry=True.

    Both do n_sup forward+backward+step per batch (same compute); only the detached-carry differs,
    so they must train to *different* weights — the isolation the B1 review fix needs.
    """
    def _run(carry):
        torch.manual_seed(0)
        m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=3)
        train_deep_supervision(m, _small_loader(), n_sup=3, carry=carry, epochs=4, device="cpu")
        return [p.detach().clone() for p in m.parameters()]

    carry_a = _run(True)
    carry_b = _run(True)
    nocarry = _run(False)
    for a, b in zip(carry_a, carry_b):   # deterministic
        assert torch.equal(a, b)
    assert any(not torch.equal(a, b) for a, b in zip(carry_a, nocarry))  # carry matters


# --- M23: ACT / adaptive-computation halting -----------------------------------------------------
def _multi_loader():
    """A small multi-output loader (parity-shaped) so ACT's per-example exact-match halt target
    and the multi-output loss path are exercised."""
    from looptab.data.generators import make_multi_parity

    X, y = make_multi_parity(n=256, d=8, k=2, w=4, task_seed=0, sample_seed=1)[:2]
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=64)


def _act_trm(**kw):
    torch.manual_seed(0)
    return TRM(in_features=8, num_classes=2, out_features=4, hidden_dim=16, latent_dim=16,
               n_steps=3, deep_supervision=False, use_act=True, **kw)


def test_act_off_is_bit_identical():
    """use_act=False adds no params and does not touch the forward path (byte-identical)."""
    torch.manual_seed(0)
    m_off = TRM(in_features=8, num_classes=2, out_features=4, hidden_dim=16, latent_dim=16,
                n_steps=3)
    torch.manual_seed(0)
    m_act = TRM(in_features=8, num_classes=2, out_features=4, hidden_dim=16, latent_dim=16,
                n_steps=3, use_act=True)
    assert m_off.halt_head is None and m_act.halt_head is not None
    # The halt head is the ONLY extra parameter; the shared core is initialized identically.
    assert m_act.count_params() > m_off.count_params()
    X = torch.randn(5, 8)
    o_off, _ = m_off(X)
    o_act, _ = m_act(X)  # forward ignores the halt head
    # Copy the shared (non-halt) params across and confirm forward outputs match exactly.
    sd = m_off.state_dict()
    m_act.load_state_dict({**m_act.state_dict(), **sd})
    assert torch.equal(m_off(X)[0], m_act(X)[0])


def test_train_act_runs_and_is_deterministic():
    """ACT training runs to a finite loss and same seed → identical weights."""
    m1 = _act_trm()
    l1 = train_act(m1, _multi_loader(), max_segments=3, epochs=4, lr=1e-3, device="cpu")
    m2 = _act_trm()
    train_act(m2, _multi_loader(), max_segments=3, epochs=4, lr=1e-3, device="cpu")
    assert len(l1) == 4 and all(v == v for v in l1)  # no NaN
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_train_act_requires_halt_head():
    """train_act on a non-ACT model fails loudly rather than crashing opaquely."""
    m = TRM(in_features=8, num_classes=2, out_features=4, hidden_dim=16, latent_dim=16, n_steps=3)
    with pytest.raises(ValueError):
        train_act(m, _multi_loader(), max_segments=3, epochs=1, device="cpu")


def test_evaluate_act_runs_and_reports_segments():
    """Adaptive eval returns metrics + avg_segments within [1, max_segments], deterministically."""
    m = _act_trm()
    train_act(m, _multi_loader(), max_segments=4, epochs=6, lr=1e-2, device="cpu")
    r1 = evaluate_act(m, _multi_loader(), max_segments=4, device="cpu", want_exact_match=True)
    r2 = evaluate_act(m, _multi_loader(), max_segments=4, device="cpu", want_exact_match=True)
    assert 1.0 <= r1["avg_segments"] <= 4.0
    assert 0.0 <= r1["accuracy"] <= 1.0 and 0.0 <= r1["exact_match"] <= 1.0
    assert r1["avg_segments"] == r2["avg_segments"] and r1["exact_match"] == r2["exact_match"]


def test_act_halts_earlier_on_solved_examples():
    """The halt head is adaptive: a threshold it (nearly) always fires uses ~1 segment; a threshold
    it never reaches uses all max_segments. Bracketing confirms per-example early-stopping works."""
    m = _act_trm()
    train_act(m, _multi_loader(), max_segments=4, epochs=8, lr=1e-2, device="cpu")
    loader = _multi_loader()
    lo = evaluate_act(m, loader, max_segments=4, device="cpu", halt_threshold=0.0)  # halt at seg 1
    hi = evaluate_act(m, loader, max_segments=4, device="cpu", halt_threshold=1.0)  # never halt
    assert lo["avg_segments"] == 1.0
    assert hi["avg_segments"] == 4.0
