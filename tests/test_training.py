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


def test_microbatch_preserves_effective_batch_update():
    """Gradient accumulation keeps one optimizer update per loader batch and weights ragged
    microbatches by their sample share, matching the full-batch mean up to matmul reduction
    order."""
    X, y = make_linear(n=128, d=10, task_seed=0, sample_seed=1)
    loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=64,
        shuffle=False,
    )

    torch.manual_seed(7)
    full = FFMatched(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    torch.manual_seed(7)
    micro = FFMatched(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    train(full, loader, epochs=1, lr=1e-3, weight_decay=0.0, device="cpu")
    train(
        micro,
        loader,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        device="cpu",
        microbatch_size=24,
    )
    for a, b in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_microbatch_rejects_invalid_size_and_cuda_graph_combination():
    m = FFMatched(in_features=10, num_classes=2, hidden_dim=8, latent_dim=8, n_steps=2)
    with pytest.raises(ValueError, match="microbatch_size must be"):
        train(m, _small_loader(), epochs=1, microbatch_size=0)
    with pytest.raises(ValueError, match="microbatch_size and cuda_graph"):
        train(m, _small_loader(), epochs=1, microbatch_size=8, cuda_graph=True)


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


def test_sign_test_eps_reclassifies_near_ties():
    """M29c guard: |Δ|<=eps is a practical tie. eps=0 (default) is the classic test (bit-identical);
    a small eps drops one-cell ceiling differences so they stop counting as votes."""
    from looptab.eval.metrics import sign_test

    deltas = [0.2, 0.2, 4e-5, 4e-5, 4e-5, 4e-5, 0.009, 0.0007]  # the m29c carry−nocarry shape
    raw = sign_test(deltas)  # eps=0.0
    assert raw["n_pos"] == 8 and raw["n_zero"] == 0  # all 8 count → an impressive-looking 8/0
    assert raw["p_value"] < 0.05
    robust = sign_test(deltas, eps=1e-3)  # near-ties (the 4e-5s and 0.0007) dropped
    assert robust["n_zero"] == 5 and robust["n_pos"] == 3  # only {0.2,0.2,0.009} remain
    assert robust["p_value"] >= 0.05  # 3/3 → p=0.25, no longer significant


def test_delta_report_reports_ceiling_tie_robustness():
    """delta_report carries the ceiling-tie-robust companion so a reader can catch a raw sign count
    that is riding near-saturation ties (the M29c pathology) without recomputing."""
    rec = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99971, 1.0]
    ctl = [0.79367, 0.78633, 0.99996, 0.99996, 0.99992, 0.99996, 0.99058, 0.99933]  # m29c nocarry
    r = delta_report(rec, ctl)
    assert r["sign_test"]["n_pos"] == 8 and r["sign_test"]["p_value"] < 0.05  # raw: 8/0
    assert r["n_near_tie"] >= 4  # ~4 seeds are within one test-cell of a tie at ceiling
    assert r["sign_test_robust"]["p_value"] >= 0.05  # collapses once ceiling-ties are dropped
    # eps=0 must reproduce the classic sign test exactly (bit-identical default behaviour).
    from looptab.eval.metrics import sign_test

    assert delta_report(rec, ctl, near_tie_eps=0.0)["sign_test_robust"] == sign_test(
        [a - b for a, b in zip(rec, ctl)]
    )


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


def test_cuda_graph_amp_combination_is_accepted_and_inert_on_cpu():
    """amp+cuda_graph used to raise; the static-loss-scale capture path (see _GraphedStepBase)
    now supports the combination. On CPU both flags stay inert, so the run must be bit-identical
    to the plain eager one rather than raising."""
    torch.manual_seed(0)
    m_ref = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    ref = train(m_ref, _small_loader(), epochs=3, lr=1e-2, device="cpu")
    torch.manual_seed(0)
    m_both = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    got = train(m_both, _small_loader(), epochs=3, lr=1e-2, device="cpu", amp=True,
                cuda_graph=True)
    assert ref == got
    for a, b in zip(m_ref.parameters(), m_both.parameters()):
        assert torch.equal(a, b)


def test_microbatch_still_rejects_cuda_graph():
    """Lifting the amp+cuda_graph exclusion must NOT loosen the microbatch one: gradient
    accumulation spans several forward/backward passes per optimizer step, which a single
    fixed-shape captured step cannot represent."""
    m = TRM(in_features=10, num_classes=2, hidden_dim=8, latent_dim=8, n_steps=2)
    with pytest.raises(ValueError, match="microbatch_size and cuda_graph"):
        train(m, _small_loader(), epochs=1, device="cpu", microbatch_size=8, cuda_graph=True)


def test_ds_family_speed_flags_are_inert_on_cpu():
    """amp/cuda_graph on the deep-supervision family must be inert on CPU, i.e. bit-identical to
    the pre-flag routine — the same portability contract `train` has, so a cuda config can run
    unchanged on a CPU box."""
    def _run(**kw):
        torch.manual_seed(0)
        m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=3)
        losses = train_deep_supervision(m, _small_loader(), n_sup=3, epochs=3, lr=1e-2,
                                        device="cpu", **kw)
        return losses, [p.detach().clone() for p in m.parameters()]

    ref_losses, ref_params = _run()
    got_losses, got_params = _run(amp=True, cuda_graph=True)
    assert ref_losses == got_losses
    for a, b in zip(ref_params, got_params):
        assert torch.equal(a, b)


def test_carried_state_cast_is_a_semantic_noop():
    """Pins down what the DS routines' `.to(X.dtype)` on the carried (z, a) actually does.

    Under AMP the loop returns fp16 state while X stays fp32. It would be easy to assume that
    breaks the models' `torch.cat([X, z, a])` — it does NOT: `cat` type-promotes, so the mixed
    carry silently becomes fp32 anyway. The explicit cast therefore changes nothing numerically;
    it is there to make the dtype of the cross-pass state stated rather than incidental (and so
    the captured-graph path can hold fixed-dtype static buffers). This test exists so nobody
    "fixes" a crash that was never there, or deletes the cast believing it was load-bearing.
    """
    torch.manual_seed(0)
    m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    X = torch.randn(4, 10)
    _, _, (z, a) = m(X, n_steps=1, return_state=True)
    half_state = (z.detach().half(), a.detach().half())
    promoted, _, _ = m(X, n_steps=1, init_state=half_state, return_state=True)
    cast_state = (half_state[0].to(X.dtype), half_state[1].to(X.dtype))
    explicit, _, _ = m(X, n_steps=1, init_state=cast_state, return_state=True)
    assert torch.equal(promoted, explicit)


def test_async_loss_accumulation_matches_python_sum():
    """The per-epoch device-side float64 accumulation replaced a per-batch `.item()` Python sum.
    On CPU the two must agree bit-for-bit, so the reported loss curves in committed run records
    stay comparable across the change."""
    torch.manual_seed(0)
    m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    losses = train_deep_supervision(m, _small_loader(), n_sup=2, epochs=3, lr=1e-2, device="cpu")
    # Recompute the same quantity the routine reports, the old way, from an identical rerun.
    torch.manual_seed(0)
    m2 = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    manual = []
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-2, weight_decay=1e-4)
    from looptab.train.loop import _loss_fn
    for _ in range(3):
        m2.train()
        total, n = 0.0, 0
        for X, y in _small_loader():
            state = None
            for _ in range(2):
                opt.zero_grad()
                logits, all_logits, state = m2(X, n_steps=None, init_state=state,
                                               return_state=True)
                loss = _loss_fn(logits, y)
                if all_logits is not None:
                    loss = loss + sum(_loss_fn(sl, y) for sl in all_logits) / len(all_logits)
                loss.backward()
                opt.step()
                state = (state[0].detach(), state[1].detach())
                total += loss.item()
                n += 1
        manual.append(total / n)
    assert losses == manual


def test_cuda_graph_is_inert_on_cpu():
    """cuda_graph=True on a CPU device must silently take the eager path (no CUDAGraph calls),
    same inertness contract as amp."""
    torch.manual_seed(0)
    m_ref = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    losses_ref = train(m_ref, _small_loader(), epochs=3, lr=1e-2, device="cpu", cuda_graph=False)
    torch.manual_seed(0)
    m_graph = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    losses_graph = train(m_graph, _small_loader(), epochs=3, lr=1e-2, device="cpu", cuda_graph=True)
    assert losses_ref == losses_graph
    for a, b in zip(m_ref.parameters(), m_graph.parameters()):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestCudaGraphOnGPU:
    """`_small_loader()` (n=200, batch_size=64) has a ragged last batch of 8 — exactly the case
    cuda_graph must drop rather than crash on, since a CUDA graph needs a fixed replay shape."""

    def test_trains_without_crashing_and_learns(self):
        torch.manual_seed(0)
        m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
        before = [p.detach().clone() for p in m.parameters()]
        losses = train(m, _small_loader(), epochs=10, lr=1e-2, device="cuda", cuda_graph=True)
        assert len(losses) == 10
        assert all(v == v for v in losses)  # no NaN
        assert any(
            not torch.equal(a, b.cpu()) for a, b in zip(before, m.parameters())
        )  # weights actually moved

    def test_roughly_matches_eager_trajectory(self):
        """Not bit-identical (warmup can pick different algorithms), but should land in the same
        ballpark as eager training on the same data/init — a real correctness signal beyond
        'didn't crash', since a badly-broken capture (e.g. the stream bug found during the POC)
        produces a plausible-looking but non-decreasing or wildly different loss curve."""
        def _run(cuda_graph):
            torch.manual_seed(0)
            m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
            return train(m, _small_loader(), epochs=15, lr=1e-2, device="cuda",
                         cuda_graph=cuda_graph)

        eager_losses = _run(False)
        graph_losses = _run(True)
        assert graph_losses[-1] < graph_losses[0] * 0.6  # genuinely decreasing, not plateaued
        assert abs(graph_losses[-1] - eager_losses[-1]) < 0.2  # same ballpark as eager

    def test_works_with_ema(self):
        torch.manual_seed(0)
        m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
        losses = train(m, _small_loader(), epochs=8, lr=1e-2, device="cuda", ema_decay=0.9,
                        cuda_graph=True)
        assert len(losses) == 8
        assert all(v == v for v in losses)

    def test_tail_graph_covers_the_ragged_batch(self):
        """`cuda_graph_tail` captures a SECOND graph for the smaller last batch instead of
        dropping it. n=200/batch=64 gives batches of 64,64,64,8 — so the tail flag must raise the
        per-epoch batch count from 3 to 4 and still train (the mid-training capture path, which
        must restore real optimizer moments rather than zeroing them)."""
        def _run(tail):
            torch.manual_seed(0)
            m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
            losses = train(m, _small_loader(), epochs=12, lr=1e-2, device="cuda",
                           cuda_graph=True, cuda_graph_tail=tail)
            return losses, m

        dropped_losses, _ = _run(False)
        tail_losses, m_tail = _run(True)
        assert all(v == v for v in tail_losses)  # no NaN from the second capture
        assert tail_losses[-1] < tail_losses[0] * 0.6  # still genuinely learning
        # The two differ: the tail run trains on 8 extra rows per epoch.
        assert tail_losses != dropped_losses

    def test_amp_plus_cuda_graph_trains(self):
        """The static-loss-scale AMP capture must converge, not just avoid crashing — the failure
        mode #17 documents (a wrong-but-plausible gradient) looks exactly like 'ran fine' unless
        the loss is checked for genuine descent."""
        torch.manual_seed(0)
        m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
        losses = train(m, _small_loader(), epochs=15, lr=1e-2, device="cuda", amp=True,
                       cuda_graph=True)
        assert all(v == v for v in losses)
        assert losses[-1] < losses[0] * 0.6

    def test_amp_static_scale_overflow_raises_actionably(self):
        """An absurd static scale overflows fp16 gradients; because a replayed graph cannot do
        GradScaler's host-side inf check, the routine must catch it at the epoch boundary and say
        which knob to turn — not train silently on NaN weights."""
        torch.manual_seed(0)
        m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
        with pytest.raises(RuntimeError, match="amp_static_loss_scale"):
            train(m, _small_loader(), epochs=3, lr=1e-2, device="cuda", amp=True,
                  cuda_graph=True, amp_static_loss_scale=1e30)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestDeepSupervisionGraphsOnGPU:
    """The DS family runs `n_sup` identically-shaped passes per batch, so its captured unit is the
    PASS (fresh-init graph + carried-state graph). These assert real convergence, not merely that
    the capture ran — the #17 stream-ordering class of bug is invisible to a crash test."""

    def test_ds_graph_trains_and_matches_eager_ballpark(self):
        def _run(**kw):
            torch.manual_seed(0)
            m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=3)
            return train_deep_supervision(m, _small_loader(), n_sup=3, epochs=15, lr=1e-2,
                                          device="cuda", **kw)

        eager = _run()
        graphed = _run(cuda_graph=True)
        assert all(v == v for v in graphed)
        assert graphed[-1] < graphed[0] * 0.6
        assert abs(graphed[-1] - eager[-1]) < 0.2

    def test_ds_graph_respects_carry_false(self):
        """carry=False must replay the FRESH graph every pass (the compute-matched control), so it
        has to keep differing from carry=True under capture exactly as it does eagerly."""
        def _run(carry):
            torch.manual_seed(0)
            m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=3)
            train_deep_supervision(m, _small_loader(), n_sup=3, epochs=6, lr=1e-2,
                                   device="cuda", carry=carry, cuda_graph=True)
            return [p.detach().cpu().clone() for p in m.parameters()]

        assert any(not torch.equal(a, b) for a, b in zip(_run(True), _run(False)))

    def test_ds_graph_with_amp_and_ema(self):
        torch.manual_seed(0)
        m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=3)
        losses = train_deep_supervision(m, _small_loader(), n_sup=3, epochs=12, lr=1e-2,
                                        device="cuda", ema_decay=0.9, amp=True, cuda_graph=True)
        assert all(v == v for v in losses)
        assert losses[-1] < losses[0] * 0.6


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
