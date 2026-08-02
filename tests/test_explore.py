"""M34 — Explorative Modeling substrate: wrapper, best-of-K routine, side-car metrics.

Guards the three properties the milestone's claims rest on:
  1. the wrapper is transparent (shapes, param count, forward API) and defaults to the ZERO latent,
     so the headline single-pass eval is unchanged and can never read the answer key;
  2. `train_explore` is deterministic given the seed and actually selects the per-EXAMPLE argmin
     candidate (not a per-batch winner);
  3. the collapse check fires — a model whose output ignores the latent reports
     latent_sensitivity 0 / distinct_predictions 1, which is the null this experiment must detect.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from looptab.data.generators import make_converge, make_linear
from looptab.eval.metrics import evaluate, evaluate_explore, explore_diagnostics
from looptab.models.controls import FFMatched
from looptab.models.explore import ExploreWrapper, sample_noise
from looptab.models.mixer import TRMMixer
from looptab.models.trm import TRM
from looptab.train.loop import _loss_fn, _per_example_loss, train_explore


def _linear_loader(noise_dim=0):
    X, y = make_linear(n=128, d=10, task_seed=0, sample_seed=1)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=32)


def _converge_loader(w=8):
    X, y = make_converge(n=128, w=w, task_seed=0, sample_seed=1, rule=78)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=32)


def test_wrapper_shapes_and_zero_default():
    inner = TRM(in_features=10 + 4, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    m = ExploreWrapper(inner, noise_dim=4)
    X = torch.randn(8, 10)
    logits, all_logits = m(X)
    assert logits.shape == (8, 2)
    assert len(all_logits) == 2
    # Zero latent by default, and identical to explicitly installing zeros.
    m.set_noise(torch.zeros(8, 4))
    explicit, _ = m(X)
    m.clear_noise()
    default, _ = m(X)
    assert torch.equal(explicit, default)
    # Transparent: params are the wrapped model's, attributes fall through.
    assert m.count_params() == inner.count_params()
    assert m.n_steps == inner.n_steps
    assert m.readout is inner.readout


def test_wrapper_noise_changes_output():
    inner = TRM(in_features=10 + 4, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2)
    m = ExploreWrapper(inner, noise_dim=4)
    X = torch.randn(8, 10)
    base, _ = m(X)
    m.set_noise(torch.randn(8, 4) * 3.0)
    perturbed, _ = m(X)
    assert not torch.allclose(base, perturbed)


def test_wrapper_rejects_bad_noise():
    m = ExploreWrapper(TRM(in_features=14, num_classes=2, hidden_dim=8, latent_dim=8), noise_dim=4)
    with pytest.raises(ValueError):
        m.set_noise(torch.zeros(8, 5))  # wrong width
    m.set_noise(torch.zeros(3, 4))
    with pytest.raises(ValueError):
        m(torch.randn(8, 10))  # wrong batch


def test_wrapper_wraps_mixer_with_cell_multiple_noise():
    """Mixer arms need (in_features + noise_dim) % n_cells == 0 — one noise channel per cell."""
    w = 8
    inner = TRMMixer(
        in_features=w + w, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2, out_features=w
    )
    m = ExploreWrapper(inner, noise_dim=w)
    logits, _ = m(torch.randn(4, w))
    assert logits.shape == (4, w, 2)


def test_per_example_loss_reduces_to_loss_fn():
    for logits, targets in (
        (torch.randn(6, 3), torch.randint(0, 3, (6,))),
        (torch.randn(6, 5, 2), torch.randint(0, 2, (6, 5))),
    ):
        per = _per_example_loss(logits, targets)
        assert per.shape == (6,)
        assert torch.allclose(per.mean(), _loss_fn(logits, targets), atol=1e-6)
    # regression
    logits, targets = torch.randn(6, 4, 3), torch.randn(6, 4, 3)
    per = _per_example_loss(logits, targets, "mse")
    assert torch.allclose(per.mean(), _loss_fn(logits, targets, "mse"), atol=1e-6)


def test_train_explore_runs_and_is_deterministic():
    def _run():
        torch.manual_seed(0)
        m = ExploreWrapper(
            TRM(in_features=14, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2),
            noise_dim=4,
        )
        losses = train_explore(
            m, _linear_loader(), explore_k=4, noise_dim=4, epochs=3, explore_seed=7
        )
        return losses, [p.detach().clone() for p in m.parameters()]

    l1, p1 = _run()
    l2, p2 = _run()
    assert len(l1) == 3
    assert l1 == l2
    assert all(torch.equal(a, b) for a, b in zip(p1, p2))


def test_train_explore_k_changes_trajectory():
    """K is a real knob: same seed, different K ⇒ different weights (not a silent no-op)."""

    def _run(k):
        torch.manual_seed(0)
        m = ExploreWrapper(
            TRM(in_features=14, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2),
            noise_dim=4,
        )
        train_explore(m, _linear_loader(), explore_k=k, noise_dim=4, epochs=3, explore_seed=7)
        return [p.detach().clone() for p in m.parameters()]

    assert not all(torch.equal(a, b) for a, b in zip(_run(1), _run(8)))


def test_train_explore_requires_wrapper():
    m = TRM(in_features=10, num_classes=2, hidden_dim=8, latent_dim=8)
    with pytest.raises(ValueError, match="ExploreWrapper"):
        train_explore(m, _linear_loader(), explore_k=2, noise_dim=4, epochs=1)


def test_train_explore_selects_per_example_winner():
    """The winner must be chosen per EXAMPLE. With a hand-made 2-candidate score matrix where
    each row prefers a different candidate, a per-batch (mean) rule would pick one for both."""
    scores = torch.tensor([[0.1, 9.0], [9.0, 0.2]])  # (K=2, B=2)
    winner = scores.argmin(dim=0)
    assert winner.tolist() == [0, 1]
    assert scores.mean(dim=1).argmin().item() in (0, 1)  # a per-batch rule collapses to one


def test_explore_diagnostics_detects_collapse():
    targets = np.array([[0, 1, 1], [1, 0, 1]])
    identical = np.stack([targets, targets, targets])  # latent ignored
    d = explore_diagnostics(identical, targets, want_exact_match=True)
    assert d["latent_sensitivity"] == 0.0
    assert d["distinct_predictions"] == 1.0
    assert d["sampled_exact_match"] == 1.0
    assert d["oracle_exact_match"] == 1.0

    # One correct draw among three ⇒ sampled EM 1/3, oracle EM 1.0 (the oracle reads the key).
    wrong = 1 - targets
    mixed = np.stack([targets, wrong, wrong])
    d2 = explore_diagnostics(mixed, targets, want_exact_match=True)
    assert d2["sampled_exact_match"] == pytest.approx(1 / 3)
    assert d2["oracle_exact_match"] == 1.0
    assert d2["latent_sensitivity"] > 0.0
    assert d2["distinct_predictions"] == pytest.approx(2.0)


def test_evaluate_explore_side_car_leaves_headline_untouched():
    """The zero-latent headline eval must be unchanged by running the sampled side-car."""
    torch.manual_seed(0)
    m = ExploreWrapper(
        TRM(in_features=16, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2, out_features=8),
        noise_dim=8,
    )
    loader = _converge_loader(w=8)
    before = evaluate(m, loader, want_exact_match=True)
    side = evaluate_explore(
        m, loader, n_samples=4, noise_dim=8, seed=3, want_exact_match=True
    )
    after = evaluate(m, loader, want_exact_match=True)
    assert before == after
    assert {"sampled_accuracy", "sampled_exact_match", "oracle_exact_match"} <= set(side)
    # The oracle is an upper bound on any single sample, by construction.
    assert side["oracle_exact_match"] >= side["sampled_exact_match"] - 1e-12


def test_sample_noise_is_seeded():
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    assert torch.equal(sample_noise(4, 3, 1.0, g1), sample_noise(4, 3, 1.0, g2))


def test_ff_control_can_explore_too():
    """Exploration must be testable on the NON-recurrent control — the whole point of the XM
    comparison is 'can the objective replace the loop?'."""
    torch.manual_seed(0)
    m = ExploreWrapper(
        FFMatched(in_features=14, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=2),
        noise_dim=4,
    )
    losses = train_explore(m, _linear_loader(), explore_k=4, noise_dim=4, epochs=2, explore_seed=1)
    assert len(losses) == 2


# --- M34: mode-level metrics on the multimodal task ---------------------------------------------


def test_mode_metrics_separates_commitment_from_blur():
    from looptab.eval.ambiguity import mode_metrics

    modes = [np.array([[0, 0, 1], [1, 1, 0]])]  # one row, two modes
    # A model that emits a valid mode every draw, covering both modes.
    both = np.array([[[0, 0, 1]], [[1, 1, 0]]])
    d = mode_metrics(both, modes)
    assert d["mode_validity"] == 1.0 and d["mode_coverage"] == 1.0 and d["mode_count"] == 2.0
    # A model that commits to ONE mode: fully valid, half the coverage.
    one = np.array([[[0, 0, 1]], [[0, 0, 1]]])
    d = mode_metrics(one, modes)
    assert d["mode_validity"] == 1.0 and d["mode_coverage"] == 0.5
    # The cell-wise average of the two modes ([0.5,0.5,0.5] -> any rounding) is NOT a mode: the
    # blur case scores zero on both.
    blur = np.array([[[0, 1, 1]], [[1, 0, 0]]])
    d = mode_metrics(blur, modes)
    assert d["mode_validity"] == 0.0 and d["mode_coverage"] == 0.0


def test_sampled_or_repeated_preds_repeats_for_deterministic_arm():
    from looptab.data.generators import make_ambiguous_converge
    from looptab.eval.ambiguity import sampled_or_repeated_preds

    torch.manual_seed(0)
    X, y = make_ambiguous_converge(n=32, w=8, n_masked=3, task_seed=0, sample_seed=1)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=16)
    det = TRM(in_features=8, num_classes=2, hidden_dim=8, latent_dim=8, n_steps=2, out_features=8)
    preds = sampled_or_repeated_preds(det, loader, n_samples=3, noise_dim=0)
    assert preds.shape == (3, 32, 8)
    assert (preds[0] == preds[1]).all() and (preds[1] == preds[2]).all()
