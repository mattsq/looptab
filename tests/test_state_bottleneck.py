"""M37: the answer-only state bottleneck for `trm_mixer` (opt-in, off by default)."""

import pytest
import torch

from looptab.models.mixer import TRMMixer


def _mixer(**kw):
    torch.manual_seed(0)
    return TRMMixer(in_features=8, num_classes=2, hidden_dim=16, latent_dim=8, n_steps=3,
                    out_features=8, use_rmsnorm=True, token_hidden=4, **kw)


def _x(n=5):
    return torch.randint(0, 2, (n, 8), generator=torch.Generator().manual_seed(1)).float()


def test_default_is_unchanged_and_bottleneck_keeps_the_parameters():
    """"none" builds the same parameters and outputs as before; the bottleneck adds none."""
    base, none, hard = _mixer(), _mixer(state_bottleneck="none"), _mixer(state_bottleneck="hard")
    X = _x()
    assert torch.equal(base(X)[0], none(X)[0])
    assert base.count_params() == hard.count_params()
    for (n1, p1), (n2, p2) in zip(base.named_parameters(), hard.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_hard_bottleneck_is_markov_in_the_decoded_state():
    """Restarting from the decoded state after step 1 reproduces steps 2.. exactly: the loop
    carries nothing but the observable state (no input recall, no latent carry)."""
    m = _mixer(state_bottleneck="hard")
    X = _x()
    with torch.no_grad():
        _, logits = m(X, n_steps=3)
        s1 = (torch.softmax(logits[0], -1)[..., 1] > 0.5).float()
        _, restart = m(s1, n_steps=2)
    assert torch.allclose(logits[1], restart[0]) and torch.allclose(logits[2], restart[1])


def test_soft_bottleneck_ignores_the_original_input_after_step_one():
    """Two inputs with the same decoded s_1 give the same later steps (soft mode, via restart)."""
    m = _mixer(state_bottleneck="soft")
    X = _x()
    with torch.no_grad():
        _, logits = m(X, n_steps=2)
        s1 = torch.softmax(logits[0], -1)[..., 1]
        _, restart = m(s1, n_steps=1)
    assert torch.allclose(logits[1], restart[0], atol=1e-6)


def test_hard_bottleneck_passes_gradient_through_the_state():
    m = _mixer(state_bottleneck="hard")
    _, logits = m(_x(), n_steps=3)
    logits[-1].sum().backward()
    assert m.token_mix[0].weight.grad is not None
    assert m.token_mix[0].weight.grad.abs().sum() > 0


def test_bottleneck_rejects_unsupported_shapes():
    with pytest.raises(ValueError):
        TRMMixer(in_features=16, num_classes=2, out_features=8, state_bottleneck="hard")
    with pytest.raises(ValueError):
        TRMMixer(in_features=8, num_classes=3, out_features=8, state_bottleneck="soft")
