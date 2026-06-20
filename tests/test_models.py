"""Shape and forward-pass sanity tests for TRM and FFMatched."""

import pytest
import torch

from looptab.models.controls import FFMatched, UntiedStack
from looptab.models.trm import TRM


@pytest.fixture
def trm():
    return TRM(
        in_features=16,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=3,
        deep_supervision=True,
    )


@pytest.fixture
def ff():
    return FFMatched(in_features=16, num_classes=2, hidden_dim=32, latent_dim=32, n_steps=3)


def test_trm_output_shape(trm):
    X = torch.randn(8, 16)
    logits, all_logits = trm(X)
    assert logits.shape == (8, 2)
    assert len(all_logits) == 3
    for step_logits in all_logits:
        assert step_logits.shape == (8, 2)


def test_trm_no_deep_supervision():
    m = TRM(
        in_features=16,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=3,
        deep_supervision=False,
    )
    X = torch.randn(4, 16)
    logits, all_logits = m(X)
    assert logits.shape == (4, 2)
    assert all_logits is None


def test_ff_output_shape(ff):
    X = torch.randn(8, 16)
    logits, extra = ff(X)
    assert logits.shape == (8, 2)
    assert extra is None


def test_trm_multi_output_shape():
    m = TRM(
        in_features=16,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=3,
        deep_supervision=True,
        out_features=8,
    )
    X = torch.randn(5, 16)
    logits, all_logits = m(X)
    assert logits.shape == (5, 8, 2)
    assert len(all_logits) == 3
    for step_logits in all_logits:
        assert step_logits.shape == (5, 8, 2)


def test_trm_step_override():
    m = TRM(
        in_features=16,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=3,
        deep_supervision=True,
    )
    X = torch.randn(5, 16)
    logits, all_logits = m(X, n_steps=5)
    assert logits.shape == (5, 2)
    assert len(all_logits) == 5


def test_ff_multi_output_shape():
    m = FFMatched(
        in_features=16,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=3,
        out_features=8,
    )
    X = torch.randn(5, 16)
    logits, extra = m(X)
    assert logits.shape == (5, 8, 2)
    assert extra is None


def test_untied_stack_output_shape():
    m = UntiedStack(
        in_features=16,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=3,
        deep_supervision=True,
    )
    X = torch.randn(8, 16)
    logits, all_logits = m(X)
    assert logits.shape == (8, 2)
    assert len(all_logits) == 3
    for step_logits in all_logits:
        assert step_logits.shape == (8, 2)


def test_untied_stack_no_deep_supervision():
    m = UntiedStack(
        in_features=16,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=3,
        deep_supervision=False,
    )
    X = torch.randn(4, 16)
    logits, all_logits = m(X)
    assert logits.shape == (4, 2)
    assert all_logits is None


def test_untied_stack_multi_output_shape():
    m = UntiedStack(
        in_features=16,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=3,
        deep_supervision=True,
        out_features=8,
    )
    X = torch.randn(5, 16)
    logits, all_logits = m(X)
    assert logits.shape == (5, 8, 2)
    assert len(all_logits) == 3
    for step_logits in all_logits:
        assert step_logits.shape == (5, 8, 2)


def test_untied_stack_is_untied():
    """Each block must have independent weights (the whole point of §4b)."""
    m = UntiedStack(in_features=16, num_classes=2, hidden_dim=32, latent_dim=32, n_steps=3)
    assert len(m.update_nets) == 3
    assert len(m.readouts) == 3
    # Distinct parameter tensors per block (no tying / sharing).
    first = m.update_nets[0][0].weight
    for blk in m.update_nets[1:]:
        assert blk[0].weight is not first


def test_untied_stack_clamps_overunroll():
    """An untied stack has fixed depth: asking for more steps clamps to n_steps."""
    m = UntiedStack(
        in_features=16,
        num_classes=2,
        hidden_dim=32,
        latent_dim=32,
        n_steps=3,
        deep_supervision=True,
    )
    X = torch.randn(5, 16)
    _, all_logits = m(X, n_steps=10)
    assert len(all_logits) == 3  # cannot unroll past the available blocks


def test_untied_stack_has_more_params_than_trm():
    """§4b is depth/compute-matched, NOT param-matched: ~n_steps× TRM's block params."""
    kw = dict(in_features=20, num_classes=2, hidden_dim=64, latent_dim=64, n_steps=4)
    trm = TRM(**kw)
    untied = UntiedStack(**kw)
    assert untied.count_params() > 2 * trm.count_params()


def test_param_counts_roughly_matched():
    in_f, nc, hd, ld, ns = 20, 2, 64, 64, 4
    # Single-output case
    trm = TRM(
        in_features=in_f,
        num_classes=nc,
        hidden_dim=hd,
        latent_dim=ld,
        n_steps=ns,
    )
    ff = FFMatched(
        in_features=in_f,
        num_classes=nc,
        hidden_dim=hd,
        latent_dim=ld,
        n_steps=ns,
    )
    ratio = ff.count_params() / trm.count_params()
    assert 0.8 <= ratio <= 1.2, f"FF/TRM param ratio (single) = {ratio:.3f}"

    # Multi-output case
    trm_mo = TRM(
        in_features=in_f,
        num_classes=nc,
        hidden_dim=hd,
        latent_dim=ld,
        n_steps=ns,
        out_features=10,
    )
    ff_mo = FFMatched(
        in_features=in_f,
        num_classes=nc,
        hidden_dim=hd,
        latent_dim=ld,
        n_steps=ns,
        out_features=10,
    )
    ratio_mo = ff_mo.count_params() / trm_mo.count_params()
    assert 0.8 <= ratio_mo <= 1.2, f"FF/TRM param ratio (multi) = {ratio_mo:.3f}"
