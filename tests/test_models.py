"""Shape and forward-pass sanity tests for TRM and FFMatched."""

import pytest
import torch

from looptab.models.controls import FFMatched
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
