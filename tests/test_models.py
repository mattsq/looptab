"""Shape and forward-pass sanity tests for TRM and FFMatched."""

import pytest
import torch

from looptab.models.controls import FFMatched
from looptab.models.trm import TRM


@pytest.fixture
def trm():
    return TRM(in_features=16, num_classes=2, hidden_dim=32, latent_dim=32,
               n_steps=3, deep_supervision=True)


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
    m = TRM(in_features=16, num_classes=2, hidden_dim=32, latent_dim=32,
            n_steps=3, deep_supervision=False)
    X = torch.randn(4, 16)
    logits, all_logits = m(X)
    assert logits.shape == (4, 2)
    assert all_logits is None


def test_ff_output_shape(ff):
    X = torch.randn(8, 16)
    logits, extra = ff(X)
    assert logits.shape == (8, 2)
    assert extra is None


def test_param_counts_roughly_matched():
    in_f, nc, hd, ld, ns = 20, 2, 64, 64, 4
    trm = TRM(in_features=in_f, num_classes=nc, hidden_dim=hd, latent_dim=ld, n_steps=ns)
    ff = FFMatched(in_features=in_f, num_classes=nc, hidden_dim=hd, latent_dim=ld, n_steps=ns)
    ratio = ff.count_params() / trm.count_params()
    # Should be within 20% of TRM's param count
    assert 0.8 <= ratio <= 1.2, f"FF/TRM param ratio = {ratio:.3f}"
