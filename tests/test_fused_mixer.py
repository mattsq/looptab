"""Correctness tests for the fused cross-cell token-mix CUDA kernel (models/csrc/fused_token_mix.cu)
and its `trm_mixer_fused` arm (models/mixer.py::TRMMixerFused).

Construction-time validation (unsupported shapes, disable_token_mix conflict) needs no CUDA and
always runs. Everything that actually executes the kernel needs CUDA + a working extension build
toolchain (MSVC on Windows) and is skipped module-wide otherwise, since compiling it is the point
of `_fused_kernel_available()` below (not a cheap check).
"""

import pytest
import torch

from looptab.models.mixer import TRMMixer, TRMMixerFused
from looptab.registry import get_model
from looptab.train.loop import train


def test_registry_resolves_trm_mixer_fused_at_a_supported_shape():
    m = get_model(
        "trm_mixer_fused", in_features=30, num_classes=3, out_features=6,
        hidden_dim=8, latent_dim=4, n_steps=2, token_hidden=6,
    )
    assert isinstance(m, TRMMixerFused)
    assert m.use_fused_kernel is True


def test_unsupported_shape_raises_at_construction():
    """(n_cells=5, token_hidden=5) is not in the compiled shape table -- must fail loudly at
    construction, not silently fall back to eager or fail deep inside a training run."""
    with pytest.raises(ValueError, match="not a compiled kernel shape"):
        TRMMixer(
            in_features=25, num_classes=3, out_features=5,
            hidden_dim=8, latent_dim=4, n_steps=2, token_hidden=5, use_fused_kernel=True,
        )


def test_disable_token_mix_with_fused_kernel_raises():
    with pytest.raises(ValueError, match="nothing to fuse"):
        TRMMixer(
            in_features=30, num_classes=3, out_features=6,
            hidden_dim=8, latent_dim=4, n_steps=2, token_hidden=6,
            use_fused_kernel=True, disable_token_mix=True,
        )


def _fused_kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from looptab.models._fused_token_mix_ext import load_extension

        load_extension()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _fused_kernel_available(), reason="needs CUDA + a working CUDA extension build toolchain"
)


def _build(cls, seed=0):
    torch.manual_seed(seed)
    return cls(
        in_features=30, num_classes=3, out_features=6, hidden_dim=8, latent_dim=4,
        n_steps=2, n_latent=1, token_hidden=6, use_rmsnorm=True, deep_supervision=True,
    ).to("cuda")


def test_fused_kernel_matches_eager_forward_and_gradients():
    m_eager = _build(TRMMixer)
    m_fused = _build(TRMMixerFused)
    X = torch.randn(9, 30, device="cuda")
    y = torch.randint(0, 3, (9, 6), device="cuda")

    logits_e, all_e = m_eager(X)
    logits_f, all_f = m_fused(X)
    assert torch.allclose(logits_e, logits_f, atol=1e-4, rtol=1e-3)

    ce = torch.nn.functional.cross_entropy
    loss_e = sum(ce(sl.reshape(-1, 3), y.reshape(-1)) for sl in all_e)
    loss_f = sum(ce(sl.reshape(-1, 3), y.reshape(-1)) for sl in all_f)
    loss_e.backward()
    loss_f.backward()
    for (ne, pe), (nf, pf) in zip(m_eager.named_parameters(), m_fused.named_parameters()):
        assert ne == nf
        assert torch.allclose(pe.grad, pf.grad, atol=1e-3, rtol=1e-2), f"grad mismatch at {ne}"


def _learnable_batch(n=64, seed=1):
    """Labels are an actual (if easy) function of X — a fixed random per-cell linear readout,
    argmax'd to a class — not independent noise, so a working model can genuinely learn them."""
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 30, generator=gen)
    W = torch.randn(30, 6 * 3, generator=gen)
    y = (X @ W).view(n, 6, 3).argmax(dim=-1)
    return X, y


def test_fused_kernel_trains_and_converges():
    torch.manual_seed(0)
    X, y = _learnable_batch()
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=32)
    m = _build(TRMMixerFused)
    losses = train(m, loader, epochs=80, lr=2e-2, device="cuda")
    assert losses[-1] < losses[0] * 0.6


def test_fused_kernel_composes_with_cuda_graph():
    """Regression guard for the exact bug the POC hit: a kernel that launches on the wrong CUDA
    stream trains 'successfully' (parameters visibly move, no crash, no NaN) but the loss
    oscillates/plateaus instead of decreasing, under cuda_graph capture specifically -- invisible
    without it. If this ever regresses, it will look like a passing-but-non-converging run, not a
    crash, so the assertion is on genuine convergence, not just 'ran without error'."""
    torch.manual_seed(0)
    X, y = _learnable_batch()
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=32)
    m = _build(TRMMixerFused)
    losses = train(m, loader, epochs=80, lr=2e-2, device="cuda", cuda_graph=True)
    assert len(losses) == 80
    assert all(v == v for v in losses)  # no NaN
    assert losses[-1] < losses[0] * 0.6  # genuinely decreasing, not plateaued
