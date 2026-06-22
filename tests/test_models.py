"""Shape and forward-pass sanity tests for TRM and FFMatched."""

import pytest
import torch

from looptab.models.controls import FFMatched, UntiedStack, UntiedStackMatched
from looptab.models.decoupled import TRMDecoupled
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


def test_trm_return_state_composition_bit_identical():
    """M7: unrolling n+m steps == unrolling n (return_state) then resuming m. Bit-identical.

    This is the invariant the progressive-loss detach relies on (Deep Thinking, M7).
    """
    for out_features in (None, 7):
        m = TRM(
            in_features=12,
            num_classes=2,
            hidden_dim=32,
            latent_dim=32,
            n_steps=4,
            deep_supervision=False,
            out_features=out_features,
        )
        m.eval()
        X = torch.randn(8, 12)
        with torch.no_grad():
            full, _ = m(X, n_steps=7)
            part, _, state = m(X, n_steps=3, return_state=True)
            resumed, _ = m(X, n_steps=4, init_state=state)
        torch.testing.assert_close(resumed, full, rtol=0, atol=0)


def test_trm_init_state_none_unchanged():
    """A fresh forward (init_state=None) is identical with and without return_state."""
    m = TRM(in_features=10, num_classes=2, hidden_dim=16, latent_dim=16, n_steps=4)
    m.eval()
    X = torch.randn(6, 10)
    with torch.no_grad():
        a, _ = m(X, n_steps=4)
        b, _, _ = m(X, n_steps=4, return_state=True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


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
    """§4b is depth/compute-matched, NOT param-matched: ~n_steps× TRM's block params.
    Assert close to n_steps× (not merely >2×) so a half-tied regression is caught."""
    kw = dict(in_features=20, num_classes=2, hidden_dim=64, latent_dim=64, n_steps=4)
    trm = TRM(**kw)
    untied = UntiedStack(**kw)
    ratio = untied.count_params() / trm.count_params()
    assert 3.5 <= ratio <= 4.0, f"untied/TRM ratio = {ratio:.3f} (expect ~n_steps=4)"


def test_untied_matched_is_param_matched():
    """The clean control: width-shrunk untied stack with total params ≈ TRM's (§8)."""
    for out_features in (None, 9):
        kw = dict(
            in_features=20 if out_features is None else 13,
            num_classes=2,
            hidden_dim=64,
            latent_dim=64,
            n_steps=4,
            out_features=out_features,
        )
        trm = TRM(**kw)
        matched = UntiedStackMatched(**kw)
        ratio = matched.count_params() / trm.count_params()
        assert 0.8 <= ratio <= 1.2, f"untied_matched/TRM ratio = {ratio:.3f} (out={out_features})"
        # It is genuinely untied (more blocks than the loop) but narrower than the full
        # untied stack, so its width is below the reference 64.
        assert matched.matched_width < 64


def test_untied_matched_output_shape():
    m = UntiedStackMatched(
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


def test_untied_matched_clamps_overunroll():
    """Fixed-depth like the plain untied stack: over-unrolling clamps to n_steps."""
    m = UntiedStackMatched(in_features=16, num_classes=2, hidden_dim=32, latent_dim=32, n_steps=3)
    X = torch.randn(4, 16)
    logits, all_logits = m(X, n_steps=9)
    assert logits.shape == (4, 2)
    assert len(all_logits) == 3


def test_decoupled_multi_output_shape():
    m = TRMDecoupled(
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


def test_decoupled_no_deep_supervision():
    m = TRMDecoupled(
        in_features=16, num_classes=2, hidden_dim=32, latent_dim=32, n_steps=3,
        deep_supervision=False, out_features=8,
    )
    X = torch.randn(4, 16)
    logits, all_logits = m(X)
    assert logits.shape == (4, 8, 2)
    assert all_logits is None


def test_decoupled_requires_out_features():
    """The decoupled-head ablation is meaningless for a single output (no whole row)."""
    with pytest.raises(ValueError):
        TRMDecoupled(in_features=16, num_classes=2, hidden_dim=32, latent_dim=32, n_steps=3)


def test_decoupled_step_override():
    m = TRMDecoupled(
        in_features=16, num_classes=2, hidden_dim=32, latent_dim=32, n_steps=3,
        deep_supervision=True, out_features=6,
    )
    X = torch.randn(5, 16)
    logits, all_logits = m(X, n_steps=5)
    assert logits.shape == (5, 6, 2)
    assert len(all_logits) == 5


def test_decoupled_is_param_matched():
    """Per-cell width is solved to the TRM loop's budget, exactly like UntiedStackMatched (§8)."""
    for out_features in (9, 24, 32):
        kw = dict(
            in_features=out_features + 8,  # converge: in = w + distractors
            num_classes=2,
            hidden_dim=64,
            latent_dim=64,
            n_steps=6,
            out_features=out_features,
        )
        trm = TRM(**kw)
        dec = TRMDecoupled(**kw)
        ratio = dec.count_params() / trm.count_params()
        assert 0.8 <= ratio <= 1.2, f"decoupled/TRM ratio = {ratio:.3f} (out={out_features})"


def test_decoupled_no_cross_cell_leakage():
    """THE decoupling invariant (M10): cell c's output depends ONLY on (X, z_c, a_c).

    Perturbing one cell's refinement state must leave every *other* cell's output bit-identical
    — this is precisely what severs the joint-state coupling the canonical TRM has, and what the
    M10 ablation tests. A regression that lets cells mix (e.g. a stray reduction over the cell
    dim) is caught here.
    """
    m = TRMDecoupled(
        in_features=12, num_classes=2, hidden_dim=32, latent_dim=32, n_steps=3,
        deep_supervision=False, out_features=5,
    )
    m.eval()
    X = torch.randn(4, 12)
    B, w, dim = 4, 5, m.cell_latent_dim
    z = torch.randn(B, w, dim)
    a = torch.randn(B, w, 2)
    with torch.no_grad():
        out1, _ = m(X, init_state=(z, a))
        z2, a2 = z.clone(), a.clone()
        z2[:, 2, :] += torch.randn(B, dim)
        a2[:, 2, :] += torch.randn(B, 2)
        out2, _ = m(X, init_state=(z2, a2))
    others = [c for c in range(w) if c != 2]
    torch.testing.assert_close(out1[:, others, :], out2[:, others, :], rtol=0, atol=0)
    assert not torch.allclose(out1[:, 2, :], out2[:, 2, :]), "perturbed cell must change"


def test_decoupled_return_state_composition_bit_identical():
    """Unrolling n+m steps == unrolling n (return_state) then resuming m (interface parity)."""
    m = TRMDecoupled(
        in_features=12, num_classes=2, hidden_dim=32, latent_dim=32, n_steps=4,
        deep_supervision=False, out_features=7,
    )
    m.eval()
    X = torch.randn(8, 12)
    with torch.no_grad():
        full, _ = m(X, n_steps=7)
        _, _, state = m(X, n_steps=3, return_state=True)
        resumed, _ = m(X, n_steps=4, init_state=state)
    torch.testing.assert_close(resumed, full, rtol=0, atol=0)


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
