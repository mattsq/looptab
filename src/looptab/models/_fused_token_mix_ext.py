"""Lazy JIT loader for the fused cross-cell token-mix CUDA extension (csrc/fused_token_mix.cu).

Compiled on first actual use (``TRMMixer(..., use_fused_kernel=True)``), not at import time, so
importing ``looptab.models`` never requires a working CUDA/MSVC toolchain.
"""

import sys
from pathlib import Path

import torch

_CSRC = Path(__file__).parent / "csrc" / "fused_token_mix.cu"

# Kept in sync with FOR_EACH_SUPPORTED_SHAPE in fused_token_mix.cu — a closed set of (n_cells,
# token_hidden) pairs compiled as fixed template instantiations (needed for register-resident
# unrolling in the kernel, so this can't be an arbitrary shape). Covers the repo's real
# `trm_mixer` configs: sudoku (36, 64 — configs/experiments/m23_sudoku_mixer_*.yaml), ETTh1
# forecasting (7, 8 — m26_etth1_forecast.yaml), weather forecasting (21, 8 —
# m26_weather_forecast.yaml), plus (6, 6) for fast unit tests. Adding a new shape means adding an
# FWD/BWD entry to the .cu file's macro AND here, then re-validating gradients
# (tests/test_fused_mixer.py) before trusting it — do not just add the tuple.
SUPPORTED_SHAPES = {(36, 64), (7, 8), (21, 8), (6, 6)}

_ext = None


def load_extension():
    """Compile (once per process, cached thereafter) and return the extension module."""
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load

    extra_ldflags = None
    if sys.platform == "win32":
        # A `uv`-created venv doesn't carry a `libs/pythonXYZ.lib` (only the real interpreter
        # install does), so the MSVC linker fails with LNK1104 unless pointed at it explicitly.
        # `sys.base_prefix` is the real install regardless of which venv is active.
        extra_ldflags = [f"/LIBPATH:{Path(sys.base_prefix) / 'libs'}"]

    try:
        _ext = load(
            name="looptab_fused_token_mix",
            sources=[str(_CSRC)],
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(
            "TRMMixer(use_fused_kernel=True) needs a working CUDA extension build toolchain "
            f"({type(e).__name__}: {e}). On Windows this needs MSVC on PATH — source "
            r'"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build'
            r'\vcvars64.bat" in the SAME shell invocation that runs this (a separate process does '
            "not inherit it). Set use_fused_kernel=False to run without it."
        ) from e
    return _ext


class FusedTokenMix(torch.autograd.Function):
    """Autograd wrapper around the compiled ``tm_forward``/``tm_backward`` kernels.

    Numerically exact (verified against the eager ``Sequential`` token-mix via gradcheck at every
    entry in ``SUPPORTED_SHAPES``, see ``tests/test_fused_mixer.py``), not an approximation — this
    is a pure reimplementation of the same computation, not a precision tradeoff like AMP.
    """

    @staticmethod
    def forward(ctx, inp, w1, b1, w2, b2):
        ext = load_extension()
        out, h, pre = ext.tm_forward(inp, w1, b1, w2, b2)
        ctx.save_for_backward(inp, w1, w2, h, pre)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = load_extension()
        inp, w1, w2, h, pre = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_inp, dpre_h = ext.tm_backward(w1, w2, pre, grad_out)
        grad_w1 = torch.bmm(dpre_h, inp.transpose(1, 2)).sum(0)
        grad_b1 = dpre_h.sum(dim=(0, 2))
        grad_w2 = torch.bmm(grad_out, h.transpose(1, 2)).sum(0)
        grad_b2 = grad_out.sum(dim=(0, 2))
        return grad_inp, grad_w1, grad_b1, grad_w2, grad_b2
