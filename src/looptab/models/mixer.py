"""TRM variant with a cross-cell MIXING operator (M23 adversarial-review re-test).

The flat `TRM` (models/trm.py) concatenates the whole grid into ONE vector and refines it with a
structureless MLP — no per-cell / positional structure, no operator that lets cells communicate,
so it cannot express constraint propagation (the M23 review's decisive finding). This variant keeps
the grid as PER-CELL TOKENS ``(B, n_cells, cell_dim)`` and, each refinement step, applies a
**token-mixing MLP across the cell axis** (every cell sees every other cell — the propagation
operator) followed by a per-cell **channel MLP** — the MLP-mixer the TRM paper credits for its
Sudoku win (74.7%→87.4% vs a flat block). Weight-tied across steps, per-step readouts for deep
supervision, same ``(B, out_features, num_classes)`` output as the flat TRM so it is a drop-in for
the training/eval/control machinery.

``n_cells`` and ``cell_dim`` are derived from ``out_features`` (= #cells) and ``in_features`` (=
n_cells·cell_dim); appended distractor columns break that factorization and are unsupported here.
Like `trm_decoupled`, the 3-D batched matmuls are BLAS-order sensitive, so it is bit-reproducible
only at a fixed ``num_threads`` (committed runs pin 1).
"""

from typing import Optional

import torch
import torch.nn as nn

from .trm import RMSNorm


class TRMMixer(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        n_steps: int = 4,
        deep_supervision: bool = True,
        out_features: Optional[int] = None,
        use_rmsnorm: bool = False,
        n_latent: int = 1,
        token_hidden: Optional[int] = None,
    ):
        super().__init__()
        if out_features is None:
            raise ValueError("TRMMixer is multi-output only (out_features = number of cells).")
        if in_features % out_features != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by out_features ({out_features}) = "
                "n_cells; appended distractors are unsupported by the cell-mixing model."
            )
        if n_latent < 1:
            raise ValueError(f"n_latent must be >= 1, got {n_latent}")
        self.n_cells = out_features
        self.cell_dim = in_features // out_features
        self.num_classes = num_classes
        self.out_features = out_features
        self.n_steps = n_steps
        self.deep_supervision = deep_supervision
        self.n_latent = n_latent
        self.latent_dim = latent_dim

        in_dim = self.cell_dim + latent_dim + num_classes  # per-cell input: [x_cell, z, a]
        token_hidden = token_hidden if token_hidden is not None else self.n_cells
        # Token-mixing MLP over the CELL axis (applied per feature channel) — the cross-cell
        # propagation operator the flat TRM lacks. Residual, so it preserves the per-cell in_dim.
        self.token_mix = nn.Sequential(
            nn.Linear(self.n_cells, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, self.n_cells),
        )
        # Channel MLP: per-cell [x_cell, z, a] -> new latent.
        self.channel = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.readout = nn.Linear(latent_dim, num_classes)  # shared across cells
        self.norm = RMSNorm(latent_dim) if use_rmsnorm else nn.Identity()
        # Per-cell learned initial latent (doubles as a positional embedding — the structure the
        # flat TRM's single shared z0 lacks). Small init to break cell symmetry from step 0.
        self.z0 = nn.Parameter(torch.zeros(self.n_cells, latent_dim))
        nn.init.normal_(self.z0, std=0.02)

    def forward(
        self,
        X: torch.Tensor,
        n_steps: Optional[int] = None,
        init_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        B = X.shape[0]
        steps = n_steps if n_steps is not None else self.n_steps
        x_cells = X.view(B, self.n_cells, self.cell_dim)  # un-flatten the grid into tokens

        if init_state is None:
            z = self.z0.unsqueeze(0).expand(B, -1, -1)  # (B, n_cells, latent)
            a = torch.zeros(B, self.n_cells, self.num_classes, device=X.device)
        else:
            z, a = init_state

        all_logits = [] if self.deep_supervision else None
        for _ in range(steps):
            for _ in range(self.n_latent):
                inp = torch.cat([x_cells, z, a], dim=-1)  # (B, n_cells, in_dim)
                # Token mixing across cells (residual, preserves in_dim), then channel MLP → latent.
                mixed = inp + self.token_mix(inp.transpose(1, 2)).transpose(1, 2)
                z = self.norm(self.channel(mixed))
            a = self.readout(z)  # (B, n_cells, num_classes)
            if self.deep_supervision:
                all_logits.append(a)

        if return_state:
            return a, all_logits, (z, a)
        return a, all_logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _MixerBlock(nn.Module):
    """One untied mixer step: token-mix (cross-cell) + channel MLP + readout.

    Byte-for-byte the same computation as one ``TRMMixer`` step at ``n_latent=1`` — a
    standalone module so an UNTIED stack can reuse it without refactoring (and thus
    perturbing) the committed tied ``TRMMixer``.
    """

    def __init__(self, n_cells, cell_dim, num_classes, hidden_dim, latent_dim,
                 token_hidden, use_rmsnorm):
        super().__init__()
        in_dim = cell_dim + latent_dim + num_classes
        self.token_mix = nn.Sequential(
            nn.Linear(n_cells, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, n_cells),
        )
        self.channel = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.readout = nn.Linear(latent_dim, num_classes)
        self.norm = RMSNorm(latent_dim) if use_rmsnorm else nn.Identity()

    def forward(self, x_cells, z, a):
        inp = torch.cat([x_cells, z, a], dim=-1)  # (B, n_cells, in_dim)
        mixed = inp + self.token_mix(inp.transpose(1, 2)).transpose(1, 2)
        z = self.norm(self.channel(mixed))
        a = self.readout(z)
        return z, a


class UntiedMixerStack(nn.Module):
    """Depth/compute-matched UNTIED mixer stack — the §4b control for ``trm_mixer``.

    The ``TRMMixer`` block stacked ``n_steps`` times *without* weight tying: each step its
    own ``_MixerBlock`` (own token-mix / channel / readout / norm), carrying ``(z, a)`` across
    steps exactly as the tied loop does. The ONLY axis that differs from ``trm_mixer`` is the
    weight tying, so Δ(trm_mixer − untied_mixer) isolates whether weight-tied recurrence of the
    MIXING operator helps beyond an untied stack of mixer blocks (B1: "does the LOOP contribute,
    or is it the mixer ARCHITECTURE?"). Like ``UntiedStack`` this is NOT param-matched (~n_steps×
    the block params) — a labelled ceiling; the param-matched control is
    ``UntiedMixerStackMatched``. ``n_latent`` is fixed to 1 (the M24 setting). 3-D matmul ⇒ pin
    ``num_threads=1``.
    """

    def __init__(self, in_features, num_classes, hidden_dim=64, latent_dim=64, n_steps=4,
                 deep_supervision=True, out_features=None, use_rmsnorm=False, token_hidden=None):
        super().__init__()
        if out_features is None:
            raise ValueError("UntiedMixerStack is multi-output only (out_features = n cells).")
        if in_features % out_features != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by out_features ({out_features})."
            )
        self.n_cells = out_features
        self.cell_dim = in_features // out_features
        self.num_classes = num_classes
        self.out_features = out_features
        self.n_steps = n_steps
        self.deep_supervision = deep_supervision
        self.latent_dim = latent_dim
        token_hidden = token_hidden if token_hidden is not None else self.n_cells
        self.blocks = nn.ModuleList(
            _MixerBlock(self.n_cells, self.cell_dim, num_classes, hidden_dim, latent_dim,
                        token_hidden, use_rmsnorm)
            for _ in range(n_steps)
        )
        self.z0 = nn.Parameter(torch.zeros(self.n_cells, latent_dim))
        nn.init.normal_(self.z0, std=0.02)

    def forward(self, X, n_steps=None):
        # An untied stack has a fixed depth (one weight set per block); clamp like UntiedStack.
        B = X.shape[0]
        steps = self.n_steps if n_steps is None else min(n_steps, self.n_steps)
        x_cells = X.view(B, self.n_cells, self.cell_dim)
        z = self.z0.unsqueeze(0).expand(B, -1, -1)
        a = torch.zeros(B, self.n_cells, self.num_classes, device=X.device)
        all_logits = [] if self.deep_supervision else None
        for i in range(steps):
            z, a = self.blocks[i](x_cells, z, a)
            if self.deep_supervision:
                all_logits.append(a)
        return a, all_logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _mixer_block_params(n_cells, cell_dim, num_classes, hidden, latent, token_hidden, use_rmsnorm):
    """Analytic param count of ONE mixer block (token-mix + channel + readout + norm)."""
    in_dim = cell_dim + latent + num_classes
    token = (n_cells * token_hidden + token_hidden) + (token_hidden * n_cells + n_cells)
    channel = (in_dim * hidden + hidden) + (hidden * latent + latent)
    readout = latent * num_classes + num_classes
    norm = latent if use_rmsnorm else 0
    return token + channel + readout + norm


def _tied_mixer_params(n_cells, cell_dim, num_classes, hidden, latent, token_hidden, use_rmsnorm):
    """Param count of a tied ``TRMMixer`` = one block + per-cell z0 (matches TRMMixer.__init__)."""
    return (_mixer_block_params(n_cells, cell_dim, num_classes, hidden, latent, token_hidden,
                                use_rmsnorm) + n_cells * latent)


def _mixer_stack_params(n_cells, cell_dim, num_classes, w, use_rmsnorm, n_steps):
    """Param count of an ``UntiedMixerStack`` at hidden=latent=token_hidden=w (blocks + z0).

    ``token_hidden`` shrinks WITH the channel width in the matched control: the untied stack's 8
    token-mix layers dominate the budget, so matching by shrinking only the channel is infeasible
    (stays >budget at channel-width 1). Shrinking token_hidden=w too is the standard §4b tradeoff
    (untying forces narrower blocks — as ``UntiedStackMatched`` does), reported alongside the
    full-width ``untied_mixer`` ceiling so the capacity-constrained and unconstrained reads ship
    together.
    """
    block = _mixer_block_params(n_cells, cell_dim, num_classes, w, w, w, use_rmsnorm)
    return n_steps * block + n_cells * w


class UntiedMixerStackMatched(nn.Module):
    """Param-matched UNTIED mixer stack — the CLEAN weight-tying control for ``trm_mixer`` (§4b/§8).

    ``UntiedMixerStack`` is depth-matched but ~n_steps× over budget, so its Δ co-varies tying with
    capacity. This width-shrinks every block (hidden = latent = w) so the stack's TOTAL params ≈ the
    tied mixer's budget (same nearest-match search the flat ``UntiedStackMatched`` uses), holding
    capacity AND depth fixed and varying ONLY weight tying. ``token_hidden`` is held at the tied
    mixer's value (the token-mix is the operator under test); only the channel/latent width shrinks.
    Budget reference = a ``TRMMixer`` at the passed ``hidden_dim/latent_dim/token_hidden``.
    """

    def __init__(self, in_features, num_classes, hidden_dim=64, latent_dim=64, n_steps=4,
                 deep_supervision=True, out_features=None, use_rmsnorm=False, token_hidden=None):
        super().__init__()
        if out_features is None:
            raise ValueError("UntiedMixerStackMatched is multi-output only.")
        if in_features % out_features != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by out_features ({out_features})."
            )
        n_cells = out_features
        cell_dim = in_features // out_features
        th = token_hidden if token_hidden is not None else n_cells
        # Budget target = a tied TRMMixer at the given width (analytic, RNG-free).
        target = _tied_mixer_params(n_cells, cell_dim, num_classes, hidden_dim, latent_dim, th,
                                    use_rmsnorm)

        def stack_params(w):
            return _mixer_stack_params(n_cells, cell_dim, num_classes, w, use_rmsnorm, n_steps)

        w = 1
        while stack_params(w) < target:
            w += 1
        if abs(stack_params(w - 1) - target) < abs(stack_params(w) - target):
            w = max(1, w - 1)
        self.matched_width = w
        # token_hidden shrinks with the channel width (=w) so the untied stack fits the tied budget.
        self.inner = UntiedMixerStack(
            in_features, num_classes, hidden_dim=w, latent_dim=w, n_steps=n_steps,
            deep_supervision=deep_supervision, out_features=out_features,
            use_rmsnorm=use_rmsnorm, token_hidden=w,
        )

    def forward(self, X, n_steps=None):
        return self.inner(X, n_steps=n_steps)

    def count_params(self) -> int:
        return self.inner.count_params()
