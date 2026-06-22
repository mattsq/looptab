"""Decoupled-head TRM variant — the M10 coherence-mechanism ablation.

M9 established (on the multi-output fixed-point task `converge`) that the weight-tied
loop buys *whole-row coherence*: at matched token-accuracy it produces coherent whole
rows a shallow MLP cannot (loop vs ff @ w=24: ΔEM +0.133, p=.002). M9 did **not** isolate
*why*. The canonical ``TRM`` (models/trm.py) refines a SINGLE shared latent ``z`` and feeds
the FULL flat answer ``a`` (all ``w`` cells) back into every update — so each output cell's
refinement conditions on the current estimate of every other cell. That cross-cell coupling
is the obvious candidate for "whole-row coherence."

``TRMDecoupled`` is the ablation that severs exactly that coupling and nothing else: every
output cell carries its **own** latent slice and sees **only its own** answer during
refinement. Everything else is held identical to ``TRM`` — weight-tied recurrence (one shared
update net reused every step *and* across cells), the same input ``X`` re-injected each step
("recall"), the same per-step readout interface (deep supervision), and the **same total
parameter budget** (the per-cell latent width is solved to the loop's budget exactly as
``FFMatched`` / ``UntiedStackMatched`` do). The axis that differs is joint-state vs
per-cell-state refinement — total budget matched and capacity not handicapped (the per-cell
net is wider: m≈73-80 vs the joint 64), though the parameter *allocation* necessarily differs
(the per-cell ``z0`` below costs ~8-13% of budget vs ~0.4% for the joint loop's single ``z0``).
So it isolates the mechanism (cross-cell information flow), not a single-knob weight edit.

Pre-registered honesty fork (CLAUDE.md §8), tested by M10:
  - decoupled collapses to ff-like coherence  ⇒  the JOINT state is the mechanism
    (cross-cell information flow during refinement is what buys whole-row coherence).
  - decoupled keeps the coherence edge       ⇒  recurrence per se drives it, the joint
    readout is incidental.

Per-cell identity: each cell gets its own learned initial latent ``z0`` (shape ``(w, m)``),
exactly mirroring ``TRM``'s single learned ``z0`` but one-per-cell. This is what makes the
cells distinguishable (with a fully shared net and ``X`` shared across cells, a single shared
``z0`` would make every cell compute the identical update forever). It carries cell identity,
NOT any other cell's dynamic state, so the decoupling is clean.
"""

from typing import Optional

import torch
import torch.nn as nn

from .controls import _count_trm_params


def _count_decoupled_params(in_features, num_classes, m, out_features):
    """Total params of a width-``m`` decoupled head (hidden = latent = m).

    Weight-tied across steps AND across cells, so the update net + readout are counted once
    (not multiplied by ``n_steps`` or ``out_features``); only the per-cell initial latent
    ``z0`` scales with ``out_features`` (= the cell count ``w``).
    """
    w = out_features
    # update net: Linear(in + m + C -> m), GELU, Linear(m -> m)
    update = ((in_features + m + num_classes) * m + m) + (m * m + m)
    readout = m * num_classes + num_classes  # Linear(m -> C), shared across cells
    z0 = w * m  # per-cell learned init latent
    return update + readout + z0


class TRMDecoupled(nn.Module):
    """TRM with per-cell (decoupled) refinement instead of a joint latent/answer state.

    Param-matched to the ``TRM`` loop by solving the per-cell latent width ``m`` to the loop's
    budget (``_count_trm_params``), the same nearest-match search ``UntiedStackMatched`` uses.
    Multi-output only: the decoupled-head question is meaningless for a single output.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        n_steps: int = 4,
        deep_supervision: bool = True,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        if out_features is None:
            raise ValueError(
                "TRMDecoupled is a multi-output ablation (per-cell refinement); "
                "out_features must be set."
            )
        self.n_steps = n_steps
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.out_features = out_features

        # Match the loop's parameter budget: solve the per-cell width m so the decoupled head's
        # TOTAL params ≈ TRM(hidden=latent=latent_dim)'s, exactly as FFMatched/UntiedStackMatched.
        target = _count_trm_params(in_features, num_classes, hidden_dim, latent_dim, out_features)
        m = 1
        while _count_decoupled_params(in_features, num_classes, m, out_features) < target:
            m += 1
        if abs(
            _count_decoupled_params(in_features, num_classes, m - 1, out_features) - target
        ) < abs(_count_decoupled_params(in_features, num_classes, m, out_features) - target):
            m = max(1, m - 1)
        self.cell_latent_dim = m

        # ONE update net + readout, reused every step and across all cells (weight-tied, like
        # TRM). Per-cell input is [X, z_c, a_c] — no other cell's state ever enters.
        self.update_net = nn.Sequential(
            nn.Linear(in_features + m + num_classes, m),
            nn.GELU(),
            nn.Linear(m, m),
        )
        self.readout = nn.Linear(m, num_classes)

        # Per-cell learned initial latent — the cells' only identity signal. Unlike TRM's
        # single ``z0`` (zeros is fine for one latent), this MUST be randomly initialized: with
        # a fully shared net and ``X`` shared across cells, a zero ``z0`` leaves every cell
        # computing the identical update forever, and the inter-cell symmetry breaks only
        # slowly through per-cell gradients (empirically the arm sits at the majority baseline
        # for many epochs). A small normal init differentiates the cells from step 0 — standard
        # embedding practice — so the decoupled head trains on the same footing as the joint
        # loop and the ablation isn't confounded by an optimization artifact. Deterministic:
        # the runner seeds torch before building each arm, so this draw is reproducible.
        self.z0 = nn.Parameter(torch.empty(out_features, m))
        nn.init.normal_(self.z0, std=0.02)

    def forward(
        self,
        X: torch.Tensor,
        n_steps: Optional[int] = None,
        init_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        """
        Args mirror ``TRM.forward`` for interface parity (the curriculum/extrapolation harness
        calls every arm uniformly). State is ``(z, a)`` with ``z`` (B, w, m) and ``a`` (B, w, C).

        Returns:
            logits: (B, out_features, num_classes) final-step logits
            all_logits: list of per-step (B, out_features, num_classes) logits if deep_supervision
            state (only if return_state): final ``(z, a)``
        """
        B = X.shape[0]
        w = self.out_features
        steps = n_steps if n_steps is not None else self.n_steps

        # Broadcast the shared input to every cell once: (B, w, in_features).
        X_cell = X.unsqueeze(1).expand(B, w, -1)

        if init_state is None:
            z = self.z0.unsqueeze(0).expand(B, w, -1)  # (B, w, m) — per-cell init
            a = torch.zeros(B, w, self.num_classes, device=X.device)  # per-cell answer
        else:
            z, a = init_state

        all_logits = [] if self.deep_supervision else None

        for _ in range(steps):
            # Per-cell update: concat over the feature dim only; cells never mix.
            inp = torch.cat([X_cell, z, a], dim=-1)  # (B, w, in + m + C)
            z = self.update_net(inp)  # shared net applied per cell
            a = self.readout(z)  # (B, w, C)
            if self.deep_supervision:
                all_logits.append(a)

        if return_state:
            return a, all_logits, (z, a)
        return a, all_logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
