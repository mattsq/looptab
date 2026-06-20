"""Matched controls for the TRM recurrent model.

(a) FFMatched: param-matched feedforward, no weight sharing, no loop.
(b) UntiedStack: same block stacked N times without weight tying (M2).
"""

from typing import Optional

import torch
import torch.nn as nn


def _count_trm_params(in_features, num_classes, hidden_dim, latent_dim, out_features=None):
    """Compute TRM param count so we can match it in a control.

    TRM is weight-tied, so its parameter count is independent of `n_steps` (the same
    block is reused every step); step count is deliberately not an argument here.
    """
    answer_dim = out_features * num_classes if out_features is not None else num_classes
    update_in = in_features + latent_dim + answer_dim
    update_params = (update_in * hidden_dim + hidden_dim) + (hidden_dim * latent_dim + latent_dim)
    readout_params = latent_dim * answer_dim + answer_dim
    latent_param = latent_dim  # z0
    return update_params + readout_params + latent_param


class FFMatched(nn.Module):
    """
    Feedforward control matched to TRM in parameter count.

    We build a deep MLP whose total parameters ≈ TRM's, computed analytically.
    The architecture is a stack of linear+GELU layers; depth is chosen to absorb
    the budget while keeping width reasonable.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        n_steps: int = 4,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.out_features = out_features

        target = _count_trm_params(in_features, num_classes, hidden_dim, latent_dim, out_features)

        # Build a wide-enough 2-layer MLP with a chosen width w such that
        # params(in -> w -> w -> out_dim) ≈ target. Solve approximately.
        # params = in*w+w + w*w+w + w*out_dim+out_dim
        # We binary-search for w.
        out_dim = out_features * num_classes if out_features is not None else num_classes

        def ff_params(w):
            return (in_features * w + w) + (w * w + w) + (w * out_dim + out_dim)

        w = 1
        while ff_params(w) < target:
            w += 1
        # pick whichever is closer
        if abs(ff_params(w - 1) - target) < abs(ff_params(w) - target):
            w = max(1, w - 1)

        self.net = nn.Sequential(
            nn.Linear(in_features, w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.GELU(),
            nn.Linear(w, out_dim),
        )
        self._actual_params = sum(p.numel() for p in self.parameters())

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits = self.net(X)
        if self.out_features is not None:
            B = X.shape[0]
            logits = logits.view(B, self.out_features, self.num_classes)
        return logits, None

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UntiedStack(nn.Module):
    """
    Depth/compute-matched untied control (CLAUDE.md §4b).

    The TRM block stacked ``n_steps`` times *without* weight tying: each step gets
    its own ``update_net`` and ``readout``, but the per-step computation is identical
    to TRM (same shapes, same FLOPs, same depth). The only axis that differs from the
    TRM recurrent core is the weight tying — so Δ(TRM − UntiedStack) isolates whether
    weight-tied *recurrence* helps beyond mere depth (§8: "recurrence is algorithmic"
    vs "recurrence is dressed-up depth").

    This is deliberately **not** param-matched: it has ~``n_steps``× TRM's update/readout
    parameters by construction. The param-matched control is ``FFMatched`` (§4a); the two
    controls answer different questions and ship together. Deep supervision is supported
    (per-step readouts) so it can be ablated on the same axis as TRM, but the canonical
    M2 Δ compares the no-DS arms so only the tying axis varies.
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
        self.n_steps = n_steps
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.out_features = out_features

        answer_dim = out_features * num_classes if out_features is not None else num_classes
        self.answer_dim = answer_dim

        # One independent block per step (untied). Each block mirrors TRM's update_net
        # + readout exactly, so depth and per-step compute match the recurrent core.
        self.update_nets = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features + latent_dim + answer_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            for _ in range(n_steps)
        )
        self.readouts = nn.ModuleList(nn.Linear(latent_dim, answer_dim) for _ in range(n_steps))

        # Initial latent (a single learned init, as in TRM).
        self.z0 = nn.Parameter(torch.zeros(latent_dim))

    def forward(
        self, X: torch.Tensor, n_steps: Optional[int] = None
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """
        Returns:
            logits: (B, num_classes) or (B, out_features, num_classes) final-block logits
            all_logits: list of per-block logits if deep_supervision, else None

        ``n_steps`` is accepted for interface parity with TRM, but the untied stack has
        a *fixed* depth (one weight set per block): it cannot be unrolled past
        ``self.n_steps``. We therefore clamp to the available blocks — over-unrolling is
        simply not defined for an untied stack, which is exactly why it's the depth
        control rather than a recurrence.
        """
        B = X.shape[0]
        steps = self.n_steps if n_steps is None else min(n_steps, self.n_steps)
        z = self.z0.unsqueeze(0).expand(B, -1)  # (B, latent_dim)
        a = torch.zeros(B, self.answer_dim, device=X.device)  # answer state

        all_logits = [] if self.deep_supervision else None

        for i in range(steps):
            inp = torch.cat([X, z, a], dim=-1)
            z = self.update_nets[i](inp)
            a = self.readouts[i](z)
            if self.deep_supervision:
                if self.out_features is not None:
                    all_logits.append(a.view(B, self.out_features, self.num_classes))
                else:
                    all_logits.append(a)

        if self.out_features is not None:
            a = a.view(B, self.out_features, self.num_classes)

        return a, all_logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UntiedStackMatched(nn.Module):
    """
    Param-matched untied stack — the *clean* weight-tying control (§4b / §8).

    The plain ``UntiedStack`` is depth/compute-matched but **not** param-matched: untying
    a weight-tied loop necessarily multiplies the block params by ~``n_steps`` (~4× here),
    so Δ(TRM − UntiedStack) co-varies *tying* with *capacity*. This variant width-shrinks
    every block (``hidden = latent = w``) so the stack's **total** params ≈ the TRM loop's,
    exactly as ``FFMatched`` solves a width to the same budget. It therefore holds both
    capacity *and* depth fixed and varies **only** weight tying — so Δ(TRM − UntiedStackMatched)
    isolates whether weight-tied recurrence helps at a *fixed parameter budget*.

    Implemented by delegating to a width-``w`` ``UntiedStack``; ``w`` is chosen by the same
    nearest-match search ``FFMatched`` uses, against the TRM target from ``_count_trm_params``.
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
        target = _count_trm_params(in_features, num_classes, hidden_dim, latent_dim, out_features)
        answer_dim = out_features * num_classes if out_features is not None else num_classes

        # Total params of `n_steps` untied blocks at width w (hidden = latent = w), plus
        # one z0 of size w. Each block = update_net (2 linears) + readout (1 linear).
        def stack_params(w):
            update = ((in_features + w + answer_dim) * w + w) + (w * w + w)
            readout = w * answer_dim + answer_dim
            return n_steps * (update + readout) + w  # + z0

        w = 1
        while stack_params(w) < target:
            w += 1
        if abs(stack_params(w - 1) - target) < abs(stack_params(w) - target):
            w = max(1, w - 1)
        self.matched_width = w

        self.inner = UntiedStack(
            in_features,
            num_classes,
            hidden_dim=w,
            latent_dim=w,
            n_steps=n_steps,
            deep_supervision=deep_supervision,
            out_features=out_features,
        )

    def forward(
        self, X: torch.Tensor, n_steps: Optional[int] = None
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        return self.inner(X, n_steps=n_steps)

    def count_params(self) -> int:
        return self.inner.count_params()
