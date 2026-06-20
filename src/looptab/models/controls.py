"""Matched controls for the TRM recurrent model.

(a) FFMatched: param-matched feedforward, no weight sharing, no loop.
(b) UntiedStack: same block stacked N times without weight tying (M2).
"""

import torch
import torch.nn as nn
from typing import Optional


def _count_trm_params(in_features, num_classes, hidden_dim, latent_dim, n_steps):
    """Compute TRM param count so we can match it in the feedforward control."""
    answer_dim = num_classes
    update_in = in_features + latent_dim + answer_dim
    update_params = (update_in * hidden_dim + hidden_dim) + (hidden_dim * latent_dim + latent_dim)
    readout_params = latent_dim * num_classes + num_classes
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
    ):
        super().__init__()
        target = _count_trm_params(in_features, num_classes, hidden_dim, latent_dim, n_steps)

        # Build a wide-enough 2-layer MLP with a chosen width w such that
        # params(in -> w -> w -> num_classes) ≈ target. Solve approximately.
        # params = in*w+w + w*w+w + w*num_classes+num_classes
        # We binary-search for w.
        def ff_params(w):
            return (in_features * w + w) + (w * w + w) + (w * num_classes + num_classes)

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
            nn.Linear(w, num_classes),
        )
        self._actual_params = sum(p.numel() for p in self.parameters())

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.net(X), None

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
