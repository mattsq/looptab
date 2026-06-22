"""TRM-style iterative latent refinement model.

Based on Jolicoeur-Martineau 2025 (arXiv 2510.04871). A single tiny 2-layer net
is applied repeatedly: each step updates a latent z and produces a readout.
Deep supervision is enabled by returning per-step logits.
"""

from typing import Optional

import torch
import torch.nn as nn


class TRM(nn.Module):
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

        # Projects input + latent + answer into the update space
        answer_dim = out_features * num_classes if out_features is not None else num_classes
        self.update_net = nn.Sequential(
            nn.Linear(in_features + latent_dim + answer_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.readout = nn.Linear(latent_dim, answer_dim)

        # Initial latent
        self.z0 = nn.Parameter(torch.zeros(latent_dim))

    def forward(
        self,
        X: torch.Tensor,
        n_steps: Optional[int] = None,
        init_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        """
        Args:
            init_state: optional ``(z, a)`` to resume from instead of the learned ``z0`` /
                zero answer. ``a`` is the *flat* (B, answer_dim) answer state. Lets the
                progressive-loss routine (M7, Deep Thinking) detach an intermediate rollout
                and continue from it. ``None`` reproduces the canonical fresh start exactly.
            return_state: if True, also return the final flat ``(z, a)`` so a rollout can be
                detached and resumed. Composition is bit-identical: unrolling ``n+m`` steps
                equals unrolling ``n`` (return_state) then resuming ``m`` from that state.

        Returns:
            logits: (B, num_classes) or (B, out_features, num_classes) final step logits
            all_logits: list of per-step logits if deep_supervision, else None
            state (only if return_state): final flat ``(z, a)``
        """
        B = X.shape[0]
        steps = n_steps if n_steps is not None else self.n_steps

        if self.out_features is not None:
            answer_dim = self.out_features * self.num_classes
        else:
            answer_dim = self.num_classes

        if init_state is None:
            z = self.z0.unsqueeze(0).expand(B, -1)  # (B, latent_dim)
            a = torch.zeros(B, answer_dim, device=X.device)  # flat answer state
        else:
            z, a = init_state

        all_logits = [] if self.deep_supervision else None

        for _ in range(steps):
            inp = torch.cat([X, z, a], dim=-1)
            z = self.update_net(inp)
            a = self.readout(z)
            if self.deep_supervision:
                if self.out_features is not None:
                    all_logits.append(a.view(B, self.out_features, self.num_classes))
                else:
                    all_logits.append(a)

        out = a.view(B, self.out_features, self.num_classes) if self.out_features is not None else a

        if return_state:
            return out, all_logits, (z, a)
        return out, all_logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
