"""TRM-style iterative latent refinement model.

Based on Jolicoeur-Martineau 2025 (arXiv 2510.04871). A single tiny 2-layer net
is applied repeatedly: each step updates a latent z and produces a readout.
Deep supervision is enabled by returning per-step logits.
"""

import torch
import torch.nn as nn
from typing import Optional


class TRM(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        n_steps: int = 4,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # Projects input + latent + answer into the update space
        answer_dim = num_classes  # logit vector as the "answer state"
        self.update_net = nn.Sequential(
            nn.Linear(in_features + latent_dim + answer_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.readout = nn.Linear(latent_dim, num_classes)

        # Initial latent
        self.z0 = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """
        Returns:
            logits: (B, num_classes) final step logits
            all_logits: list of (B, num_classes) per-step if deep_supervision, else None
        """
        B = X.shape[0]
        z = self.z0.unsqueeze(0).expand(B, -1)  # (B, latent_dim)
        a = torch.zeros(B, self.num_classes, device=X.device)  # answer state

        all_logits = [] if self.deep_supervision else None

        for _ in range(self.n_steps):
            inp = torch.cat([X, z, a], dim=-1)
            z = self.update_net(inp)
            a = self.readout(z)
            if self.deep_supervision:
                all_logits.append(a)

        return a, all_logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
