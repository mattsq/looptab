"""Training loop. Supports deep supervision for TRM-style models."""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that handles both single-output (B,C) and multi-output (B,W,C)."""
    if targets.ndim == 1:
        return nn.functional.cross_entropy(logits, targets)
    # multi-output: logits (B, W, C), targets (B, W)
    B, W, C = logits.shape
    return nn.functional.cross_entropy(logits.view(B * W, C), targets.view(B * W))


def train(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    deep_supervision_weight: float = 1.0,
    device: str = "cpu",
    verbose: bool = False,
) -> list[float]:
    """Train model; return per-epoch train losses."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits, all_logits = model(X)
            loss = _loss_fn(logits, y)
            if all_logits is not None and deep_supervision_weight > 0:
                ds_loss = sum(
                    _loss_fn(step_logits, y) for step_logits in all_logits
                ) / len(all_logits)
                loss = loss + deep_supervision_weight * ds_loss
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={avg:.4f}")

    return losses
