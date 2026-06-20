"""Training loop. Supports deep supervision for TRM-style models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that handles both single-output (B,C) and multi-output (B,W,C)."""
    if targets.ndim == 1:
        return nn.functional.cross_entropy(logits, targets)
    # multi-output: logits (B, W, C), targets (B, W). Use reshape (not view) because
    # step-aligned DS targets are non-contiguous trajectory slices traj[:, i, :].
    B, W, C = logits.shape
    return nn.functional.cross_entropy(logits.reshape(B * W, C), targets.reshape(B * W))


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
                ds_loss = sum(_loss_fn(step_logits, y) for step_logits in all_logits) / len(
                    all_logits
                )
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


def _forward_steps(model: nn.Module, X: torch.Tensor, n_steps: int):
    """Call a model, passing ``n_steps`` to recurrent/untied arms, omitting it for plain MLPs.

    Lets the curriculum trainer drive every arm through one code path: the loop and untied
    stacks unroll to the sampled depth, while ``ff_matched`` (no depth notion) ignores it.
    """
    try:
        return model(X, n_steps=n_steps)
    except TypeError:
        return model(X)


def train_curriculum(
    model: nn.Module,
    traj_loader: DataLoader,
    *,
    T_min: int,
    T_max: int,
    ds_mode: str = "final",
    deep_supervision_weight: float = 1.0,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> list[float]:
    """Train across a CA-depth curriculum (M3b).

    Each batch samples a depth ``T ~ Uniform{T_min..T_max}``, unrolls the model to T, and
    supervises against the trajectory. Two distinguishable DS modes:
      - ``"final"``: loss on the final readout vs s_T (+ optional final-state DS on every step,
        the M0–M3a behaviour) — the contrast arm that isolates step-alignment.
      - ``"step_aligned"``: loop step i ↔ intermediate state s_i. Requires the model to emit
        exactly T per-step readouts (n_steps == T per batch), else raises — the alignment is
        otherwise undefined.

    Depth sampling uses a dedicated seeded generator so the per-batch T schedule is identical
    across arms (a fair contrast) and reproducible, independent of the dataloader shuffle.
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator()  # CPU generator drives integer depth sampling deterministically
    gen.manual_seed(seed)
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X, traj in traj_loader:
            X = X.to(device)
            traj = traj.to(device)  # (B, T_max, w) int64
            T = int(torch.randint(T_min, T_max + 1, (1,), generator=gen).item())
            s_T = traj[:, T - 1, :]  # (B, w)

            opt.zero_grad()
            final_logits, all_logits = _forward_steps(model, X, T)

            if ds_mode == "step_aligned":
                if all_logits is None:
                    raise ValueError("step_aligned DS requires a model emitting per-step readouts")
                if len(all_logits) != T:
                    raise ValueError(
                        f"step_aligned DS requires n_steps == T; got {len(all_logits)} readouts "
                        f"for T={T}. Couple the arm's depth to the curriculum param."
                    )
                # Loop step i supervised against intermediate CA state s_i (i = 1..T).
                loss = sum(
                    _loss_fn(all_logits[i], traj[:, i, :]) for i in range(T)
                ) / T
            else:  # "final"
                loss = _loss_fn(final_logits, s_T)
                if all_logits is not None and deep_supervision_weight > 0:
                    ds_loss = sum(_loss_fn(sl, s_T) for sl in all_logits) / len(all_logits)
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
