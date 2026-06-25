"""Training loop. Supports deep supervision for TRM-style models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EMA:
    """Exponential moving average of model weights (M18 ingredient 2).

    TRM's ablation ranks EMA as its 2nd-largest training knob (no-EMA 79.9% vs 87.4% on
    Sudoku-Extreme); it stabilizes small-data / weight-tied training and is the natural
    variance-reducer for this repo's seed-sensitive regime. ``decay`` is the smoothing
    coefficient (TRM uses 0.999). ``update`` is called after every optimizer step; ``copy_to``
    folds the averaged weights into the model so evaluation runs on the EMA copy (the canonical
    TRM eval). Deterministic given the weight trajectory, so it does not break reproducibility.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        # Shadow copy of the trainable params, detached from the graph.
        self.shadow = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.copy_(self.shadow[n])


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
    ema_decay: float | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> list[float]:
    """Train model; return per-epoch train losses.

    ``ema_decay`` (M18 ingredient 2): if set, maintain an EMA of the weights and fold it into
    the model at the end, so evaluation runs on the averaged weights. ``None`` = no EMA,
    bit-identical to the pre-M18 routine.
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = EMA(model, ema_decay) if ema_decay is not None else None
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
            if ema is not None:
                ema.update(model)
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={avg:.4f}")

    if ema is not None:
        ema.copy_to(model)
    return losses


def train_deep_supervision(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    n_sup: int,
    carry: bool = True,
    n_steps: int | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    deep_supervision_weight: float = 1.0,
    ema_decay: float | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> list[float]:
    """Canonical TRM/HRM deep supervision (M18 ingredient 1).

    The repo's existing "deep supervision" is per-step readout losses *inside one fully
    back-propagated forward* — NOT the mechanism the ARC autopsy credits. This routine adds the
    real thing: an OUTER loop of ``n_sup`` supervised passes per batch, where the recurrent
    state ``(z, a)`` is **carried across passes and detached** between them. Each pass runs the
    loop for ``n_steps``, takes a loss, steps the optimizer, then detaches ``(z, a)`` and feeds
    them as the init of the next pass. This emulates a very deep network (``n_sup × n_steps``
    effective depth) without long backprop-through-time — the bounded gradient horizon is one
    pass. Requires a model whose ``forward`` accepts ``init_state`` / ``return_state`` (TRM,
    TRMDecoupled). ``n_sup=1`` reduces to one ordinary supervised forward.

    ``deep_supervision_weight`` still weights the within-pass per-step readout losses (when the
    model emits them); the cross-pass carry is the new axis. ``ema_decay`` folds an EMA of the
    weights into the model at the end (ingredient 2). Deterministic given seed: the detach and
    EMA are pure functions of the weight/state trajectory.

    ``carry`` (M18 review fix B1 — the COMPUTE-MATCHED control). With ``carry=True`` (default) the
    detached ``(z, a)`` is fed as the init of the next pass — the actual deep-supervision
    mechanism. With ``carry=False`` every pass restarts from the fresh ``z0`` / zero answer, so the
    routine becomes *exactly ``n_sup`` independent supervised forwards per batch* — the SAME
    optimizer-step count and per-pass compute, MINUS the carry. Δ(carry − no-carry) therefore
    isolates whether the detached **carry** helps beyond the raw 4× step-count it also buys, closing
    the §8 confound the bundle/ablation otherwise leaves open.
    """
    if n_sup < 1:
        raise ValueError(f"n_sup must be >= 1, got {n_sup}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = EMA(model, ema_decay) if ema_decay is not None else None
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_passes = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            state = None  # fresh (learned z0 / zero answer) at the start of each batch
            for _ in range(n_sup):
                opt.zero_grad()
                logits, all_logits, state = model(
                    X, n_steps=n_steps, init_state=state, return_state=True
                )
                loss = _loss_fn(logits, y)
                if all_logits is not None and deep_supervision_weight > 0:
                    ds_loss = sum(_loss_fn(sl, y) for sl in all_logits) / len(all_logits)
                    loss = loss + deep_supervision_weight * ds_loss
                loss.backward()
                opt.step()
                if ema is not None:
                    ema.update(model)
                # Detach the carried state so the next pass's gradient stops here — the bounded
                # horizon that lets effective depth grow with n_sup without long BPTT. With
                # carry=False, drop the state so the next pass restarts fresh (the compute-matched
                # control: same step count, no carry).
                state = (state[0].detach(), state[1].detach()) if carry else None
                epoch_loss += loss.item()
                n_passes += 1
        avg = epoch_loss / max(n_passes, 1)
        losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={avg:.4f}")

    if ema is not None:
        ema.copy_to(model)
    return losses


def _forward_steps(model: nn.Module, X: torch.Tensor, n_steps: int):
    """Call a model with an explicit unroll depth, uniformly across arms.

    Every model in this repo accepts ``n_steps`` (the recurrent/untied arms unroll to it;
    ``FFMatched`` accepts and ignores it for interface parity), so we pass it directly. We do
    NOT swallow a ``TypeError`` here: a future model whose ``forward`` lacks ``n_steps`` should
    fail loudly rather than be silently retried as ``model(X)`` and mis-unrolled to its default
    depth — which under step-aligned DS would surface as a confusing readout-count mismatch.
    """
    return model(X, n_steps=n_steps)


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


def train_progressive(
    model: nn.Module,
    traj_loader: DataLoader,
    *,
    T_min: int,
    T_max: int,
    ds_mode: str = "progressive_final",
    alpha: float = 0.5,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> list[float]:
    """Deep Thinking progressive-loss training (M7; Bansal et al. 2022, arXiv 2202.05826).

    The depth-extrapolation lever. Each batch samples a depth ``T ~ Uniform{T_min..T_max}``
    (the same curriculum as ``train_curriculum``) and a gradient budget ``k ~ Uniform{1..T}``.
    It then runs the recurrent loop for ``(T−k)`` steps with **gradients detached**, and ``k``
    further steps **with** gradient resuming from that detached state — supervising only the
    grad steps. This penalizes iteration-count-specific behaviour (the operator must make
    progress from an *arbitrary* intermediate state, not only from ``s_0``), pushing the loop
    toward a repeatable step operator / path-independent steady state. "Recall" (re-injecting
    the input every step) is already built into ``TRM`` (``cat[X, z, a]``).

    Two target alignments:
      - ``"progressive_final"``: the k grad steps are supervised against the final state ``s_T``.
      - ``"progressive_step"``: step-aligned — the k grad steps are supervised against the CA
        states ``s_{T−k+1..T}`` (combines M3b's step-alignment with the progressive detach).

    Loss = ``alpha·L_progressive + (1−alpha)·L_full``, where ``L_full`` is the standard full-T
    forward (with gradient) supervised the same way — Deep Thinking keeps both so the model
    stays anchored. Depth/k sampling uses a dedicated seeded generator so the schedule is
    identical across arms and reproducible.
    """
    if ds_mode not in ("progressive_final", "progressive_step"):
        raise ValueError(f"train_progressive expects a progressive ds_mode, got {ds_mode!r}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator()
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
            k = int(torch.randint(1, T + 1, (1,), generator=gen).item())  # grad steps, 1..T

            opt.zero_grad()

            # (T−k) detached warmup steps, then k gradient steps resuming from that state.
            init = None
            if T - k > 0:
                with torch.no_grad():
                    _, _, state = model(X, n_steps=T - k, return_state=True)
                init = (state[0].detach(), state[1].detach())
            out_prog, logits_prog, _ = model(X, n_steps=k, init_state=init, return_state=True)

            # Standard full-T term (with gradient), supervised the same way (the anchor term).
            out_full, logits_full, _ = model(X, n_steps=T, return_state=True)

            if ds_mode == "progressive_step":
                if logits_prog is None or logits_full is None:
                    raise ValueError(
                        "progressive_step DS requires a model emitting per-step readouts"
                    )
                # The k grad steps map to CA depths (T−k+1 .. T) → traj indices (T−k .. T−1).
                loss_prog = sum(
                    _loss_fn(logits_prog[i], traj[:, (T - k) + i, :]) for i in range(k)
                ) / k
                loss_full = sum(_loss_fn(logits_full[i], traj[:, i, :]) for i in range(T)) / T
            else:  # progressive_final
                s_T = traj[:, T - 1, :]
                loss_prog = _loss_fn(out_prog, s_T)
                loss_full = _loss_fn(out_full, s_T)

            loss = alpha * loss_prog + (1.0 - alpha) * loss_full
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={avg:.4f}")

    return losses
