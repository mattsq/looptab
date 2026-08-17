"""Training loop. Supports deep supervision for TRM-style models."""

import math

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


def _loss_fn(logits: torch.Tensor, targets: torch.Tensor, loss_type: str = "ce") -> torch.Tensor:
    """Task loss over single-output (B,C) or multi-output (B,W,C) outputs.

    ``loss_type="ce"`` (default) = cross-entropy for classification (bit-identical to before).
    ``loss_type="mse"`` = mean-squared error for M26 forecasting REGRESSION: the readout
    ``(B, M, H)`` is interpreted as raw forecast values (no softmax) against a same-shape float
    target.
    """
    if loss_type == "mse":
        return nn.functional.mse_loss(logits, targets.to(logits.dtype))
    if targets.ndim == 1:
        return nn.functional.cross_entropy(logits, targets)
    # multi-output: logits (B, W, C), targets (B, W). Use reshape (not view) because
    # step-aligned DS targets are non-contiguous trajectory slices traj[:, i, :].
    B, W, C = logits.shape
    return nn.functional.cross_entropy(logits.reshape(B * W, C), targets.reshape(B * W))


class _GraphedStepBase:
    """Shared mechanics for capturing a training step as a CUDA graph.

    Handles the four things every captured step needs, so `_CapturedStep` (standard path) and
    `_CapturedPassStep` (deep-supervision/ACT passes) stay small:

    - **Fixed-address gradient buffers.** Replay needs stable tensor addresses;
      ``opt.zero_grad(set_to_none=True)`` would free/realloc them each call. Allocate once,
      zero in-place inside the captured region instead.
    - **Warmup + undo.** Warmup runs the step for real a few times on a side stream (required
      before capture, so cuDNN/allocator one-time init doesn't get baked into the graph) and is
      then UNDONE — parameters, optimizer moment buffers, and the EMA shadow are restored to
      their pre-warmup values via a full SNAPSHOT (not zeroing): capture can now also happen
      MID-training (`cuda_graph_tail` captures the ragged-batch graph whenever that shape first
      appears), where zeroing the moments would erase real optimizer state. At step 0 the two
      are equivalent (state tensors created by warmup are reset to zero). Not bit-identical to
      eager (warmup can select different cuDNN/cuBLAS algorithms).
    - **Optional fp16 autocast with a STATIC loss scale.** ``GradScaler``'s dynamic scale needs
      a host-side inf/nan check every step, which a replayed graph cannot do. A fixed scale
      (multiply the loss, divide the grads in-graph) is the standard graph-compatible AMP
      recipe; overflow shows up as a non-finite epoch loss, checked once per epoch by the
      caller. Weights stay fp32 masters exactly as in eager AMP.
    - **A shared memory pool**, so multiple graphs (tail-shape graph, fresh/carry pass graphs)
      coexist without each reserving its own private pool — the #18 memory-competition failure
      mode on 8GB cards is per-pool, so sharing keeps the footprint at one pool per arm.
    """

    def __init__(self, model, opt, ema, *, amp=False, amp_static_loss_scale=8192.0,
                 warmup_iters=5, pool=None):
        self.model = model
        self.opt = opt
        self.ema = ema
        self.use_amp = bool(amp)
        self.amp_scale = float(amp_static_loss_scale)
        self.warmup_iters = warmup_iters
        self.pool = pool
        for p in model.parameters():
            if p.requires_grad and p.grad is None:
                p.grad = torch.zeros_like(p)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp)

    def _zero_grads(self):
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad.zero_()

    def _backward_and_step(self, loss):
        if self.use_amp:
            (loss * self.amp_scale).backward()
            inv_scale = 1.0 / self.amp_scale
            for p in self.model.parameters():
                if p.requires_grad:
                    p.grad.mul_(inv_scale)
        else:
            loss.backward()
        self.opt.step()
        if self.ema is not None:
            self.ema.update(self.model)

    def _snapshot(self):
        params = {
            n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
        }
        opt_state = {}
        for group in self.opt.param_groups:
            for p in group["params"]:
                state = self.opt.state.get(p)
                if state:
                    opt_state[id(p)] = {
                        k: v.clone() for k, v in state.items() if torch.is_tensor(v)
                    }
        ema_state = (
            {n: v.clone() for n, v in self.ema.shadow.items()} if self.ema is not None else None
        )
        return params, opt_state, ema_state

    def _restore(self, snapshot):
        params, opt_state, ema_state = snapshot
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.copy_(params[n])
            for group in self.opt.param_groups:
                for p in group["params"]:
                    state = self.opt.state.get(p)
                    if not state:
                        continue
                    saved = opt_state.get(id(p))
                    for key, val in state.items():
                        if not torch.is_tensor(val):
                            continue
                        if saved is not None and key in saved:
                            val.copy_(saved[key])  # mid-training capture: restore real moments
                        else:
                            val.zero_()  # state created BY warmup: reset to the step-0 state
            if self.ema is not None:
                for n, v in self.ema.shadow.items():
                    v.copy_(ema_state[n])

    def _capture(self, step_fns):
        """Warm up every closure, then capture each as its own graph on one shared pool.

        Returns ``[(graph, outputs), ...]`` aligned with ``step_fns``; parameters/optimizer/EMA
        are restored to their pre-warmup values afterwards, so the first replay is the genuine
        next training step.
        """
        snapshot = self._snapshot()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.warmup_iters):
                for fn in step_fns:
                    fn()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        if self.pool is None:
            self.pool = torch.cuda.graphs.graph_pool_handle()
        captured = []
        for fn in step_fns:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = fn()
            captured.append((graph, out))
        self._restore(snapshot)
        return captured


class _CapturedStep(_GraphedStepBase):
    """A training step (zero-grad + forward + loss + backward + opt.step [+ EMA]) captured as a
    CUDA graph, replayable against new batches via ``.run(X, y)``.

    Built once per batch SHAPE (the first batch of training; with ``cuda_graph_tail`` also the
    smaller last batch when it first appears — see `_GraphedStepBase` for why mid-training
    capture is safe). ``amp=True`` runs the captured forward/loss under fp16 autocast with a
    static loss scale (see `_GraphedStepBase`); ``amp=False`` is the pre-existing fp32 capture.
    """

    def __init__(self, model, opt, ema, X0, y0, deep_supervision_weight, loss_type,
                 warmup_iters=5, amp=False, amp_static_loss_scale=8192.0, pool=None):
        super().__init__(
            model, opt, ema, amp=amp, amp_static_loss_scale=amp_static_loss_scale,
            warmup_iters=warmup_iters, pool=pool,
        )
        self.static_X = torch.empty_like(X0)
        self.static_y = torch.empty_like(y0)
        self.static_X.copy_(X0)
        self.static_y.copy_(y0)
        self.batch_size = int(X0.shape[0])

        def _step():
            self._zero_grads()
            with self._autocast():
                logits, all_logits = model(self.static_X)
                loss = _loss_fn(logits, self.static_y, loss_type)
                if all_logits is not None and deep_supervision_weight > 0:
                    ds_loss = sum(
                        _loss_fn(sl, self.static_y, loss_type) for sl in all_logits
                    ) / len(all_logits)
                    loss = loss + deep_supervision_weight * ds_loss
            self._backward_and_step(loss)
            return loss

        [(self.graph, self.static_loss)] = self._capture([_step])

    def run(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.static_X.copy_(X)
        self.static_y.copy_(y)
        self.graph.replay()
        return self.static_loss


class _CapturedPassStep(_GraphedStepBase):
    """One deep-supervision PASS (zero-grad + forward-from-state + loss + backward + opt.step
    [+ EMA] [+ ACT halt loss]) captured as CUDA graphs, replayed ``n_sup`` times per batch.

    The DS-family routines (`train_deep_supervision`, `train_act`) run a fixed number of
    IDENTICALLY-SHAPED passes per batch, differing only in the carried ``(z, a)`` state — a
    shape-static inner loop, so graph capture applies pass-wise even though a single whole-batch
    graph cannot represent the detach-between-passes structure. Two graphs are captured on one
    shared pool:

    - **fresh**: ``init_state=None`` (learned z0 / zero answer) — the first pass of every batch
      (and EVERY pass when ``carry=False``, the compute-matched control).
    - **carry**: ``init_state=(static_z_in, static_a_in)`` — passes 2..n_sup; the previous
      pass's detached output state is copied into the static input buffers between replays,
      which is exactly the detach the eager routine performs (copy_ breaks the graph history).

    ``halt_weight`` (train_act) adds the halting-head BCE inside the captured region — the halt
    target (per-example exact-match) is argmax/eq/all, all capturable. ``amp`` uses the static
    loss scale (see `_GraphedStepBase`). Same warmup-undo semantics as `_CapturedStep`, so the
    first replay is the genuine first optimizer step.
    """

    def __init__(self, model, opt, ema, X0, y0, *, n_steps, carry, n_sup,
                 deep_supervision_weight, halt_weight=None, warmup_iters=5,
                 amp=False, amp_static_loss_scale=8192.0, pool=None):
        super().__init__(
            model, opt, ema, amp=amp, amp_static_loss_scale=amp_static_loss_scale,
            warmup_iters=warmup_iters, pool=pool,
        )
        self.static_X = torch.empty_like(X0)
        self.static_y = torch.empty_like(y0)
        self.static_X.copy_(X0)
        self.static_y.copy_(y0)
        self.batch_size = int(X0.shape[0])

        # Probe (no_grad, no optimizer mutation) for the carried state's shapes; its values also
        # seed the carry graph's warmup with realistic magnitudes.
        #
        # The carry buffers are held in X's dtype (fp32), NOT the autocast output dtype. Under
        # autocast the loop's `z`/`a` come back fp16; `torch.cat([X, z, a])` would type-promote
        # them back to fp32 anyway, so this is not a correctness fix but a shape/dtype CONTRACT:
        # a replayed graph needs static input buffers whose dtype never changes, and pinning them
        # to the input dtype makes the eager and captured paths carry state at the same precision.
        with torch.no_grad(), self._autocast():
            _, _, (z_probe, a_probe) = model(
                self.static_X, n_steps=n_steps, init_state=None, return_state=True
            )
        self.static_z_in = z_probe.detach().to(X0.dtype).clone()
        self.static_a_in = a_probe.detach().to(X0.dtype).clone()

        def _pass(init_state):
            self._zero_grads()
            with self._autocast():
                logits, all_logits, (z, a) = model(
                    self.static_X, n_steps=n_steps, init_state=init_state, return_state=True
                )
                loss = _loss_fn(logits, self.static_y)
                if all_logits is not None and deep_supervision_weight > 0:
                    ds_loss = sum(
                        _loss_fn(sl, self.static_y) for sl in all_logits
                    ) / len(all_logits)
                    loss = loss + deep_supervision_weight * ds_loss
                if halt_weight is not None:
                    with torch.no_grad():
                        correct = logits.argmax(dim=-1) == self.static_y
                        is_solved = (
                            correct.all(dim=-1).float()
                            if self.static_y.ndim > 1
                            else correct.float()
                        )
                    halt_logit = model.halt_head(z).squeeze(-1)
                    loss = loss + halt_weight * nn.functional.binary_cross_entropy_with_logits(
                        halt_logit, is_solved
                    )
            self._backward_and_step(loss)
            return loss, z.detach(), a.detach()

        fns = [lambda: _pass(None)]
        self.has_carry_graph = bool(carry) and n_sup > 1
        if self.has_carry_graph:
            fns.append(lambda: _pass((self.static_z_in, self.static_a_in)))
        captured = self._capture(fns)
        self.graph_fresh, (self.loss_fresh, self.z_out_fresh, self.a_out_fresh) = captured[0]
        if self.has_carry_graph:
            self.graph_carry, (self.loss_carry, self.z_out_carry, self.a_out_carry) = captured[1]

    def replay_fresh(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """First pass of a batch (or every pass under carry=False, X/y already staged)."""
        self.static_X.copy_(X)
        self.static_y.copy_(y)
        self.graph_fresh.replay()
        self._last = (self.z_out_fresh, self.a_out_fresh)
        return self.loss_fresh

    def replay_fresh_again(self) -> torch.Tensor:
        """carry=False passes 2..n_sup: same batch, fresh init — no copies needed."""
        self.graph_fresh.replay()
        self._last = (self.z_out_fresh, self.a_out_fresh)
        return self.loss_fresh

    def replay_carry(self) -> torch.Tensor:
        """Pass i>1 under carry=True: feed the previous pass's detached state forward."""
        z_prev, a_prev = self._last
        self.static_z_in.copy_(z_prev)
        self.static_a_in.copy_(a_prev)
        self.graph_carry.replay()
        self._last = (self.z_out_carry, self.a_out_carry)
        return self.loss_carry


def train(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    deep_supervision_weight: float = 1.0,
    ema_decay: float | None = None,
    loss_type: str = "ce",
    device: str = "cpu",
    amp: bool = False,
    cuda_graph: bool = False,
    cuda_graph_tail: bool = False,
    amp_static_loss_scale: float = 8192.0,
    microbatch_size: int | None = None,
    verbose: bool = False,
) -> list[float]:
    """Train model; return per-epoch train losses.

    ``ema_decay`` (M18 ingredient 2): if set, maintain an EMA of the weights and fold it into
    the model at the end, so evaluation runs on the averaged weights. ``None`` = no EMA,
    bit-identical to the pre-M18 routine. ``loss_type`` ("ce" default / "mse" for M26 regression)
    selects the task loss; "ce" is bit-identical to the pre-M26 routine.

    ``amp`` (opt-in, CUDA-only): run the forward/loss under fp16 autocast and scale gradients.
    Weights stay fp32 masters and evaluation is untouched (fp32), so only the training arithmetic
    changes. **When ``amp=False`` this routine is bit-identical to the pre-AMP one**: an autocast
    context with ``enabled=False`` and a ``GradScaler`` with ``enabled=False`` are documented
    no-ops (``scale``/``step``/``update`` degrade to ``loss``/``opt.step()``/nothing).

    ``cuda_graph`` (opt-in, CUDA-only): capture the whole step into a CUDA graph on the first
    batch and replay it thereafter — see ``_CapturedStep`` and the ``TrainConfig.cuda_graph``
    docstring for the mechanism and caveats. By default the last batch of an epoch is dropped
    if it is smaller than the captured shape (a graph needs a fixed shape); ``cuda_graph_tail``
    (opt-in) instead captures a SECOND graph for that tail shape on the same memory pool, so no
    batch is dropped (matches eager batch coverage; still not bit-identical to eager).

    ``amp`` + ``cuda_graph`` TOGETHER (opt-in): the captured step runs under fp16 autocast with
    the STATIC loss scale ``amp_static_loss_scale`` — ``GradScaler``'s dynamic scale needs a
    host-side inf check per step, which replay can't do. Overflow therefore surfaces as a
    non-finite epoch loss, checked once per epoch here (raises with guidance to lower the
    scale). ``amp=False, cuda_graph=False`` is bit-identical to the pre-AMP routine.
    """
    if microbatch_size is not None and microbatch_size < 1:
        raise ValueError(f"microbatch_size must be >= 1, got {microbatch_size}")
    if microbatch_size is not None and cuda_graph:
        raise ValueError("train(): microbatch_size and cuda_graph cannot be combined yet.")
    model = model.to(device)
    use_cuda_graph = cuda_graph and torch.device(device).type == "cuda"
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, capturable=use_cuda_graph
    )
    ema = EMA(model, ema_decay) if ema_decay is not None else None
    losses = []

    # fp16 tensor cores are a CUDA feature; on CPU autocast(float16) would be a slow no-win, so
    # `amp` is inert there rather than an error (configs stay portable between cpu/cuda boxes).
    # `cuda_graph` is inert on CPU for the same reason (no CUDA graphs to capture).
    use_amp = amp and torch.device(device).type == "cuda"
    # Under a captured graph the dynamic GradScaler is unusable (host-side inf checks); the
    # static-scale path inside _GraphedStepBase takes over, so the scaler stays disabled there.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not use_cuda_graph)
    captured_steps: dict[int, _CapturedStep] = {}  # batch_size -> graph (tail adds a 2nd entry)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        # The graph and microbatch paths are explicitly opt-in, so keep their diagnostic loss
        # accumulation on-device and synchronize only once per epoch.  Calling `.item()` after
        # every graph replay serializes the CPU and GPU, throwing away much of graph replay's
        # launch-ahead benefit.  float64 accumulation reproduces Python's sum of float32 losses.
        async_epoch_loss = (
            torch.zeros((), dtype=torch.float64, device=device)
            if use_cuda_graph or microbatch_size is not None
            else None
        )
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            if use_cuda_graph:
                B = int(X.shape[0])
                captured = captured_steps.get(B)
                if captured is None:
                    if not captured_steps or cuda_graph_tail:
                        # First shape seen — or, with cuda_graph_tail, the ragged tail shape's
                        # first appearance (mid-training capture; see _GraphedStepBase). The
                        # graphs share one memory pool so the footprint stays one-pool-per-arm.
                        shared_pool = next(iter(captured_steps.values())).pool if (
                            captured_steps
                        ) else None
                        captured = _CapturedStep(
                            model, opt, ema, X, y, deep_supervision_weight, loss_type,
                            amp=use_amp, amp_static_loss_scale=amp_static_loss_scale,
                            pool=shared_pool,
                        )
                        captured_steps[B] = captured
                    else:
                        continue  # drop the ragged last batch — the graph's shape is fixed
                loss_tensor = captured.run(X, y).detach()
                async_epoch_loss.add_(loss_tensor.to(torch.float64))
                loss_val = None
            elif microbatch_size is not None and microbatch_size < X.shape[0]:
                # Exact effective-batch gradient accumulation: each microbatch mean is weighted
                # by its share of the full batch, so the summed gradient equals the full-batch
                # mean mathematically.  This keeps the experiment's configured batch size (and
                # one optimizer/EMA update per batch) while bounding activation memory.
                opt.zero_grad()
                batch_n = int(X.shape[0])
                batch_loss = torch.zeros((), dtype=torch.float64, device=device)
                for start in range(0, batch_n, microbatch_size):
                    stop = min(start + microbatch_size, batch_n)
                    X_mb, y_mb = X[start:stop], y[start:stop]
                    weight = (stop - start) / batch_n
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        logits, all_logits = model(X_mb)
                        loss = _loss_fn(logits, y_mb, loss_type)
                        if all_logits is not None and deep_supervision_weight > 0:
                            ds_loss = sum(
                                _loss_fn(sl, y_mb, loss_type) for sl in all_logits
                            ) / len(all_logits)
                            loss = loss + deep_supervision_weight * ds_loss
                    scaler.scale(loss * weight).backward()
                    batch_loss.add_(loss.detach().to(torch.float64), alpha=weight)
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                async_epoch_loss.add_(batch_loss)
                loss_val = None
            else:
                opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits, all_logits = model(X)
                    loss = _loss_fn(logits, y, loss_type)
                    if all_logits is not None and deep_supervision_weight > 0:
                        ds_loss = sum(
                            _loss_fn(sl, y, loss_type) for sl in all_logits
                        ) / len(all_logits)
                        loss = loss + deep_supervision_weight * ds_loss
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                loss_val = loss.item()
            if loss_val is not None:
                epoch_loss += loss_val
            n_batches += 1
        if async_epoch_loss is not None:
            epoch_loss += async_epoch_loss.item()  # one synchronization per epoch, not per batch
        if use_amp and use_cuda_graph and not math.isfinite(epoch_loss):
            raise RuntimeError(
                f"train(): non-finite epoch loss at epoch {epoch} under amp+cuda_graph. The "
                f"static loss scale ({amp_static_loss_scale:g}) has overflowed fp16 gradients "
                "— lower train.amp_static_loss_scale (e.g. halve it) or run amp without "
                "cuda_graph to get dynamic loss scaling."
            )
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
    amp: bool = False,
    cuda_graph: bool = False,
    amp_static_loss_scale: float = 8192.0,
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

    ``amp`` / ``cuda_graph`` (opt-in, CUDA-only, both inert on CPU): same contracts as ``train``.
    ``amp`` autocasts each pass's forward/loss to fp16 (GradScaler; weights stay fp32 masters).
    ``cuda_graph`` captures the PASS as a pair of graphs (fresh-init + carried-state — see
    ``_CapturedPassStep``) and replays them; combined with ``amp`` the pass uses the static loss
    scale ``amp_static_loss_scale`` (non-finite epoch loss raises with guidance). The ragged
    last batch is dropped under ``cuda_graph`` (fixed replay shape, same as ``train``). With
    both flags off this routine is bit-identical to the pre-flag one (disabled autocast/scaler
    are documented no-ops; the loss bookkeeping accumulates in float64 on-device, which reproduces
    Python's float sum of float32 losses bit-for-bit while synchronizing once per epoch instead
    of once per pass).
    """
    if n_sup < 1:
        raise ValueError(f"n_sup must be >= 1, got {n_sup}")
    model = model.to(device)
    use_graph = cuda_graph and torch.device(device).type == "cuda"
    use_amp = amp and torch.device(device).type == "cuda"
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, capturable=use_graph
    )
    ema = EMA(model, ema_decay) if ema_decay is not None else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not use_graph)
    captured: _CapturedPassStep | None = None
    losses = []

    for epoch in range(epochs):
        model.train()
        # Device-side float64 loss accumulation: one host synchronization per EPOCH instead of
        # one `.item()` per PASS. The launch-bound small-model regime these routines run in was
        # paying a full CPU/GPU round-trip n_sup times per batch; float64 accumulation of the
        # exactly-converted float32 losses reproduces the former Python float sum bit-for-bit.
        async_epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
        n_passes = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            if use_graph:
                if captured is None:
                    captured = _CapturedPassStep(
                        model, opt, ema, X, y, n_steps=n_steps, carry=carry, n_sup=n_sup,
                        deep_supervision_weight=deep_supervision_weight,
                        amp=use_amp, amp_static_loss_scale=amp_static_loss_scale,
                    )
                if int(X.shape[0]) != captured.batch_size:
                    continue  # drop the ragged last batch — the graphs' shape is fixed
                loss = captured.replay_fresh(X, y)
                async_epoch_loss.add_(loss.detach().to(torch.float64))
                n_passes += 1
                for _ in range(n_sup - 1):
                    loss = (
                        captured.replay_carry() if carry else captured.replay_fresh_again()
                    )
                    async_epoch_loss.add_(loss.detach().to(torch.float64))
                    n_passes += 1
                continue
            state = None  # fresh (learned z0 / zero answer) at the start of each batch
            for _ in range(n_sup):
                opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits, all_logits, state = model(
                        X, n_steps=n_steps, init_state=state, return_state=True
                    )
                    loss = _loss_fn(logits, y)
                    if all_logits is not None and deep_supervision_weight > 0:
                        ds_loss = sum(_loss_fn(sl, y) for sl in all_logits) / len(all_logits)
                        loss = loss + deep_supervision_weight * ds_loss
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                # Detach the carried state so the next pass's gradient stops here — the bounded
                # horizon that lets effective depth grow with n_sup without long BPTT. With
                # carry=False, drop the state so the next pass restarts fresh (the compute-matched
                # control: same step count, no carry).
                # `.to(X.dtype)` is a no-op (returns self, bit-identical) without AMP; under AMP
                # it makes explicit the fp32 promotion `torch.cat([X, z, a])` would apply anyway,
                # so the eager carry matches the captured path's fixed-dtype static buffers.
                state = (
                    (state[0].detach().to(X.dtype), state[1].detach().to(X.dtype))
                    if carry
                    else None
                )
                async_epoch_loss.add_(loss.detach().to(torch.float64))
                n_passes += 1
        epoch_loss = async_epoch_loss.item()  # the single per-epoch synchronization
        if use_amp and use_graph and not math.isfinite(epoch_loss):
            raise RuntimeError(
                f"train_deep_supervision(): non-finite epoch loss at epoch {epoch} under "
                f"amp+cuda_graph. The static loss scale ({amp_static_loss_scale:g}) has "
                "overflowed fp16 gradients — lower train.amp_static_loss_scale (e.g. halve "
                "it) or run amp without cuda_graph to get dynamic loss scaling."
            )
        avg = epoch_loss / max(n_passes, 1)
        losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={avg:.4f}")

    if ema is not None:
        ema.copy_to(model)
    return losses


def train_act(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    max_segments: int,
    n_steps: int | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    deep_supervision_weight: float = 1.0,
    halt_weight: float = 0.5,
    ema_decay: float | None = None,
    device: str = "cpu",
    amp: bool = False,
    cuda_graph: bool = False,
    amp_static_loss_scale: float = 8192.0,
    verbose: bool = False,
) -> list[float]:
    """ACT / adaptive-computation deep supervision (M23 — the §4/§12 unbuilt TRM ingredient).

    Extends ``train_deep_supervision`` (detached-carry segmented deep supervision — the autopsy's
    active ingredient) with a learned **halting head**. Each batch runs ``max_segments`` supervised
    passes carrying ``(z, a)`` detached between them (effective depth = ``max_segments × n_steps``
    without long BPTT). Per segment, alongside the task loss, the model's ``halt_head`` reads the
    latent ``z`` and is trained (BCE) to predict whether the segment's answer is EXACTLY correct
    (per-example exact-match). It is the simplification of HRM's Q-halt to direct correctness
    prediction: the head learns "have I solved this row yet?", which at inference (``act_predict``)
    lets each example halt as soon as it is confidently solved and spend the remaining segments only
    on the rows that still need refinement — adaptive test-time compute, the TRM/HRM mechanism.

    Training always runs the full ``max_segments`` (every segment supervised); the adaptivity is a
    property of the learned head, exercised at eval. Requires a model with ``halt_head`` (TRM with
    ``use_act=True``) and the ``init_state``/``return_state`` API. Deterministic given seed.

    ``amp`` / ``cuda_graph`` (opt-in, CUDA-only, both inert on CPU): same contracts as ``train``
    and ``train_deep_supervision`` — the captured unit is one SEGMENT (task loss + per-step DS +
    the halting BCE, all capturable), replayed ``max_segments`` times per batch with the detached
    carry copied between replays. With both off, bit-identical to the pre-flag routine.
    """
    if max_segments < 1:
        raise ValueError(f"max_segments must be >= 1, got {max_segments}")
    if getattr(model, "halt_head", None) is None:
        raise ValueError("train_act requires a model built with use_act=True (a halt_head).")
    model = model.to(device)
    use_graph = cuda_graph and torch.device(device).type == "cuda"
    use_amp = amp and torch.device(device).type == "cuda"
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, capturable=use_graph
    )
    ema = EMA(model, ema_decay) if ema_decay is not None else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not use_graph)
    captured: _CapturedPassStep | None = None
    losses = []
    bce = nn.functional.binary_cross_entropy_with_logits

    for epoch in range(epochs):
        model.train()
        # One host synchronization per epoch rather than one per segment — see the same comment
        # in train_deep_supervision; float64 accumulation reproduces the Python float sum exactly.
        async_epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
        n_passes = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            if use_graph:
                if captured is None:
                    captured = _CapturedPassStep(
                        model, opt, ema, X, y, n_steps=n_steps, carry=True,
                        n_sup=max_segments,
                        deep_supervision_weight=deep_supervision_weight,
                        halt_weight=halt_weight,
                        amp=use_amp, amp_static_loss_scale=amp_static_loss_scale,
                    )
                if int(X.shape[0]) != captured.batch_size:
                    continue  # drop the ragged last batch — the graphs' shape is fixed
                loss = captured.replay_fresh(X, y)
                async_epoch_loss.add_(loss.detach().to(torch.float64))
                n_passes += 1
                for _ in range(max_segments - 1):
                    loss = captured.replay_carry()
                    async_epoch_loss.add_(loss.detach().to(torch.float64))
                    n_passes += 1
                continue
            state = None  # fresh (learned z0 / zero answer) at the start of each batch
            for _ in range(max_segments):
                opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits, all_logits, state = model(
                        X, n_steps=n_steps, init_state=state, return_state=True
                    )
                    z, a = state
                    loss = _loss_fn(logits, y)
                    if all_logits is not None and deep_supervision_weight > 0:
                        ds_loss = sum(_loss_fn(sl, y) for sl in all_logits) / len(all_logits)
                        loss = loss + deep_supervision_weight * ds_loss
                    # Halt target = is the current answer exactly correct (per example)? Detached:
                    # the halt head learns to *predict* correctness, it does not shape the answer
                    # logits.
                    with torch.no_grad():
                        correct = logits.argmax(dim=-1) == y
                        is_solved = correct.all(dim=-1).float() if y.ndim > 1 else correct.float()
                    halt_logit = model.halt_head(z).squeeze(-1)
                    loss = loss + halt_weight * bce(halt_logit, is_solved)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                # Bounded gradient horizon (carry, detached). `.to(X.dtype)` is a no-op without
                # AMP and restores fp32 under it — see train_deep_supervision's note.
                state = (z.detach().to(X.dtype), a.detach().to(X.dtype))
                async_epoch_loss.add_(loss.detach().to(torch.float64))
                n_passes += 1
        epoch_loss = async_epoch_loss.item()  # the single per-epoch synchronization
        if use_amp and use_graph and not math.isfinite(epoch_loss):
            raise RuntimeError(
                f"train_act(): non-finite epoch loss at epoch {epoch} under amp+cuda_graph. The "
                f"static loss scale ({amp_static_loss_scale:g}) has overflowed fp16 gradients — "
                "lower train.amp_static_loss_scale (e.g. halve it) or run amp without cuda_graph "
                "to get dynamic loss scaling."
            )
        avg = epoch_loss / max(n_passes, 1)
        losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={avg:.4f}")

    if ema is not None:
        ema.copy_to(model)
    return losses


def _stable_step_map(model: nn.Module, X: torch.Tensor):
    """The one-step latent map ``F(z) = update(cat[X, z, readout(z)])`` as a function of ``z`` only.

    Mirrors ``introspection._step_z_fn`` (the answer ``a`` is slaved to ``z`` via ``readout``, as it
    is past step 0 of the loop), but grad-ENABLED so its Jacobian penalty backpropagates to the
    weights. ``X`` is the fixed batch; the map is per-row independent, so a single random tangent
    gives an unbiased per-row Hutchinson estimate. Faithful for any ``n_latent`` because it goes
    through the model's own one-outer-step ``forward`` (which runs the inner z-updates)."""

    def F(z: torch.Tensor) -> torch.Tensor:
        a = model.readout(z)
        _, _, (z2, _) = model(X, n_steps=1, init_state=(z, a), return_state=True)
        return z2

    return F


def train_stable(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    jac_reg_weight: float = 0.0,
    fixed_point_weight: float = 0.0,
    n_reg_steps: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    deep_supervision_weight: float = 1.0,
    ema_decay: float | None = None,
    reg_seed: int = 0,
    device: str = "cpu",
    amp: bool = False,
    verbose: bool = False,
) -> list[float]:
    """Standard training + a contraction penalty on the loop's one-step latent map (M27).

    On top of the ordinary task (+ per-step DS) loss this adds, per batch:
      - a **Jacobian penalty** (DEQ; Bai 2021 arXiv 2106.14342): ``jac_reg_weight · mean‖J v‖²``
        with ``J = ∂F/∂z`` of the one-step map ``F`` at the (detached) ``n_reg_steps``-rolled state
        and ``v`` a fresh random unit-variance tangent — a Hutchinson estimate that drives the map's
        amplification (ρ / σ_max, the M21 ``jacobian_spectrum`` quantities) down toward contraction;
      - a **fixed-point residual penalty** (path independence; Anil 2022 arXiv 2211.09961):
        ``fixed_point_weight · mean(‖z_{t+1}−z_t‖ / ‖z_t‖)`` at the rolled state — the residual
        M21's ``latent_dynamics`` measures, driven toward 0.

    Requires the ``init_state``/``return_state`` API and a ``readout`` (``trm`` flat-z or
    ``trm_mixer`` per-cell-z; ``_stable_step_map`` is shape-agnostic). With both
    weights 0 the penalties are skipped entirely (but the routine is a distinct code path from
    ``train`` and is NOT asserted bit-identical to it — the runner only dispatches here when a
    weight is > 0). Determinism caveat: the ``jvp`` involves float-reduction-order-sensitive ops, so
    like
    ``trm_decoupled`` / the M21 diagnostics this reproduces bit-for-bit only with CPU threads pinned
    (``num_threads=1``, the committed default). ``reg_seed`` seeds the tangent sampler.

    ``amp`` (opt-in, CUDA-only, inert on CPU) autocasts the TASK (+ per-step DS) loss to fp16
    exactly as ``train`` does, but deliberately keeps the **contraction penalties in fp32**: the
    Jacobian-vector probe measures an amplification factor whose whole point is small-magnitude
    fidelity, and fp16 would degrade the very quantity being regularized. This is not the kind of
    per-arm precision difference the "amp must be uniform" contract guards against — the penalty
    term exists only on contraction arms in the first place, so it cannot shift a between-arm Δ
    that no other arm shares. ``cuda_graph`` is NOT supported here (the ``torch.func.jvp`` probe
    builds a fresh graph structure per batch); the runner rejects that combination loudly.
    ``amp=False`` is bit-identical to the pre-flag routine.
    """
    model = model.to(device)
    use_amp = amp and torch.device(device).type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = EMA(model, ema_decay) if ema_decay is not None else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    gen = torch.Generator(device=device).manual_seed(reg_seed)
    losses = []

    for epoch in range(epochs):
        model.train()
        # One host synchronization per epoch (see train_deep_supervision); float64 accumulation
        # of the float32 batch losses reproduces the former Python float sum bit-for-bit.
        async_epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, all_logits = model(X)
                loss = _loss_fn(logits, y)
                if all_logits is not None and deep_supervision_weight > 0:
                    ds_loss = sum(_loss_fn(sl, y) for sl in all_logits) / len(all_logits)
                    loss = loss + deep_supervision_weight * ds_loss

            if jac_reg_weight > 0 or fixed_point_weight > 0:
                # Detached linearization / residual point: the n_reg_steps-rolled latent. We
                # penalize the operator's LOCAL amplification at that point (point held constant,
                # à la DEQ), so the gradient flows through the penalty to the weights, not state.
                with torch.no_grad():
                    _, _, (z_pt, a_pt) = model(X, n_steps=n_reg_steps, return_state=True)
                z_pt = z_pt.detach()
                if jac_reg_weight > 0:
                    F = _stable_step_map(model, X)
                    v = torch.randn(z_pt.shape, generator=gen, device=device, dtype=z_pt.dtype)
                    _, Jv = torch.func.jvp(F, (z_pt,), (v,))
                    loss = loss + jac_reg_weight * Jv.pow(2).sum(dim=-1).mean()
                if fixed_point_weight > 0:
                    _, _, (z2, _) = model(
                        X, n_steps=1, init_state=(z_pt, a_pt.detach()), return_state=True
                    )
                    resid = (z2 - z_pt).norm(dim=-1) / z_pt.norm(dim=-1).clamp_min(1e-12)
                    loss = loss + fixed_point_weight * resid.mean()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            async_epoch_loss.add_(loss.detach().to(torch.float64))
            n_batches += 1
        epoch_loss = async_epoch_loss.item()  # the single per-epoch synchronization
        avg = epoch_loss / max(n_batches, 1)
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

    No ``amp``/``cuda_graph`` here by design: the per-batch depth ``T`` is resampled every batch,
    so both the unroll length and the number of step-aligned readouts change batch to batch —
    there is no fixed shape for a graph to capture, and the routine serves closed levers (M3b /
    M7) rather than the launch-bound regime those knobs target. The runner rejects the
    combination loudly rather than silently dropping the flag. The loss bookkeeping still
    accumulates on-device (one host synchronization per epoch instead of per batch), which is
    bit-identical to the former Python float sum.
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator()  # CPU generator drives integer depth sampling deterministically
    gen.manual_seed(seed)
    losses = []

    for epoch in range(epochs):
        model.train()
        async_epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
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
            async_epoch_loss.add_(loss.detach().to(torch.float64))
            n_batches += 1
        avg = async_epoch_loss.item() / max(n_batches, 1)
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

    Like ``train_curriculum``, no ``amp``/``cuda_graph``: depth ``T`` and gradient budget ``k``
    are resampled per batch, so nothing about the step has a fixed capturable shape. The
    per-epoch (rather than per-batch) loss synchronization applies here too, bit-identically.
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
        async_epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
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
            async_epoch_loss.add_(loss.detach().to(torch.float64))
            n_batches += 1
        avg = async_epoch_loss.item() / max(n_batches, 1)
        losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={avg:.4f}")

    return losses
