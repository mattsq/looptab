"""Isolated M34 training-step benchmark.

Run one mode per process so CUDA allocator state and graph-private pools do not leak between
measurements.  This is deliberately a scratchpad tool, not experiment substrate.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from looptab.models.mixer import TRMMixer
from looptab.train.loop import _CapturedStep, _loss_fn  # noqa: PLC2701

SHAPES = {
    "etth": dict(batch=128, cells=7, cell_dim=96, classes=24, hidden=224, latent=96,
                 token_hidden=8, objective="mse"),
    "converge96": dict(batch=256, cells=96, cell_dim=1, classes=2, hidden=142, latent=142,
                       token_hidden=48, objective="ce"),
    "sudoku9": dict(batch=256, cells=81, cell_dim=1, classes=9, hidden=370, latent=370,
                    token_hidden=64, objective="ce"),
    "electricity": dict(batch=128, cells=321, cell_dim=96, classes=24, hidden=1110,
                        latent=1110, token_hidden=8, objective="mse"),
    "traffic": dict(batch=128, cells=862, cell_dim=96, classes=24, hidden=1760,
                    latent=1760, token_hidden=8, objective="mse"),
}


def build(
    name: str,
    batch_override: int | None,
    hidden_override: int | None,
    latent_override: int | None,
):
    cfg = SHAPES[name].copy()
    if batch_override is not None:
        cfg["batch"] = batch_override
    if hidden_override is not None:
        cfg["hidden"] = hidden_override
    if latent_override is not None:
        cfg["latent"] = latent_override
    torch.manual_seed(0)
    model = TRMMixer(
        in_features=cfg["cells"] * cfg["cell_dim"],
        out_features=cfg["cells"],
        num_classes=cfg["classes"],
        hidden_dim=cfg["hidden"],
        latent_dim=cfg["latent"],
        token_hidden=cfg["token_hidden"],
        n_steps=8,
        use_rmsnorm=True,
        deep_supervision=True,
    ).cuda()
    X = torch.randn(cfg["batch"], cfg["cells"] * cfg["cell_dim"], device="cuda")
    if cfg["objective"] == "mse":
        y = torch.randn(cfg["batch"], cfg["cells"], cfg["classes"], device="cuda")
    else:
        y = torch.randint(cfg["classes"], (cfg["batch"], cfg["cells"]), device="cuda")
    return cfg, model, X, y


def loss_for(model, X, y, objective):
    logits, all_logits = model(X)
    loss = _loss_fn(logits, y, objective)
    loss = loss + sum(_loss_fn(sl, y, objective) for sl in all_logits) / len(all_logits)
    return loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("shape", choices=SHAPES)
    p.add_argument("mode", choices=("eager", "amp", "graph", "amp_graph", "eval"))
    p.add_argument("--batch", type=int)
    p.add_argument("--hidden", type=int)
    p.add_argument("--latent", type=int)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()

    cfg, model, X, y = build(args.shape, args.batch, args.hidden, args.latent)
    objective = cfg["objective"]
    use_amp = args.mode in ("amp", "amp_graph")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, capturable=args.mode == "graph")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    graph = None
    captured = None

    def eager_step():
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_for(model, X, y, objective)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    if args.mode == "eval":
        model.eval()

        def step():
            with torch.no_grad():
                return model(X)[0].square().mean()

    elif args.mode == "graph":
        captured = _CapturedStep(model, opt, None, X, y, 1.0, objective,
                                 warmup_iters=args.warmup)

        def step():
            return captured.run(X, y)

    elif args.mode == "amp_graph":
        # PyTorch's documented AMP+CUDAGraph recipe: graph autocast forward/loss/backward;
        # leave GradScaler.step/update eager because they contain a host synchronization.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(args.warmup):
                eager_step()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        opt.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.autocast("cuda", dtype=torch.float16):
                static_loss = loss_for(model, X, y, objective)
            scaler.scale(static_loss).backward()

        def step():
            graph.replay()
            scaler.step(opt)
            scaler.update()
            return static_loss

    else:
        step = eager_step

    if args.mode not in ("graph", "amp_graph"):
        for _ in range(args.warmup):
            step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    losses = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = step()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        losses.append(float(loss.detach()))
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"shape={args.shape} mode={args.mode} batch={cfg['batch']} "
        f"median_ms={statistics.median(times):.2f} mean_ms={statistics.mean(times):.2f} "
        f"peak_GiB={peak:.3f} loss={losses[-1]:.6g} wall={sum(times)/1000:.2f}s"
    )


if __name__ == "__main__":
    main()
