"""Exact-forward-shape stress test for hypothetical DS capture on electricity's widest arm.

This is deliberately NOT a real experiment: the runner prohibits ``n_sup>1`` on regression and
CUDA graphs cannot compose with the real arm's effective-batch microbatch accumulation. It does,
however, match the forward/loss shape that the earlier M35 proxy got wrong: horizon/readout width
24, MSE targets, per-step deep-supervision readouts, and the real microbatch forward size 64.
Results from the older ``num_classes=2``/CE/batch-128 proxy must not be used to close a real-
workload hypothesis; rerun this only as a clearly labelled memory/throughput stress test.
"""
import time

import torch

from looptab.models.mixer import TRMMixer
from looptab.train.loop import train_deep_supervision

torch.manual_seed(0)
n, in_features, out_features, horizon, hidden = 128, 30816, 321, 24, 1110
X = torch.randn(n, in_features)
y = torch.randn(n, out_features, horizon)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, y), batch_size=64, shuffle=True
)

for label, use_graph in [("eager", False), ("cuda_graph", True)]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    m = TRMMixer(in_features=in_features, num_classes=horizon, out_features=out_features,
                 hidden_dim=hidden, latent_dim=hidden, token_hidden=8, n_steps=8, n_latent=1,
                 use_rmsnorm=True, deep_supervision=True)
    t0 = time.perf_counter()
    try:
        losses = train_deep_supervision(
            m, loader, n_sup=4, carry=True, epochs=3, lr=1e-3, device="cuda",
            loss_type="mse", cuda_graph=use_graph,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        n_passes = len(loader) * 3 * 4  # epochs * n_sup passes
        print(f"{label}: OK  total={elapsed:.2f}s  per-pass={elapsed/n_passes*1000:.2f}ms  "
              f"peak_alloc={peak_gb:.2f}GB  peak_reserved={peak_reserved_gb:.2f}GB  "
              f"loss(finite)={all(v == v for v in losses)}")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        elapsed = time.perf_counter() - t0
        print(f"{label}: FAILED after {elapsed:.2f}s -- {type(e).__name__}: {e}")
    del m
