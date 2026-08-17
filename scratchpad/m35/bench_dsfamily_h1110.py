"""M35 follow-up: does DS-family cuda_graph capture (fresh+carry, two graphs on one pool) hurt
worse than the already-measured standard-path single-graph capture at electricity's widest arm's
hidden_dim (1110)? Same shape #18 profiled (steady-state per-step time + peak memory), since a
short CLI run is dominated by warmup and shows nothing (see dsfamily_h1110_probe*.log).
"""
import time

import torch

from looptab.data.generators import make_converge
from looptab.models.trm import TRM
from looptab.train.loop import train_deep_supervision

X, y = make_converge(n=2048, w=24, rule=78, task_seed=42, sample_seed=1, distractors=8)
ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

for label, use_graph in [("eager", False), ("cuda_graph", True)]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    m = TRM(in_features=32, num_classes=2, out_features=24, hidden_dim=1110, latent_dim=1110,
            n_steps=6, deep_supervision=False)
    t0 = time.perf_counter()
    try:
        losses = train_deep_supervision(
            m, loader, n_sup=4, carry=True, epochs=8, lr=1e-3, device="cuda",
            cuda_graph=use_graph,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        n_batches = len(loader) * 8 * 4  # epochs * n_sup passes
        print(f"{label}: OK  total={elapsed:.2f}s  per-pass={elapsed/n_batches*1000:.2f}ms  "
              f"peak_mem={peak_gb:.2f}GB  final_loss={losses[-1]:.4f}")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        elapsed = time.perf_counter() - t0
        print(f"{label}: FAILED after {elapsed:.2f}s -- {type(e).__name__}: {e}")
    del m
