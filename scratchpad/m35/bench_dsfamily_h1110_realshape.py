"""M35 follow-up, take 2: the first proxy (converge, w=24+8 distractors -> in_features=32) was too
small to be a faithful #18 replica -- electricity's REAL memory pressure comes from the huge
M*lookback input layer (321*96=30816 in_features), not hidden_dim alone. Random data at
electricity's REAL shape (in_features=30816, out_features=321 cells, hidden=1110) isolates
whether DS-family's two-graph capture is dangerous at the shape that actually matters, without
needing real electricity data (labels are random -- this is a memory/speed probe, not a training
result).
"""
import time

import torch

from looptab.models.trm import TRM
from looptab.train.loop import train_deep_supervision

torch.manual_seed(0)
n, in_features, out_features, hidden = 512, 30816, 321, 1110
X = torch.randn(n, in_features)
y = torch.randint(0, 2, (n, out_features))
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, y), batch_size=128, shuffle=True
)

for label, use_graph in [("eager", False), ("cuda_graph", True)]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    m = TRM(in_features=in_features, num_classes=2, out_features=out_features,
            hidden_dim=hidden, latent_dim=hidden, n_steps=8, deep_supervision=False)
    t0 = time.perf_counter()
    try:
        losses = train_deep_supervision(
            m, loader, n_sup=4, carry=True, epochs=3, lr=1e-3, device="cuda",
            cuda_graph=use_graph,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        n_passes = len(loader) * 3 * 4  # epochs * n_sup passes
        print(f"{label}: OK  total={elapsed:.2f}s  per-pass={elapsed/n_passes*1000:.2f}ms  "
              f"peak_mem={peak_gb:.2f}GB  loss(finite)={all(v == v for v in losses)}")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        elapsed = time.perf_counter() - t0
        print(f"{label}: FAILED after {elapsed:.2f}s -- {type(e).__name__}: {e}")
    del m
