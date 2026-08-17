"""M35 follow-up, take 3: takes 1-2 used flat `TRM`, but electricity's real h1110 arms
(`trm_mixer`/`trm_mixer_nomix`) are `TRMMixer`, an architecturally different (per-cell,
token-mixing) class -- a flat dense in_features(30816)->hidden(1110) layer is never actually
built in the real recipe (that's exactly why the committed config keeps flat `trm`/`ff_matched`
at hidden_dim=64, not 1110). This is the faithful proxy: the real TRMMixer class, at the real
electricity in_features/out_features/hidden_dim/token_hidden, with n_sup=4 DS-family capture
bolted on (the real recipe doesn't use n_sup -- this asks "what if it did").
"""
import time

import torch

from looptab.models.mixer import TRMMixer
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
    m = TRMMixer(in_features=in_features, num_classes=2, out_features=out_features,
                 hidden_dim=hidden, latent_dim=hidden, token_hidden=8, n_steps=8, n_latent=1,
                 use_rmsnorm=True, deep_supervision=False)
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
