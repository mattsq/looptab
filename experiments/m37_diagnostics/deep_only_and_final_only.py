"""(C) is the M37 collapse caused by T_min? (D) does the hard bottleneck rescue FINAL-ONLY training?"""
import sys

import torch

from looptab.data.dataset import make_loaders, make_trajectory_dataset
from looptab.models.mixer import TRMMixer
from looptab.train.loop import train_curriculum

torch.set_num_threads(1)
dev = "cuda"
cfg = {"w": 24, "rule": 110, "distractors": 0}
ds = make_trajectory_dataset(task_cfg=cfg, task_seed=42, sample_seed=1, n=4000, T_max=8)
test = make_trajectory_dataset(task_cfg=cfg, task_seed=42, sample_seed=2, n=1000, T_max=8)
loader, _ = make_loaders(ds, test, batch_size=256, device=dev)
X, traj = torch.as_tensor(test.X).to(dev), torch.as_tensor(test.traj).to(dev)

# (label, T_min, T_max, stride); stride 8 with T<=8 = final state only
cases = {
    "C1": ("stride1 T4..8", 4, 8, 1),
    "C2": ("stride1 T=4", 4, 4, 1),
    "D1": ("FINAL-only T1..8", 1, 8, 8),
    "D2": ("FINAL-only T2..8", 2, 8, 8),
}
for key in sys.argv[1:]:
    label, tmin, tmax, stride = cases[key]
    for seed in (0, 1):
        torch.manual_seed(seed)
        m = TRMMixer(in_features=24, num_classes=2, hidden_dim=96, latent_dim=64, n_steps=8,
                     out_features=24, use_rmsnorm=True, token_hidden=32, state_bottleneck="hard")
        losses = train_curriculum(m, loader, T_min=tmin, T_max=tmax, ds_mode="step_aligned",
                                  ds_stride=stride, epochs=100, device=dev, seed=seed)
        m.eval()
        with torch.no_grad():
            _, logits = m(X, n_steps=8)
        acc = [round((lg.argmax(-1) == traj[:, i]).float().mean().item(), 3)
               for i, lg in enumerate(logits)]
        print(key, label, "seed", seed, "loss", [round(x, 3) for x in losses[::25]], "acc s1..s8",
              acc, flush=True)
