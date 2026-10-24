"""M37 diagnostic: (A) can the bottlenecked step learn ONE CA step and iterate it? (B) does a
shallow step-supervised curriculum (T in 1..4) train it? Seed 0, rule 110, w=24, GPU."""
import torch

from looptab.data.dataset import make_loaders, make_trajectory_dataset
from looptab.models.mixer import TRMMixer
from looptab.train.loop import train_curriculum

torch.set_num_threads(1)
dev = "cuda"
ds = make_trajectory_dataset(task_cfg={"w": 24, "rule": 110, "distractors": 0}, task_seed=42,
                             sample_seed=1, n=4000, T_max=8)
test = make_trajectory_dataset(task_cfg={"w": 24, "rule": 110, "distractors": 0}, task_seed=42,
                               sample_seed=2, n=1000, T_max=8)
loader, test_loader = make_loaders(ds, test, batch_size=256, device=dev)


def step_acc(m):
    m.eval()
    X, traj = torch.as_tensor(test.X).to(dev), torch.as_tensor(test.traj).to(dev)
    with torch.no_grad():
        _, logits = m(X, n_steps=8)
    return [round((lg.argmax(-1) == traj[:, i]).float().mean().item(), 3)
            for i, lg in enumerate(logits)]


def build(sb):
    torch.manual_seed(0)
    return TRMMixer(in_features=24, num_classes=2, hidden_dim=96, latent_dim=64, n_steps=8,
                    out_features=24, use_rmsnorm=True, token_hidden=32, state_bottleneck=sb)


for sb in ["none", "hard"]:
    m = build(sb)
    losses = train_curriculum(m, loader, T_min=1, T_max=1, ds_mode="step_aligned", epochs=60,
                              device=dev, seed=0)
    print("(A) T=1 only", sb, "loss %.4f" % losses[-1], "rollout acc s1..s8", step_acc(m))

for sb in ["soft", "hard"]:
    m = build(sb)
    losses = train_curriculum(m, loader, T_min=1, T_max=4, ds_mode="step_aligned", epochs=100,
                              device=dev, seed=0)
    print("(B) T=1..4 step DS, 100ep", sb, "loss", [round(x, 3) for x in losses[::20]],
          "acc s1..s8", step_acc(m))
