"""M37c control: is the bottlenecked mixer's rollout any better than rolling out a plain MLP?

M37c trains every arm on one-step pairs (s0 -> s1) only; at depth 1 the hard/soft/default mixers
are the same computation, so they train into identical weights and differ only in how the step is
iterated at test time. Feeding the decoded state back extrapolates to T=32. Any accurate one-step
model can be rolled out that way, so the fair control is `ff_matched` (same labels, same budget)
iterated autoregressively: x_{t+1} = argmax(ff(x_t)). Same data seeds and training recipe as the
runner (rule 110, w=24, n_train 4000, n_test 1000, 100 epochs, AdamW 1e-3, wd 1e-4, batch 256).
Writes a run record (config, per-seed metrics, git SHA) to results/.
"""

import json
import subprocess
import time

import numpy as np
import torch

from looptab.data.dataset import make_loaders, make_trajectory_dataset
from looptab.registry import get_model
from looptab.train.loop import train_curriculum

T_EVAL = [1, 4, 8, 16, 24, 32]
SEEDS = list(range(8))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TASK = {"w": 24, "rule": 110, "distractors": 0}


def git_sha():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def main():
    torch.set_num_threads(1)
    per_seed = {T: {"accuracy": [], "exact_match": []} for T in T_EVAL}
    for seed in SEEDS:
        # Same seed offsets as run.py (task_seed + seed; sample seeds + 100*seed).
        train = make_trajectory_dataset(task_cfg=TASK, task_seed=42 + seed,
                                        sample_seed=1 + 100 * seed, n=4000, T_max=1)
        test = make_trajectory_dataset(task_cfg=TASK, task_seed=42 + seed,
                                       sample_seed=2 + 100 * seed, n=1000, T_max=max(T_EVAL))
        loader, _ = make_loaders(train, train, batch_size=256, device=DEV)
        torch.manual_seed(seed)
        ff = get_model("ff_matched", in_features=24, num_classes=2, hidden_dim=64, latent_dim=64,
                       n_steps=1, out_features=24)
        train_curriculum(ff, loader, T_min=1, T_max=1, ds_mode="final", epochs=100,
                         device=DEV, seed=seed)
        ff.eval()
        X = torch.as_tensor(test.X).to(DEV)
        traj = torch.as_tensor(test.traj).to(DEV)
        with torch.no_grad():
            s = X
            for t in range(1, max(T_EVAL) + 1):
                logits, _ = ff(s)
                pred = logits.argmax(-1)
                s = pred.to(X.dtype)
                if t in T_EVAL:
                    correct = pred == traj[:, t - 1]
                    per_seed[t]["accuracy"].append(correct.float().mean().item())
                    per_seed[t]["exact_match"].append(correct.all(-1).float().mean().item())
        print("seed", seed, {t: round(per_seed[t]["accuracy"][-1], 3) for t in T_EVAL}, flush=True)

    summary = {T: {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}
                   for m, v in per_seed[T].items()} for T in T_EVAL}
    for T in T_EVAL:
        print(f"T={T:>2} acc {summary[T]['accuracy']['mean']:.3f} "
              f"EM {summary[T]['exact_match']['mean']:.3f}")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    record = {
        "config": {"task": "iterated", "params": TASK, "n_train": 4000, "n_test": 1000,
                   "train": "one-step pairs (T=1), train_curriculum final, 100 epochs, AdamW 1e-3,"
                            " wd 1e-4, batch 256", "model": "ff_matched hidden 64 latent 64",
                   "eval": "autoregressive rollout x_{t+1} = argmax(ff(x_t))", "device": DEV},
        "seeds": SEEDS, "git_sha": git_sha(), "timestamp": stamp, "metrics": summary,
    }
    path = f"results/m37c_ff_rollout_control_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print("record:", path)


if __name__ == "__main__":
    main()
