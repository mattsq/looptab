"""Mode-level metrics for the multimodal `ambiguous_converge` task (M34).

On a genuinely multimodal target, ordinary accuracy/EM cannot separate the two hypotheses this
milestone tests. If the label is drawn uniformly from a row's mode set, then EVERY predictor that
outputs a valid mode — a perfect mode-sampler and a deterministic pick-one-mode model alike —
scores the same expected exact-match, ``E[1/n_modes]``. EM is therefore a floor check here, not the
discriminator. What separates them is *what kind of object* the model emits:

  - ``mode_validity`` — is the prediction a member of the row's enumerated mode set at all? A
    conditional-mean predictor is not: minimizing per-cell CE across modes yields the cell-wise
    marginal argmax, which on this task is a valid mode in only ~35% of rows (screened at
    rule 78, w=24). This is the *blur vs commit* metric, and it is the honest headline.
  - ``mode_coverage`` — over S latent draws, the fraction of the row's modes the model actually
    produces. This is Explorative Modeling's central claim ("the number of modes a model can
    capture grows with exploration") stated as a directly measurable quantity, which is possible
    here only because the mode set is enumerable (see ``generators.ambiguous_modes``).

Every arm is scored with the same S draws; a deterministic arm simply repeats its single answer, so
its coverage is ``mode_validity / n_modes`` by construction. That keeps the comparison fair rather
than giving the stochastic arm S chances and the control one.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.generators import ambiguous_modes  # noqa: F401  (re-exported for callers)


def mode_metrics(preds: np.ndarray, modes: list[np.ndarray]) -> dict:
    """Mode-level metrics from ``(S, N, w)`` sampled predictions and per-row mode sets.

    ``preds[s, i]`` is draw ``s`` for row ``i``; ``modes[i]`` is that row's ``(n_modes, w)``
    enumerated support. Returns validity (any-draw and mean-draw), coverage, and the descriptive
    mean mode count.
    """
    S, N = preds.shape[0], preds.shape[1]
    if len(modes) != N:
        raise ValueError(f"mode_metrics: {len(modes)} mode sets for {N} rows")
    valid_first, valid_mean, coverage, counts = [], [], [], []
    for i in range(N):
        mset = {tuple(m) for m in modes[i]}
        drawn = [tuple(preds[s, i]) for s in range(S)]
        hits = [d in mset for d in drawn]
        valid_first.append(hits[0])                       # one honest draw
        valid_mean.append(float(np.mean(hits)))           # averaged over draws
        coverage.append(len({d for d, h in zip(drawn, hits) if h}) / max(len(mset), 1))
        counts.append(len(mset))
    return {
        "mode_validity": float(np.mean(valid_mean)),
        "mode_validity_first": float(np.mean(valid_first)),
        "mode_coverage": float(np.mean(coverage)),
        "mode_count": float(np.mean(counts)),
    }


@torch.inference_mode()
def sampled_or_repeated_preds(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    n_samples: int,
    noise_dim: int,
    noise_std: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
) -> np.ndarray:
    """``(S, N, w)`` predictions: S independent latent draws, or the single deterministic answer
    repeated S times for an arm with no latent (``noise_dim == 0``). Same S for every arm — the
    coverage comparison is only fair if the draw budget is held fixed."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    per_sample = [[] for _ in range(n_samples)]
    for X, _ in loader:
        X = X.to(device)
        if noise_dim == 0:
            out, _ = model(X)
            p = out.argmax(dim=-1).cpu().numpy()
            for s in range(n_samples):
                per_sample[s].append(p)
            continue
        for s in range(n_samples):
            eps = torch.randn(X.shape[0], noise_dim, generator=gen, device=device) * noise_std
            model.set_noise(eps)
            out, _ = model(X)
            model.clear_noise()
            per_sample[s].append(out.argmax(dim=-1).cpu().numpy())
    return np.stack([np.concatenate(p) for p in per_sample])


def evaluate_modes(
    model: torch.nn.Module,
    loader: DataLoader,
    modes: list[np.ndarray],
    *,
    n_samples: int,
    noise_dim: int,
    noise_std: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Mode metrics for one trained arm on the pre-enumerated ``modes`` of the eval set."""
    preds = sampled_or_repeated_preds(
        model,
        loader,
        n_samples=n_samples,
        noise_dim=noise_dim,
        noise_std=noise_std,
        device=device,
        seed=seed,
    )
    return mode_metrics(preds, modes)
