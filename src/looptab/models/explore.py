"""Latent-noise wrapper — the substrate for Explorative Modeling (XM, M34).

Explorative Modeling (Gladstone/Ji/Du 2026, arXiv 2607.27372) attacks **multimodality in the
target**: when many ``y`` are consistent with one ``x``, an ERM regression/CE target is a
conditional *mean*, so predictions blur across modes. XM's fix is to factor the TRAINING loop
rather than the generation procedure — draw K latents, produce K candidate outputs, score each
against the datapoint, and backpropagate only through the closest. Each candidate can commit to a
different mode, so the number of modes a model can represent grows with the exploration budget K.

That is directly the rival explanation of this repo's loop: TRM refinement is a *multi-step*
factorization used to handle hard targets, and XM claims the objective can do that job in one
step. Testing it here needs exactly one thing the repo's models lack — a **stochastic latent the
output can depend on**. This wrapper supplies it in a model-agnostic way: it appends an
``explore_noise_dim``-wide noise vector to the input, so EVERY arm (``ff_matched``, flat ``trm``,
``trm_mixer``, the untied stacks) becomes a conditional sampler ``y = f(X, ε)`` with no change to
any model class. The training routine sets the noise per candidate; eval leaves it at its default.

Design notes:
  - **Default noise = zeros.** The deterministic single-pass prediction (``ε=0``, the latent's
    mean) is what the committed ``evaluate`` path reports, so the headline metric stays a single
    forward pass comparable to every previous milestone and can never become an oracle. The
    *sampled* and best-of-K numbers live in the ``evaluate_explore`` side-car, labelled.
  - **The K=1 control must also be wrapped.** ``Δ(K=8 − K=1)`` at an identical architecture and an
    identical noise channel is the single-knob exploration ablation (§5.6); an unwrapped arm would
    confound exploration with *having* a stochastic latent. Configs therefore carry both.
  - Attribute access falls through to the wrapped model, so ``readout`` / ``halt_head`` /
    ``n_steps`` and the ``init_state``/``return_state``/``n_steps`` forward API keep working.
"""

from typing import Optional

import torch
import torch.nn as nn


class ExploreWrapper(nn.Module):
    """Wrap a model so its input carries a latent noise channel of width ``noise_dim``.

    The wrapped model must be constructed with ``in_features = d + noise_dim``. ``set_noise``
    installs the ε used by the next forward(s); ``clear_noise`` restores the zero default. The
    wrapper holds no parameters of its own, so ``count_params`` (and the budget audit) reads the
    wrapped model exactly.
    """

    def __init__(self, model: nn.Module, noise_dim: int):
        super().__init__()
        if noise_dim < 1:
            raise ValueError(f"noise_dim must be >= 1 to wrap a model, got {noise_dim}")
        self.model = model
        self.noise_dim = noise_dim
        self._noise: Optional[torch.Tensor] = None

    def set_noise(self, noise: torch.Tensor) -> None:
        """Install the (B, noise_dim) latent used by subsequent forwards."""
        if noise.ndim != 2 or noise.shape[-1] != self.noise_dim:
            raise ValueError(
                f"noise must be (B, {self.noise_dim}), got {tuple(noise.shape)}"
            )
        self._noise = noise

    def clear_noise(self) -> None:
        """Return to the zero latent (the deterministic single-pass prediction)."""
        self._noise = None

    def forward(self, X: torch.Tensor, *args, **kwargs):
        eps = self._noise
        if eps is None:
            eps = X.new_zeros(X.shape[0], self.noise_dim)
        elif eps.shape[0] != X.shape[0]:
            raise ValueError(
                f"installed noise batch {eps.shape[0]} != input batch {X.shape[0]}"
            )
        return self.model(torch.cat([X, eps.to(X.dtype)], dim=-1), *args, **kwargs)

    def count_params(self) -> int:
        return self.model.count_params()

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ resolves parameters/buffers/submodules; anything else (readout,
        # halt_head, n_steps, out_features, ...) falls through to the wrapped model so the wrapper
        # is transparent to the eval/introspection machinery.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["model"], name)


def sample_noise(
    batch: int, noise_dim: int, std: float, generator: torch.Generator, device: str = "cpu"
) -> torch.Tensor:
    """i.i.d. N(0, std²) latent for one batch, drawn from an explicitly seeded generator.

    The generator is owned by the training routine (seeded from the run seed), so the whole
    exploration stream is a pure function of the config + seed — §5.3 determinism, unchanged.
    """
    return torch.randn(batch, noise_dim, generator=generator, device=device) * std
