"""Physically-gated supervision weights.

For each frame t with precomputed ARAP-prior energy E_t, the per-frame
photometric loss is multiplied by

    w_t = exp(-alpha(t) * E_t)

where alpha(t) is adaptive -- normalized by an EMA of the scene's global
ARAP energy so the gating is scene-invariant.

E_t is computed offline from the video prior's intrinsic geometry (optical
flow lifted to sparse 3D control points via the static-3DGS depth map). It
does NOT depend on the deformation MLP's output, so the gating signal is
independent of training state.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_gating_weights(energies: Tensor, alpha: float) -> Tensor:
    """Per-frame gating weights.

    Args:
        energies: nonnegative tensor of shape (T,) -- per-frame ARAP-prior energy.
        alpha: nonnegative scalar -- gating strength (typically from AdaptiveAlpha).

    Returns:
        Tensor of shape (T,), values in (0, 1].
    """
    if alpha < 0:
        raise ValueError(f"alpha must be nonnegative, got {alpha}")
    if torch.any(energies < 0):
        raise ValueError("energies must be nonnegative")
    return torch.exp(-alpha * energies)


class AdaptiveAlpha:
    """alpha(t) = alpha0 / EMA_t(E_prior).

    EMA on a running scalar (the mean ARAP-prior energy). Call once per
    training step with the current step's mean energy; it returns the
    current alpha to use for gating.
    """

    def __init__(self, alpha0: float, momentum: float = 0.99, eps: float = 1e-6) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if alpha0 < 0:
            raise ValueError(f"alpha0 must be nonnegative, got {alpha0}")
        self.alpha0 = float(alpha0)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self._ema: float | None = None

    def __call__(self, energy_mean: Tensor) -> Tensor:
        e = float(energy_mean.detach().cpu().item())
        if self._ema is None:
            self._ema = e
        else:
            self._ema = self.momentum * self._ema + (1.0 - self.momentum) * e
        return torch.tensor(self.alpha0 / max(self._ema, self.eps))

    @property
    def ema(self) -> float | None:
        return self._ema
