"""Rest-state L2 anchor.

A single fixed-weight L2 term pulling each Gaussian's deformed position
toward its canonical rest position (the static input image's 3DGS). No
schedule, no decay, no floor -- the elaborate version was descoped per
2026-05-11 advisor feedback ("energy may be too ambitious").
"""

from __future__ import annotations

import torch
from torch import Tensor


def rest_state_l2(
    deformed_positions: Tensor,
    rest_positions: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Weighted mean of squared displacement from the canonical rest state.

    Args:
        deformed_positions: (N, D) positions predicted by the deformation MLP.
        rest_positions: (N, D) canonical positions from the static 3DGS.
        weights: optional (N,) per-point weights. Useful for downweighting
            static-labelled Gaussians.

    Returns:
        Scalar tensor. Multiply by eta (small constant, e.g. 1e-3) externally.
    """
    if deformed_positions.shape != rest_positions.shape:
        raise ValueError(
            f"shape mismatch: deformed {tuple(deformed_positions.shape)} "
            f"vs rest {tuple(rest_positions.shape)}"
        )
    sq = (deformed_positions - rest_positions).pow(2).sum(dim=-1)  # (N,)
    if weights is None:
        return sq.mean()
    if weights.shape != sq.shape:
        raise ValueError(
            f"weights shape {tuple(weights.shape)} must match {tuple(sq.shape)}"
        )
    denom = weights.sum().clamp(min=1e-12)
    return (sq * weights).sum() / denom
