"""Articulation-aware per-edge weights for SC-GS's ARAP regularizer.

SC-GS applies a uniform lambda to every K-NN edge between control points. That
penalty over-smooths joints -- at a pendulum hinge or an elbow, neighboring
points belong to different rigid parts and should be allowed to rotate
independently. We replace the uniform lambda with a per-edge weight:

    lambda_intra  for edges where both endpoints share a part label
    lambda_inter  for edges that cross a part boundary  (much smaller, slack)
    0             for edges where both endpoints are labelled STATIC (no deformation)

Part labels come from `motionprior.segmentation.parts`. A label of
`static_label` (default -1) means the Gaussian is in the frozen background.
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor

LambdaLike = Union[float, Tensor]


def _as_tensor(x: LambdaLike, ref: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=ref.device, dtype=torch.float32)
    return torch.as_tensor(float(x), dtype=torch.float32, device=ref.device)


def articulated_edge_weights(
    edges: Tensor,
    parts: Tensor,
    lambda_intra: LambdaLike,
    lambda_inter: LambdaLike,
    static_label: int = -1,
) -> Tensor:
    """Compute per-edge ARAP scaling weights.

    Args:
        edges: LongTensor of shape (E, 2) -- index pairs into the parts tensor.
        parts: LongTensor of shape (N,) -- per-control-point part label.
            Use `static_label` (default -1) for background / non-deformable.
        lambda_intra: weight for edges where both endpoints share a non-static part.
        lambda_inter: weight for edges crossing a part boundary, or static<->dynamic.
        static_label: sentinel value for static (frozen) control points.

    Returns:
        FloatTensor of shape (E,) with values in {0, lambda_inter, lambda_intra}.
    """
    li = _as_tensor(lambda_intra, parts.float())
    le = _as_tensor(lambda_inter, parts.float())
    if li.item() < 0 or le.item() < 0:
        raise ValueError(
            f"lambda values must be nonnegative; got intra={li.item()}, inter={le.item()}"
        )
    n = parts.shape[0]
    if torch.any(edges < 0) or torch.any(edges >= n):
        raise IndexError(
            f"edge indices out of range for parts tensor of length {n}"
        )

    a = parts[edges[:, 0]]
    b = parts[edges[:, 1]]
    both_static = (a == static_label) & (b == static_label)
    same_part = (a == b) & (~both_static)
    # Anything else is inter (including static<->dynamic; one rigid + one frozen
    # endpoint is correctly slack -- the frozen point's rest position serves as
    # the anchor, and inter weight permits small relative motion).

    weights = torch.zeros(edges.shape[0], dtype=torch.float32, device=parts.device)
    weights = torch.where(same_part, weights + li, weights)
    weights = torch.where(~(same_part | both_static), weights + le, weights)
    return weights
