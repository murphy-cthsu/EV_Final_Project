"""Cross-view consistency gating.

The 4th protective component in MotionPrior. It addresses the failure mode
specific to *multi-view* VGM supervision: the VGM (e.g. SV4D 2.0) produces
several views that are each temporally consistent within themselves but
mutually disagree on the underlying 3D geometry (hallucinations differ
across views). The existing per-frame ARAP-energy gating (gating.py) does
not capture this -- its trustworthiness signal is purely temporal.

CVCG's idea mirrors the existing gating contract:

    w_iter = exp(-beta(iter) * r_iter)

where r_iter is the mean photometric residual when the *current* canonical
Gaussian state is rendered into the OTHER training views at the same
timestep. beta(iter) is adaptive (normalized by an EMA of r_iter), so the
gate is scene-invariant -- same role AdaptiveAlpha plays for the
ARAP-energy variant.

Semantics of the gate:

    sibling renders match their GTs well  ==>  r_iter small  ==>  gate ~ 1
        (the canonical model already explains the sibling views; we can
         trust the current view's photometric gradient too)
    sibling renders disagree with their GTs  ==>  r_iter large  ==>  gate ~ 0
        (the canonical model can't simultaneously fit current + siblings;
         the current view is likely pulling the model toward a
         multi-view-inconsistent state -- damp the photometric gradient)

Cost: one rendering of the current canonical state into each sibling
viewpoint per training iter (4 extra renders for our 5-view scenes).
Decision rationale: see docs/design/scgs_hook_design.md (patch site D).
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_cross_view_gate(sibling_residuals: Tensor, beta: float) -> Tensor:
    """Per-iter cross-view consistency weight.

    Args:
        sibling_residuals: nonnegative tensor of shape (V_sib,) -- the L1
            photometric residual of the current canonical state rendered
            from each sibling viewpoint vs that sibling's GT image. V_sib
            is typically 4 (n_views - 1).
        beta: nonnegative scalar -- gating strength (typically from
            AdaptiveBeta below).

    Returns:
        Scalar tensor in (0, 1].
    """
    if beta < 0:
        raise ValueError(f"beta must be nonnegative, got {beta}")
    if torch.any(sibling_residuals < 0):
        raise ValueError("sibling_residuals must be nonnegative")
    if sibling_residuals.numel() == 0:
        return torch.tensor(1.0, dtype=sibling_residuals.dtype,
                            device=sibling_residuals.device)
    mean_r = sibling_residuals.mean()
    return torch.exp(-beta * mean_r)


class AdaptiveBeta:
    """beta(iter) = beta0 / EMA_iter(r_iter).

    Mirrors AdaptiveAlpha (gating.py:40). One scalar EMA over training
    iters; call once per step with the current step's mean sibling
    residual, it returns the beta to feed into compute_cross_view_gate.
    """

    def __init__(self, beta0: float, momentum: float = 0.99, eps: float = 1e-6) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if beta0 < 0:
            raise ValueError(f"beta0 must be nonnegative, got {beta0}")
        self.beta0 = float(beta0)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self._ema: float | None = None

    def __call__(self, mean_residual: Tensor) -> Tensor:
        r = float(mean_residual.detach().cpu().item())
        if self._ema is None:
            self._ema = r
        else:
            self._ema = self.momentum * self._ema + (1.0 - self.momentum) * r
        return torch.tensor(self.beta0 / max(self._ema, self.eps))

    @property
    def ema(self) -> float | None:
        return self._ema


def build_sibling_map(cam_view_idx: list[int], cam_frame_idx: list[int]) -> dict[int, list[int]]:
    """Given the per-camera (view_idx, frame_idx) metadata in train order,
    build flat-cam-idx -> list of flat-cam-idx for cameras at the SAME
    frame_idx but DIFFERENT view_idx.

    Args:
        cam_view_idx: length-N list of view indices for each training cam.
        cam_frame_idx: length-N list of frame indices for each training cam.

    Returns:
        Dict mapping each cam's index in the train list to a list of its
        sibling cam indices (same frame_idx, different view_idx).

    Raises:
        ValueError: if the two input lists have mismatched lengths.
    """
    if len(cam_view_idx) != len(cam_frame_idx):
        raise ValueError(
            f"len(cam_view_idx)={len(cam_view_idx)} != "
            f"len(cam_frame_idx)={len(cam_frame_idx)}"
        )
    by_frame: dict[int, list[int]] = {}
    for i, ft in enumerate(cam_frame_idx):
        by_frame.setdefault(int(ft), []).append(i)
    out: dict[int, list[int]] = {}
    for i, (v, ft) in enumerate(zip(cam_view_idx, cam_frame_idx)):
        siblings = [j for j in by_frame[int(ft)] if cam_view_idx[j] != v]
        out[i] = siblings
    return out
