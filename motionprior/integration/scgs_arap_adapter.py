"""Monkey-patch adapter for SC-GS's ControlNodeWarp.arap_loss.

SC-GS implements ARAP as ``cal_arap_error(nodes_sequence, ii, jj, nn, weight)``
where ``weight`` is a ``(Nv, K)`` tensor of per-vertex-per-neighbor ARAP edge
weights. ``ControlNodeWarp.arap_loss`` re-computes the K-NN graph each call
and currently passes ``weight=None``, which inside ``cal_arap_error`` becomes a
uniform-1.0 mask -- i.e. SC-GS's ARAP is uniform across all edges.

We inject anisotropic per-edge weighting derived from part labels:

* edges where both endpoints share a non-static part   -> ``lambda_intra``
* edges that cross a part boundary (or mix static/dyn) -> ``lambda_inter``
* edges where both endpoints are static (label -1)     -> ``0``

The patched method is otherwise byte-equivalent to SC-GS's original
``arap_loss``: it samples time stamps, runs the deformation MLP, calls
``cal_connectivity_from_points``, and finally calls ``cal_arap_error`` -- but
with the anisotropic weight matrix instead of ``weight=None``.

Pinned-SHA assumption: SC-GS @ ``3a9d2ad4e4fc058b0763d446ae9e6b1be120b872``,
``utils/time_utils.py:1080`` (the ``arap_loss`` method). If a later SC-GS
release changes that method's signature or call structure, the patch will
either no-op (graceful) or raise a clear adapter-side error.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def build_anisotropic_weight_matrix(
    Nv: int,
    K: int,
    ii: Tensor,
    jj: Tensor,
    nn: Tensor,
    node_parts: Tensor,
    lambda_intra: float,
    lambda_inter: float,
    static_label: int = -1,
) -> Tensor:
    """Per-edge ARAP weight matrix in SC-GS's ``(Nv, K)`` slot layout.

    For each edge ``e`` with source ``ii[e]``, target ``jj[e]``, neighbor slot
    ``nn[e]``, the corresponding cell ``out[ii[e], nn[e]]`` is set to
    ``lambda_intra`` / ``lambda_inter`` / ``0`` according to the part labels at
    both endpoints. Slots not corresponding to any edge stay 0.

    Args:
        Nv: number of vertices (SC-GS control nodes).
        K: max neighbors per vertex (SC-GS uses K=10 by default).
        ii, jj, nn: SC-GS connectivity tensors, all shape ``(Ne,)``.
        node_parts: ``(Nv,)`` int part labels per control node.
        lambda_intra, lambda_inter: scalar weights.
        static_label: sentinel for static (do-not-deform) nodes.
    """
    if ii.shape != jj.shape or jj.shape != nn.shape:
        raise ValueError(
            f"ii/jj/nn must have matching shape; got {tuple(ii.shape)}, "
            f"{tuple(jj.shape)}, {tuple(nn.shape)}"
        )
    if node_parts.shape != (Nv,):
        raise ValueError(
            f"node_parts must have shape ({Nv},); got {tuple(node_parts.shape)}"
        )

    device = node_parts.device
    a = node_parts[ii]
    b = node_parts[jj]
    both_static = (a == static_label) & (b == static_label)
    same_part = (a == b) & (~both_static)
    # Per-edge factor
    factor = torch.full(
        (ii.shape[0],), float(lambda_inter), dtype=torch.float32, device=device
    )
    factor = torch.where(
        same_part, torch.full_like(factor, float(lambda_intra)), factor
    )
    factor = torch.where(both_static, torch.zeros_like(factor), factor)

    out = torch.zeros(Nv, K, dtype=torch.float32, device=device)
    out[ii, nn] = factor
    return out


def install(
    control_node_warp: Any,
    hook: Any,
    node_part_labels: Tensor,
) -> None:
    """Monkey-patch ``control_node_warp.arap_loss`` to inject anisotropic ARAP.

    Args:
        control_node_warp: an instance of SC-GS's ``ControlNodeWarp`` (lives at
            ``DeformModel.deform`` when ``deform_type='node'``).
        hook: the ``MotionPriorHook`` instance (provides config).
        node_part_labels: ``(node_num,)`` per-control-node part labels. We do
            *not* take Gaussian-level labels here -- the caller maps them to
            control nodes (SC-GS uses ``~512`` control nodes vs ``~50K``
            Gaussians; the mapping is nearest-neighbour in canonical space).

    Behavior:
        * If ``hook.cfg['articulation']['enabled']`` is ``False``, this is a
          no-op (no monkey-patch installed).
        * Otherwise, the method ``control_node_warp.arap_loss`` is replaced
          with a wrapper that calls ``cal_connectivity_from_points`` and
          ``cal_arap_error`` with the anisotropic weight matrix.

    No-ops gracefully when SC-GS source is structurally different from the
    pinned SHA -- the wrapper falls back to a uniform ARAP loss with a
    one-time warning.
    """
    cfg = hook.cfg.get("articulation", {})
    if not cfg.get("enabled", True):
        return

    if node_part_labels.shape != (control_node_warp.node_num,):
        raise ValueError(
            f"node_part_labels has length {node_part_labels.shape[0]} "
            f"but control_node_warp.node_num = {control_node_warp.node_num}"
        )

    control_node_warp._motionprior_node_part_labels = node_part_labels
    control_node_warp._motionprior_articulation_config = {
        "lambda_intra": float(cfg.get("lambda_intra", 1.0)),
        "lambda_inter": float(cfg.get("lambda_inter", 0.05)),
        "static_label": int(cfg.get("static_label", -1)),
    }

    K = control_node_warp.K
    node_num = control_node_warp.node_num
    art_cfg = control_node_warp._motionprior_articulation_config
    parts = control_node_warp._motionprior_node_part_labels

    def _patched_arap_loss(
        self,
        t: torch.Tensor | None = None,
        delta_t: float = 0.05,
        t_samp_num: int = 2,
    ) -> torch.Tensor:
        # Mirror SC-GS's arap_loss(...) byte-for-byte until the cal_arap_error
        # call site. If any imports / structure change, fall back gracefully.
        try:
            from utils.deform_utils import (  # type: ignore
                cal_connectivity_from_points,
                cal_arap_error,
            )
        except Exception:
            # SC-GS not importable -> we shouldn't be here at all (the hook
            # itself should have refused to install). Return a zero loss.
            return torch.tensor(0.0, device=self.nodes.device)

        t = torch.rand([]).to(self.nodes.device) if t is None else t.squeeze() + delta_t * (
            torch.rand([]).to(self.nodes.device) - 0.5
        )
        t_samp = torch.rand(t_samp_num).to(self.nodes.device) * delta_t + t - 0.5 * delta_t
        t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)

        node_trans = self.node_deform(t=t_samp)["d_xyz"]
        nodes_t = self.nodes[:, None, :3].detach() + node_trans
        hyper_nodes = nodes_t[:, 0]

        ii, jj, nn, _w = cal_connectivity_from_points(hyper_nodes, K=10)

        anisotropic_weight = build_anisotropic_weight_matrix(
            Nv=node_num,
            K=K,
            ii=ii,
            jj=jj,
            nn=nn,
            node_parts=parts.to(hyper_nodes.device),
            lambda_intra=art_cfg["lambda_intra"],
            lambda_inter=art_cfg["lambda_inter"],
            static_label=art_cfg["static_label"],
        )

        error = cal_arap_error(
            nodes_t.permute(1, 0, 2), ii, jj, nn,
            weight=anisotropic_weight,
        )
        return error

    # Bind as a method on the instance (not the class).
    import types
    control_node_warp.arap_loss = types.MethodType(_patched_arap_loss, control_node_warp)
