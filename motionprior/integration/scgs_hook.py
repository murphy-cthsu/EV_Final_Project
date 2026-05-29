"""SC-GS integration hook.

The single Python object that wires our four `motionprior` components into
SC-GS's training loop. See `docs/design/scgs_hook_design.md` for the contract and
the three patch sites in `train_gui.py`.

Design principle: every method degrades to a no-op (returns identity / 1.0 /
zero loss) if its component is disabled or its inputs don't fit the expected
layout. This keeps SC-GS's training loop running even when individual
components are turned off for ablation rows.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from motionprior.curriculum.frequency import FrequencyCurriculum
from motionprior.losses.gating import compute_gating_weights, AdaptiveAlpha
from motionprior.losses.arap_articulated import articulated_edge_weights
from motionprior.losses.rest_state import rest_state_l2
from motionprior.losses.cross_view_consistency import (
    AdaptiveBeta,
    build_sibling_map,
    compute_cross_view_gate,
)


class MotionPriorHook:
    """Single entrypoint used by SC-GS's training loop.

    Owns the per-run state: AdaptiveAlpha's EMA, FrequencyCurriculum's
    schedule, and the precomputed scene-level tensors (ARAP-prior energies,
    part labels, K-NN edges, rest positions).
    """

    def __init__(
        self,
        config: dict[str, Any],
        arap_prior_energies: Tensor,
        part_labels: Tensor,
        edges: Tensor,
        rest_positions: Tensor,
    ) -> None:
        self.cfg = config
        self.arap_prior_energies = arap_prior_energies
        self.part_labels = part_labels
        self.edges = edges
        self.rest_positions = rest_positions

        # Build per-component state. Each component reads its own subsection
        # of the config; missing subsections default to disabled.
        self._curriculum = self._build_curriculum()
        self._adaptive_alpha = self._build_adaptive_alpha()
        self._adaptive_beta = self._build_adaptive_beta()
        self._sibling_map: dict[int, list[int]] | None = None
        self._warm_up = int(config.get("warm_up", 0))
        self._warned_about_pe_layout = False
        self._warned_about_extra_dxyz = False
        self._warned_about_missing_siblings = False
        # Set per-training-step by the loop, read by DeformNetwork.forward
        # for patch A's iteration-aware gating.
        self._current_iteration = 0

    # ------------------------------------------------------------------ #
    # Component builders                                                 #
    # ------------------------------------------------------------------ #

    def _build_curriculum(self) -> FrequencyCurriculum | None:
        c = self.cfg.get("frequency_curriculum", {})
        if not c.get("enabled", True):
            return None
        return FrequencyCurriculum(
            num_bands=c["num_bands"],
            milestones=c["milestones"],
            k_at_milestone=c["k_at_milestone"],
        )

    def _build_adaptive_alpha(self) -> AdaptiveAlpha | None:
        g = self.cfg.get("gating", {})
        if not g.get("enabled", True):
            return None
        return AdaptiveAlpha(
            alpha0=g.get("alpha0", 1.0),
            momentum=g.get("ema_momentum", 0.99),
            eps=g.get("eps", 1e-6),
        )

    def _build_adaptive_beta(self) -> AdaptiveBeta | None:
        c = self.cfg.get("cross_view_gating", {})
        if not c.get("enabled", False):
            return None
        return AdaptiveBeta(
            beta0=c.get("beta0", 1.0),
            momentum=c.get("ema_momentum", 0.99),
            eps=c.get("eps", 1e-6),
        )

    def configure_sibling_map(
        self, cam_view_idx: list[int], cam_frame_idx: list[int]
    ) -> None:
        """Build the cam_idx -> [sibling cam_idx, ...] map from train cameras.

        Call once at training start (after Scene is loaded). The hook caches
        the result and reuses it across iters. Cheap (O(N)) and required
        before cross_view_gating can return a non-trivial value.
        """
        self._sibling_map = build_sibling_map(cam_view_idx, cam_frame_idx)

    # ------------------------------------------------------------------ #
    # Patch A: temporal PE gating                                        #
    # ------------------------------------------------------------------ #

    def gate_temporal_encoding(self, time_emb: Tensor, iteration: int) -> Tensor:
        """Multiply the sinusoidal PE channels by the frequency-band mask.

        Supports two PE layouts:
            * (..., 2*num_bands)         -- pure sinusoidal layout
            * (..., 1 + 2*num_bands)     -- NeRF / SC-GS layout with identity
              prefix; gates only the sin/cos tail, identity passes through

        If `time_emb` last dim matches neither, returns input unchanged and
        warns once.
        """
        if self._curriculum is None:
            return time_emb
        expected_pure = 2 * self._curriculum.num_bands
        expected_id = 1 + 2 * self._curriculum.num_bands
        last_dim = time_emb.shape[-1]
        if last_dim == expected_pure:
            return self._curriculum.apply(time_emb, iteration)
        if last_dim == expected_id:
            identity = time_emb[..., :1]
            tail = time_emb[..., 1:]
            gated = self._curriculum.apply(tail, iteration)
            return torch.cat([identity, gated], dim=-1)
        if not self._warned_about_pe_layout:
            import warnings
            warnings.warn(
                f"MotionPriorHook: time_emb last dim {last_dim} matches "
                f"neither pure ({expected_pure}) nor identity-prefixed "
                f"({expected_id}) layout. Disabling frequency curriculum."
            )
            self._warned_about_pe_layout = True
        return time_emb

    # ------------------------------------------------------------------ #
    # Patch B: photometric gating                                        #
    # ------------------------------------------------------------------ #

    def photometric_gating(self, fid: float, iteration: int) -> float:
        """Scalar in (0, 1] to scale the per-iteration photometric loss.

        During warmup, returns 1.0 unconditionally so SC-GS's pre-deform
        Gaussian initialization runs without gating noise.
        """
        if self._adaptive_alpha is None:
            return 1.0
        if iteration < self._warm_up:
            return 1.0
        T = self.arap_prior_energies.numel()
        if T == 0:
            return 1.0
        # Continuous fid in [0, 1] -> nearest precomputed frame index.
        idx_f = float(fid) * (T - 1)
        idx = int(round(max(0.0, min(idx_f, T - 1))))
        E_t = self.arap_prior_energies[idx].clamp(min=0.0)
        # Update EMA with current frame energy and get alpha.
        alpha = self._adaptive_alpha(E_t)
        weights = compute_gating_weights(E_t.unsqueeze(0), alpha=alpha.item())
        return float(weights[0].item())

    # ------------------------------------------------------------------ #
    # Patch D: cross-view consistency gating                             #
    # ------------------------------------------------------------------ #

    def cross_view_gating(
        self,
        cam_idx: int,
        iteration: int,
        render_sibling: Any,
    ) -> float:
        """Scalar in (0, 1] to scale the per-iter photometric loss.

        Mirrors `photometric_gating` (patch B) but the trustworthiness
        signal is *cross-view* photometric consistency, not lifted-flow
        ARAP energy. Designed for multi-view VGM supervision where the
        VGM hallucinates different content into different views at the
        same timestep.

        Cost: one extra rendering per sibling per iter. With 5-view
        scenes (4 siblings) this is roughly 5x the per-iter forward cost
        of vanilla SC-GS. See docs/design/scgs_hook_design.md (patch site D).

        Args:
            cam_idx: index of the current training viewpoint in the train
                camera list.
            iteration: current main-training iter (post-pre-training).
            render_sibling: callable `(sib_cam_idx) -> (rendered_img, gt_img)`
                where both tensors are (3, H, W) in [0, 1]. The loop owns
                the deformation MLP and the renderer; the hook only owns
                the gating math. Must be a no-op-safe pure function -- the
                hook never calls .backward() on its outputs (residual is
                detached internally).

        Returns:
            Scalar in (0, 1]. Returns 1.0 (no-op) if the component is
            disabled, during warm-up, or if the sibling map is missing.
        """
        if self._adaptive_beta is None:
            return 1.0
        if iteration < self._warm_up:
            return 1.0
        if self._sibling_map is None:
            if not self._warned_about_missing_siblings:
                import warnings
                warnings.warn(
                    "MotionPriorHook.cross_view_gating: no sibling map "
                    "configured; call configure_sibling_map() at training "
                    "start. Disabling cross-view gating for this run."
                )
                self._warned_about_missing_siblings = True
            return 1.0

        siblings = self._sibling_map.get(int(cam_idx), [])
        if not siblings:
            return 1.0

        residuals: list[Tensor] = []
        with torch.no_grad():
            for sib_idx in siblings:
                out = render_sibling(sib_idx)
                if out is None:
                    continue
                sib_image, sib_gt = out
                # L1 between rendered and GT for the sibling view.
                r = (sib_image - sib_gt).abs().mean()
                residuals.append(r.detach())
        if not residuals:
            return 1.0
        r_vec = torch.stack(residuals)  # (V_sib,)
        mean_r = r_vec.mean()
        beta = self._adaptive_beta(mean_r)
        w = compute_cross_view_gate(r_vec, beta=float(beta.item()))
        return float(w.item())

    # ------------------------------------------------------------------ #
    # Patch C: extra losses (rest-state L2)                              #
    # ------------------------------------------------------------------ #

    def extra_losses(self, d_xyz: Tensor, iteration: int) -> Tensor:
        """Scalar tensor to add to the total loss after `loss = loss + reg_loss`.

        Currently: rest-state L2 anchor (eta * ||d_xyz||^2_mean).
        """
        r = self.cfg.get("rest_state", {})
        if not r.get("enabled", True):
            return torch.tensor(0.0, dtype=torch.float32, device=d_xyz.device)
        if d_xyz.dim() != 2 or d_xyz.shape[-1] != 3:
            if not self._warned_about_extra_dxyz:
                import warnings
                warnings.warn(
                    f"MotionPriorHook: d_xyz has unexpected shape "
                    f"{tuple(d_xyz.shape)} (expected (N, 3)); skipping rest-state loss."
                )
                self._warned_about_extra_dxyz = True
            return torch.tensor(0.0, dtype=torch.float32, device=d_xyz.device)
        # Rest positions are zeros in the deformation frame; d_xyz IS the displacement.
        eta = float(r.get("eta", 0.001))
        zeros = torch.zeros_like(d_xyz)
        return eta * rest_state_l2(d_xyz, zeros)

    # ------------------------------------------------------------------ #
    # Articulation-aware ARAP — exposed to SC-GS's ARAPDeformer          #
    # ------------------------------------------------------------------ #

    def articulated_edge_weights(self) -> Tensor:
        """Per-edge ARAP scaling weights.

        SC-GS's `ARAPDeformer.cal_arap_error` multiplies its per-edge energy
        by a uniform scalar. We replace that with per-edge weights from this
        method via a monkey-patch installed by `install_articulated_arap`.

        If articulation is disabled, returns a uniform `lambda_intra` vector
        so SC-GS's behavior is unchanged.
        """
        a = self.cfg.get("articulation", {})
        intra = float(a.get("lambda_intra", 1.0))
        inter = float(a.get("lambda_inter", 0.05))
        static = int(a.get("static_label", -1))
        if not a.get("enabled", True):
            return torch.full((self.edges.shape[0],), intra, dtype=torch.float32)
        return articulated_edge_weights(
            self.edges,
            self.part_labels,
            lambda_intra=intra,
            lambda_inter=inter,
            static_label=static,
        )

    def install_articulated_arap(self, deform_model: Any) -> None:
        """Monkey-patch SC-GS's ARAPDeformer to use our per-edge lambdas.

        This is called once at the start of training, after SC-GS has built
        its deformation model. We don't define the patch here in pure form
        because it depends on the exact attribute layout of
        `third_party/SC-GS/utils/arap_deform.py:ARAPDeformer`, which we only
        load on the GPU box. See `docs/design/scgs_hook_design.md` for the
        installation procedure on the RunPod side.

        Stub here so SC-GS-less unit tests still import the module.
        """
        a = self.cfg.get("articulation", {})
        if not a.get("enabled", True):
            return
        # The actual monkey-patch is performed by a small SC-GS-aware adapter
        # in `motionprior/integration/scgs_arap_adapter.py` (created when we
        # have SC-GS source loaded). Calling it here without SC-GS imports is
        # intentionally a no-op so the unit test suite remains hermetic.
        return
