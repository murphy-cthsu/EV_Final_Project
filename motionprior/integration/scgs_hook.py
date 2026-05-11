"""SC-GS integration hook.

The single Python object that wires our four `motionprior` components into
SC-GS's training loop. See `docs/scgs_hook_design.md` for the contract and
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
        self._warm_up = int(config.get("warm_up", 0))
        self._warned_about_pe_layout = False
        self._warned_about_extra_dxyz = False

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

    # ------------------------------------------------------------------ #
    # Patch A: temporal PE gating                                        #
    # ------------------------------------------------------------------ #

    def gate_temporal_encoding(self, time_emb: Tensor, iteration: int) -> Tensor:
        """Multiply the sinusoidal PE channels by the frequency-band mask.

        If `time_emb` last dim doesn't equal 2*num_bands, the layout isn't
        what we expect -- return the input unchanged and warn once.
        """
        if self._curriculum is None:
            return time_emb
        expected = 2 * self._curriculum.num_bands
        if time_emb.shape[-1] != expected:
            if not self._warned_about_pe_layout:
                import warnings
                warnings.warn(
                    f"MotionPriorHook: time_emb last dim {time_emb.shape[-1]} "
                    f"!= expected {expected} (2 * num_bands). Disabling "
                    f"frequency curriculum for this run."
                )
                self._warned_about_pe_layout = True
            return time_emb
        return self._curriculum.apply(time_emb, iteration)

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
        load on the GPU box. See `docs/scgs_hook_design.md` for the
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
