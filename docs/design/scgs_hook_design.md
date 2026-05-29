# SC-GS Hook Design

> How the four `motionprior` components plug into SC-GS's training loop.
> Read alongside `third_party/SC-GS/train_gui.py` (pinned SHA `3a9d2ad4`).

## SC-GS structure (relevant pieces)

| File | Object | Role |
|---|---|---|
| `train_gui.py` | `Trainer` class | Main training loop. The function around lines 1100–1190 assembles the per-iteration loss. |
| `scene/deform_model.py` | `DeformModel` | Wraps the deformation MLP. Exposes `step(xyz, time_emb, iteration, ...)` and `reg_loss` (the ARAP regularizer). |
| `utils/arap_deform.py` | `ARAPDeformer` | The ARAP implementation. Uses K-NN edges computed in `utils/deform_utils.py:cal_connectivity_from_points`. |
| `utils/deform_utils.py` | — | Edge graph construction; Laplacian; flow-related helpers. |

## Loss assembly in `train_gui.py` (the integration target)

The relevant lines (paraphrased; see source for exact context):

```python
# Line 1118
Ll1 = l1_loss(image, gt_image)
loss_img = (1 - lambda_dssim) * Ll1 + lambda_dssim * (1 - ssim(image, gt_image))
loss = loss_img

# Line 1122
if iteration > warm_up:
    loss = loss + self.deform.reg_loss      # <-- ARAP regularizer is here

# Line 1125
# ... flow loss, motion mask loss, etc.

loss.backward()
```

The deformation MLP's forward call sits earlier (~line 1080):

```python
d_xyz, d_rotation, d_scaling, ... = self.deform.step(
    self.gaussians.get_xyz.detach(),
    time_input + ast_noise,
    iteration=self.iteration,
    ...
)
```

`time_input` is produced by `self.deform.deform.expand_time(fid)` — this is the temporal positional encoding we want to gate.

## Hook integration points

Four patches to `train_gui.py`, all small:

| # | Where | Patch |
|---|---|---|
| **A** | Just after `time_input = ...` (~line 1075) | `time_input = hook.gate_temporal_encoding(time_input, iteration)` — applies frequency curriculum. |
| **B** | Replace `Ll1 = l1_loss(...)` (line 1118) with masked version | `Ll1 = l1_loss(image, gt_image) * hook.photometric_gating(fid, iteration)` — applies ARAP-energy gating. |
| **C** | After `loss = loss + self.deform.reg_loss` (line 1123) | `loss = loss + hook.extra_losses(d_xyz, iteration)` — adds rest-state L2. |
| **E** | Same site as B, multiplicatively combined | `Ll1 *= hook.cross_view_gating(cam_idx, iteration, render_sibling)` — applies cross-view consistency gating. The `render_sibling` closure renders the *current* canonical state (reusing the iter's `d_xyz/d_rotation/d_scaling`) from sibling viewpoints at the same `fid` and returns `(render, gt)` per sibling. **Adds ~5× per-iter forward cost** for 5-view scenes. |

Articulation-aware ARAP requires a fourth change inside `DeformModel.reg_loss` (or `ARAPDeformer.energy`): per-edge weights from `motionprior.losses.articulated_edge_weights`. This is the only patch that touches SC-GS internals rather than the training loop. Done by monkey-patching `ARAPDeformer.cal_arap_error` at hook construction time.

### Why patch E exists

The framework's three protective components (B = ARAP-energy gating, C3 = frequency curriculum, the inter-part ARAP) all target *temporal* failure modes of generative-video supervision — drift, jitter, per-frame hallucination. They do **not** address the failure mode specific to *multi-view* VGM output (SV4D 2.0 etc.): the V views are each temporally consistent within themselves but mutually disagree on the underlying 3D geometry. The W3 measurement on `data/custom/scene00_split` (5-view × 21-frame SV4D output, view-2 held out) made this concrete: PSNR 31 dB on train views, 13 dB on the held-out view — an 18 dB gap caused by canonical Gaussians being placed to fit 4 mutually-inconsistent views without geometric triangulation constraint.

CVCG generalizes patch B's *trust-by-consistency* contract from the temporal axis (lifted-trajectory rigidity) to the spatial / multi-view axis (cross-view photometric agreement). The math is the same: `w = exp(−β · residual)` with adaptive `β` normalized by an EMA. Only the residual's source differs. Composable: rows B+E together gate by both temporal and spatial trustworthiness signals.

## The hook contract

```python
class MotionPriorHook:
    def __init__(
        self,
        config: dict,             # from motionprior.configs
        arap_prior_energies: torch.Tensor,  # (T,) precomputed offline
        part_labels: torch.Tensor,          # (N_gaussians,) from segmentation
        edges: torch.Tensor,                # (E, 2) K-NN edges -- mirrors SC-GS's connectivity
        rest_positions: torch.Tensor,       # (N_gaussians, 3) canonical positions
    ):
        ...

    def gate_temporal_encoding(self, time_emb: Tensor, iteration: int) -> Tensor:
        """Patch A. Apply frequency curriculum to the temporal PE.

        If `time_emb` has shape (..., 2*num_bands) and is a standard sinusoidal
        PE, multiplies high-freq channels by 0 according to curriculum schedule.
        If shape doesn't match expected layout, returns input unchanged (no-op
        fallback) and logs a warning once.
        """

    def photometric_gating(self, fid: float, iteration: int) -> float:
        """Patch B. Scalar in (0, 1] for the photometric loss this iteration.

        Maps the current frame id (continuous in [0, 1]) to its precomputed
        ARAP-prior energy and returns `exp(-alpha(t) * E_t)` via AdaptiveAlpha.

        After warmup, gating is active; before warmup, returns 1.0 unconditionally.
        """

    def extra_losses(self, d_xyz: Tensor, iteration: int) -> Tensor:
        """Patch C. Scalar tensor to add to the total loss.

        Currently just rest-state L2:  eta * ||d_xyz||^2_mean.
        Returns 0.0 tensor if not enabled in config.
        """

    def install_articulated_arap(self, deform_model) -> None:
        """Monkey-patch SC-GS's ARAPDeformer to use per-edge intra/inter lambdas.

        Called once at construction; replaces `deform_model.deform.arap_deformer.cal_arap_error`
        with a version that scales per-edge energies by articulated_edge_weights().
        """
```

## Why a hook (not a fork)

- We keep `third_party/SC-GS` at the pinned upstream SHA — no merge maintenance.
- Patches A/B/C are 3 single-line additions to `train_gui.py`. Maintained as a `.patch` file under `scripts/patches/scgs_hook.patch` — applied by `setup_runpod.sh` after cloning.
- The articulation install runs in our hook's `__init__`; no SC-GS source modification.

## Failure modes and fallbacks

| Failure | Mitigation |
|---|---|
| SC-GS's `expand_time` returns a learned embedding (not standard sinusoidal) | `gate_temporal_encoding` detects the shape and returns input unchanged; we run with `frequency_curriculum.enabled=False`. |
| ARAP-prior energies haven't been precomputed for the scene | `photometric_gating` returns 1.0 with a warning; training proceeds without gating. |
| Part labels not available (SAM 2 didn't run on this scene) | `install_articulated_arap` returns without monkey-patching; uniform ARAP runs. |
| `d_xyz` has unexpected dtype/shape | `extra_losses` returns 0 with a one-time warning; rest-state disabled for the run. |

Each component degrades **gracefully** to baseline SC-GS behavior. The full method requires all four to work; partial method (e.g., articulation only) still trains and is a valid ablation row.

## What's NOT in the hook

- Dataset loading, optimizer setup, checkpoint saving — handled by SC-GS unchanged.
- The deformation MLP itself — we don't replace it.
- The renderer — unchanged.
- Camera pose handling — we use SC-GS's existing pose pipeline.

## Open items for W1 of the experiment timeline

1. Confirm `self.deform.deform.expand_time` source — is it standard sinusoidal PE? (action: read `scene/deform_model.py` once on RunPod with the env built).
2. Confirm `ARAPDeformer.cal_arap_error` is the right monkey-patch target (action: read `utils/arap_deform.py:96-189`).
3. The K-NN edge graph in SC-GS uses ~512 control points (not dense Gaussians). Our `part_labels` must align — i.e., we segment parts at the *control point* level, not Gaussian level. (Action: add a downsampler from Gaussian-level part labels to control-point-level part labels in the hook.)
4. SC-GS already has a flow loss that uses a "pixel rgb residual" mask as weighting — this is a third selective-trust signal we should mention in the paper for completeness. It's appearance-residual-based, distinct from ours.
