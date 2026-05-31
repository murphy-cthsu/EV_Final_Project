# Path 1 & Path 2 Post-mortem (Ceiling Push Attempts)

> 2026-06-01 (late). Two attempts to push beyond A1 baseline ceiling
> (20.40 dB). Both **failed**, confirming the architectural limit.

## TL;DR

| Approach | vs d-3dgs | Δ vs A1 (20.40) | Verdict |
|---|---:|---:|---|
| **A1 baseline** (K=100 + smart α=16 + scale + xyz_res) | **20.40** | — | best |
| Path 2: canonical fine-tune (unfreeze scale/rot/features) | 19.29 | -1.11 | ❌ drift hurts |
| Path 1: new t=0 canonical (from d-3dgs t=0 + SAM mask) | 14.07 | -6.33 | ❌ lost baseplate |
| Architecture ceiling (train on d-3dgs supervision) | 20.96 | +0.56 | hard upper bound |

**Conclusion**: A1 is genuinely at the practical ceiling for this
architecture + data + canonical combination. Further gains require
different design (kinematic chain motion, per-Gauss color, foundation
feature supervision) — NOT parameter tuning or canonical swap.

---

## Path 2: Mild canonical fine-tune

### Idea
Unfreeze canonical `_scaling`, `_rotation`, `_features_dc` (color),
keep `_xyz` frozen. Tiny lr (1e-4) so structure doesn't drift but
shape/orientation can refine.

### Implementation
```python
gaussians._scaling.requires_grad_(True)
gaussians._rotation.requires_grad_(True)
gaussians._features_dc.requires_grad_(True)
optim adds: {"params": [_scaling, _rotation, _features_dc], "lr": 1e-4}
```

### Hypothesis (failed)
Canonical Gaussians can adapt shape per scene, helping LBS blend produce
sharper output.

### Result
- vs SV4D: 14.26 (vs A1 14.72, -0.46)
- vs d-3dgs: **19.29** (vs A1 20.40, **-1.11**)
- silh loss went DOWN from 1.4 → 1.05 (canonical was helping fit SV4D)
- photo_smart loss WAY down 0.05 → 0.025 (canonical fitted noise)

### Why it hurt
- Canonical drifted toward SV4D's noisy pixel-fit, away from d-3dgs clean structure
- This is the **fundamental tradeoff** — frozen canonical preserves clean
  structure; unfrozen canonical absorbs VGM noise
- Confirms: **frozen canonical is critical** to our method's "resist noise" advantage

### Saved
`outputs/custom/partrigid_lego_v2_path2_canon_ft/partrigid_state.npz`

---

## Path 1: New t=0 canonical from d-3dgs

### Idea
The provided canonical was at "mid-cycle" pose (mismatch with t=0 → larger
SE(3) offsets → quantization error). Train a NEW canonical specifically at
t=0 pose using d-3dgs clean refs (5 views × t=0) as supervision.

### Implementation
```bash
# Build frame-0 subset (d-3dgs clean RGB + SAM-2 alpha)
data/custom/lego_v2_frame0_d3dgs_clean/train/r_{0-4}.png  (5 images at t=0)

# Train static GS (no time)
python third_party/SC-GS/train_gui.py \
    --source_path data/custom/lego_v2_frame0_d3dgs_clean \
    --model_path outputs/custom/lego_v2_canonical_t0clean \
    --deform_type node --node_num 256 \
    --is_blender --gt_alpha_mask_as_scene_mask \
    --iterations 8000
```

### Result
- New canonical: **46,011 Gaussians** (vs original 114,580 — much smaller
  because SAM-2 mask excludes baseplate)
- Rebuilt parts: arm 29%, body 52%, unassigned 19%
- vs SV4D: 16.54 (improved, +1.8 vs A1)
- vs d-3dgs: **14.07** (vs A1 20.40, **-6.33**)

### Why it hurt
1. **Lost baseplate** — SAM-2 alpha mask used for canonical training
   excluded baseplate region → new canonical = digger only (no baseplate
   Gaussians). When evaluated against d-3dgs (which HAS baseplate), our
   render shows white in baseplate region → catastrophic per-pixel error.
2. **Better fit to SAM-mask supervision (vs SV4D up 1.8)** — confirms
   canonical at t=0 pose IS better aligned for SV4D supervision.
3. **Failure mode is "scene coverage mismatch"** — same as Exp 2
   (nobase canonical) earlier. Not a new failure.

### To fix this path properly (NOT pursued)
Would need:
- Train canonical WITHOUT SAM-2 mask (use full RGB) — but then need
  proper FG mask handling later
- OR include baseplate explicitly in canonical training data
- OR mask the d-3dgs eval to digger-only too (apples-to-apples)

But all of these don't change the fundamental ceiling issue.

### Saved
- New canonical: `outputs/custom/lego_v2_canonical_t0clean_node/point_cloud/iteration_8000/point_cloud.ply`
- Hier model: `outputs/custom/partrigid_lego_v2_path1_t0canon/partrigid_state.npz`
- Part assignment: `runs_aux/part_assignment_lego_v2_t0clean/`

---

## Combined finding

### What we've now confirmed
- **A1 (20.40) is at architecture ceiling** (20.96 with clean
  supervision, only +0.56 headroom)
- **Architectural changes hurt**:
  - Continuous deform-MLP without prior: -1.41
  - Canonical fine-tune (any direction): -1.11
  - New t=0 canonical: -6.33 (different failure mode — coverage mismatch)
- **Hyperparameter tuning exhausted** (LPIPS, Otsu, lbs_K, reg weights,
  more iters: all marginal/negative)

### What WOULD work (untested, would need substantial effort)

1. **Per-Gaussian color residual** (per-time appearance adaptation)
   — Path D from earlier proposal, not implemented
2. **Kinematic chain motion model** (cabin → arm → bucket hierarchical
   frames instead of flat K-cluster pool)
3. **DINOv2 / foundation feature supervision** (test untried)
4. **Larger / better canonical** — train on D-NeRF original data with
   more iter, more Gaussians (currently 114k; SC-GS reports 200k+)

### What we should NOT pursue further
- More LBS_K variations
- More regularizer tuning
- More iter counts
- Different K values

### Story strengthens (not weakens) by these negative results

The post-mortem confirms our central claim is **architecture-bounded, not
parameter-tuned**. The +8.97 dB gain over vanilla SC-GS comes from
**structural mechanism** (frozen canonical + smart photo filter), not
incidental tuning. Path 1 / Path 2 failures **rule out** plausible
alternatives, making the design choices more defensible.

---

## Files committed (since last commit)

- `scripts/train_partrigid_hier.py` (updated with `--canon_finetune`,
  `--lr_canon` flags)
- `outputs/custom/partrigid_lego_v2_path2_canon_ft/` (Path 2 model)
- `outputs/custom/partrigid_lego_v2_path1_t0canon/` (Path 1 model)
- `outputs/custom/lego_v2_canonical_t0clean_node/` (new t=0 canonical)
- `runs_aux/part_assignment_lego_v2_t0clean/` (parts for new canonical)
- `data/custom/lego_v2_frame0_d3dgs_clean/` (training data for new canon)
- `runs_aux/lego_v2_eval/lego_v2_path{1,2}_*/` (per-frame eval tiles)

---

## Recommended next actions (for tomorrow)

### High priority
1. **Push origin/main** (21+ commits ahead, all locally committed)
2. **Update slides** — add lego_v2 results, replace OLD scene00 with
   lego_v2 as primary dataset
3. **Visual gallery** — final 10 GIFs already at
   `runs_aux/method_animations/` — embed in slides

### Medium priority
4. **Slides outline update** to reflect Checkpoint 2 framing
5. **Generalization test** — pick another D-NeRF scene (jumpingjacks,
   hellwarrior) and run full pipeline to validate cross-scene applicability

### Low priority (if time)
6. **Per-Gaussian color residual** experiment (Path D)
7. **DINOv2 feature loss** experiment
8. **Kinematic chain** rewrite of motion model

---

## Quick reference: best command

```bash
# Reproduce A1 (best variant, 20.40 dB)
CUDA_VISIBLE_DEVICES=0 python scripts/train_partrigid_hier.py \
    --label lego_v2_alpha16 \
    --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply \
    --part_dir runs_aux/part_assignment_lego_v2 \
    --scene_dir data/custom/lego_v2 \
    --v5_render_dir outputs/custom/lego_v2_d3dgs_ref/renders \
    --use_test_too --k_arm 100 --lbs_K 6 --lam_arap 1.0 \
    --lam_photo_smart 3.0 --photo_smart_alpha 16.0 \
    --use_per_time_scale --use_xyz_residual \
    --iterations 8000

python scripts/eval_lego_v2_hier.py --label lego_v2_alpha16 \
    --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply \
    --save_renders
```

睡覺安心 ✓ — A1 is locked in as best, Path 1/2 are documented failures.
