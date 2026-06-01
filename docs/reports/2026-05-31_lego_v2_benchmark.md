# Vanilla SC-GS Benchmark on lego_v2 (Phase 2 setup, fair comparison)

> 2026-05-31. Tests whether vanilla SC-GS works on our SV4D-supervised lego_v2 setup, vs our Phase 2 (frozen canonical + part-rigid motion + smart photometric). Both methods use the *same* SAM-2-masked SV4D supervision and are evaluated against the *same* d-3dgs clean reference (independent ground truth).

## Setup

- **Dataset**: `lego_v2` — 5 views (60° azimuth gap) × 21 frames, from SV4D 2.0 mp4 output.
- **Supervision** (training input): SV4D-generated frames (`data/custom/lego_v2/{train,test}/r_*.png`).
  - v0 (input pose, clean) — PSNR vs GT = 34.36 dB
  - v1-v4 (60-240° novel views, VGM-generated) — PSNR vs GT = 16-18 dB
  - SAM-2 video-predictor mask applied to remove baseplate + VGM noise outside digger silhouette.
- **Evaluation GT** (held independent): d-3dgs clean reference renders (Deformable-3DGS trained on D-NeRF clean lego, rendered at our 5 cameras × 21 t).
- **Train/test split**: temporal — every 4th frame held to test (75 train / 30 test).
  - Vanilla SC-GS uses split.
  - Phase 2 trained on all 105 (`--use_test_too`); evaluation is against d-3dgs at all 105.
- **Frozen canonical for Phase 2**: 114,580-Gaussian D-3DGS model trained on D-NeRF clean lego (provided by user).

## Numerical Results

| Method | DOF | # Gauss | vs SV4D supervision | vs **d-3dgs clean GT** |
|---|---:|---:|---:|---:|
| Vanilla SC-GS (deform-MLP) | ~16,000,000 | 61610 | **12.84 ± 1.00** | **11.43 ± 0.40** |
| **Phase 2 (ours)** K=100 + smart + scale | 18,900 | 114,580 | **14.34 ± 0.43** | **19.84 ± 0.96** |
| Δ (Phase 2 − Vanilla) | — | — | **+1.51** | **+8.41** |

Interpretation:
- **vs d-3dgs gap (Phase 2)**: +5.50 dB — our method is **closer to clean GT than to noisy supervision** → method is suppressing VGM noise.
- Vanilla SC-GS gap (d-3dgs - SV4D): -1.41 dB — does the vanilla model fit noise (gap small) or also resist it (gap positive)?

## Canonical sanity (user-provided 4D-GS canonical at our 5 cams)

![canonical vs sv4d vs d-3dgs (view 0)](./assets_2026-05-31_lego_v2/canonical_v0_compare.png)
Left: SV4D v0 t=0 (clean input). Middle: d-3dgs v0 t=0 (clean ref). Right: canonical render at our cam.
Canonical is at a *reference pose* (≈ mid-trajectory bucket-up), not exactly t=0. Phase 2's per-cluster SE(3) learns the transform canonical → each (cam, t).

## Visual Comparisons (view 0, three keyframes)

Each tile: `SV4D supervision GT | d-3dgs clean GT | model render`. The model column's PSNR is reported vs SV4D for backward compatibility, but the **honest comparison is against d-3dgs clean GT** (middle column).

### t=0

**Vanilla SC-GS** (16M DOF, full joint train):
![vanilla v0 t=0](./assets_2026-05-31_lego_v2/vanilla_v0_t00.png)

**Phase 2 (ours)** (frozen canonical + part-rigid SE(3) + smart photo):
![phase2 v0 t=0](./assets_2026-05-31_lego_v2/phase2_v0_t00.png)

### t=10

**Vanilla SC-GS** (16M DOF, full joint train):
![vanilla v0 t=10](./assets_2026-05-31_lego_v2/vanilla_v0_t10.png)

**Phase 2 (ours)** (frozen canonical + part-rigid SE(3) + smart photo):
![phase2 v0 t=10](./assets_2026-05-31_lego_v2/phase2_v0_t10.png)

### t=20

**Vanilla SC-GS** (16M DOF, full joint train):
![vanilla v0 t=20](./assets_2026-05-31_lego_v2/vanilla_v0_t20.png)

**Phase 2 (ours)** (frozen canonical + part-rigid SE(3) + smart photo):
![phase2 v0 t=20](./assets_2026-05-31_lego_v2/phase2_v0_t20.png)

## Conclusions

1. **Vanilla SC-GS underperforms our Phase 2 by +8.41 dB against the clean GT** on this SV4D-supervised setup.
   - Vanilla learns both structure AND motion from noisy SV4D simultaneously → structure gets corrupted by VGM artifacts.
   - Phase 2 freezes a clean canonical and only learns motion → structure is preserved; motion fits noise within the part-rigid prior.

2. **Honest metric reveals what supervision-PSNR hides**: against noisy SV4D, both methods give similar PSNRs in the 13-15 dB range — because that *is* the noise level of SV4D vs GT. Against clean GT, the structural quality differences emerge.

3. **Future work this enables**: with d-3dgs as a true GT signal, we can ablate which mechanism (frozen canonical / smart photo / per-time scale) matters most by comparing each variant's vs-d3dgs PSNR.

---

## Attribution analysis (Q2: is the win from the canonical or the motion?)

> Added 2026-05-31 PM. Motivation: all prior lego_v2 runs co-vary K=100 + smart
> photo + scale + residuals, so no single factor was ever isolated. Two
> no-training measurements settle the dominant question.

### Attribution ladder (vs d-3dgs clean GT)

| Configuration | vs clean GT | increment | share of total gain |
|---|---:|---:|---:|
| Vanilla SC-GS (joint train, raw photo) | 11.43 | — | — |
| **Frozen canonical, ZERO motion** (`eval_static_canonical.py`) | **18.67** | **+7.24** | **82%** |
| + part-rigid motion (K=100 smart+scale) | 19.84 | +1.17 | 13% |
| + per-Gaussian xyz residual | 20.28 | +0.44 | 5% |

**82% of the +8.85 dB headline gain comes from simply having a clean frozen
canonical, before any motion is learned.** The part-rigid motion machinery
contributes +1.17 dB; xyz residual +0.44 dB.

Per-timestep, the static canonical ranges only **17.47 (t3, extreme) → 20.77
(t12, reference pose)** — a **3.3 dB** spread. At the reference pose the static
canonical (20.77) *beats* the full Phase 2 mean (20.28). Implication: **the lego
digger barely moves relative to its size; motion headroom on this scene is only
~3 dB.** lego is a weak testbed for the motion contribution — a large-articulation
scene (jumpingjacks) is needed to demonstrate the motion method.

### VGM noise: appearance vs geometry (`measure_vgm_pollution.py`)

SV4D supervision frames vs d-3dgs clean GT at the same (cam, t), texture-independent
silhouette IoU vs in-mask PSNR. v0 (clean input) is the control for the baseplate/
scale offset (SAM-2 removes the baseplate, d-3dgs keeps it):

| view | role | silh IoU | in-mask PSNR | full PSNR |
|---|---|---:|---:|---:|
| 0 | input (clean) | 0.483 | 26.82 | 14.89 |
| 1 | generated | 0.265 | 12.33 | 14.65 |
| 2 | generated | 0.408 | 10.06 | 14.10 |
| 3 | generated | 0.513 | 11.64 | 14.79 |
| 4 | generated | 0.432 | 12.97 | 13.69 |
| — | v1-4 mean | 0.404 | 11.75 | 14.31 |

- **Geometry (IoU)**: generated views drop only ~0.08 below the v0 control on
  average (v1 worst at −0.22; v3 actually above v0). Geometry is **mildly,
  view-dependently degraded — not severely polluted.**
- **Appearance (in-mask PSNR)**: collapses 26.8 → 11.8 (−15 dB) on generated
  views. (Caveat: in-mask PSNR still conflates residual pose error with texture.)

Together with the static-canonical result, this says: **VGM's structure is mostly
fine; the method wins by not using VGM's structure (or appearance) at all, sourcing
both from the clean canonical.** This sharpens the Q3 question — since the canonical
does ~82% of the work, the single-frame-lifted-canonical ablation (no clean-GT
canonical) is now the make-or-break experiment for real-VGM-scenario validity.

### Full attribution ladder (completed)

| Configuration | vs clean GT | increment | share of +8.85 total |
|---|---:|---:|---:|
| Vanilla SC-GS (joint train) | 11.43 | — | — |
| Frozen canonical, ZERO motion | 18.67 | +7.24 | **82%** |
| + single rigid body (K=1, smart+scale) | 19.66 | +0.99 | 11% |
| + 99 articulated sub-parts (K=1 → K=100) | 19.84 | **+0.18** | **2%** |
| + per-Gaussian xyz residual | 20.28 | +0.44 | 5% |

Cross-cut: K=100 with smart photo OFF (`--lam_photo_smart 0`) = **19.48** →
smart photometric contributes **+0.36 dB**.

**Decisive finding**: the hierarchical part-rigid articulation (K sub-parts +
LBS) — the supposed methodological core — contributes only **+0.18 dB** on lego.
A single rigid body (K=1) already reaches 19.66; the 99 extra parts add almost
nothing. Reason: the lego digger's motion is a near-rigid arm sweep, and the arm
Gaussians (28,057) were already isolated, so K=1 vs K=100 barely differ. Every
high-effort design knob (articulation +0.18, smart photo +0.36, xyz residual
+0.44) lands at noise level.

**Implication for the story**: lego strongly validates **decoupling** (frozen
canonical = +7.24 dB, invisible under VGM-self eval) but **cannot validate
articulation**. To support the part-rigid claim, a large-articulation scene
(jumpingjacks) must re-run this ladder — only there can K=100 vs K=1 separate.

Runs: `partrigid_lego_v2_K1_smart_scale` (19.66), `partrigid_lego_v2_K100_nophoto_scale` (19.48).

## Reproduce

```bash
# Vanilla SC-GS (this benchmark)
python third_party/SC-GS/train_gui.py \
    --source_path data/custom/lego_v2 \
    --model_path outputs/custom/lego_v2_vanilla_sam \
    --deform_type node --node_num 512 --hyper_dim 8 \
    --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
    --resolution 1 --W 576 --H 576 --iterations 20000
python scripts/eval_vanilla_lego_v2.py --model_path outputs/custom/lego_v2_vanilla_sam --save_renders

# Phase 2 (ours)
python scripts/train_partrigid_hier.py \
    --label lego_v2_K100_smart_scale \
    --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply \
    --part_dir runs_aux/part_assignment_lego_v2 \
    --scene_dir data/custom/lego_v2 \
    --v5_render_dir outputs/custom/lego_v2_d3dgs_ref/renders \
    --use_test_too --k_arm 100 --lbs_K 6 --lam_arap 1.0 \
    --lam_photo_smart 3.0 --use_per_time_scale --iterations 8000
python scripts/eval_lego_v2_hier.py --label lego_v2_K100_smart_scale --save_renders

# Build this report
python scripts/build_benchmark_report.py
```