# EV Final Project — Structure / Motion Decoupled 4D Gaussian Reconstruction from VGM Supervision

> **Final project**, NTU 113-2 EV, 2026-05-29
> Full report: [`docs/reports/2026-05-29_final_report.md`](docs/reports/2026-05-29_final_report.md)
> Slide outline: [`docs/planning/2026-05-29_final_slides.md`](docs/planning/2026-05-29_final_slides.md)

We study what happens when **SC-GS** (4D Gaussian splatting for monocular dynamic
scenes) is supervised by **Video Generative Model (VGM) output** — specifically
**SV4D 2.0**'s 5-view × 21-frame rendering of the D-NeRF *lego* scene — and
propose a method that handles the resulting noise.

---

## Headline visual

3-column gallery — **clean D-NeRF reference** vs **SV4D GT** (our training data)
vs **our part-rigid LBS render** at 5 viewpoints, t = 0:

![3-col contact sheet at t=0](runs_aux/gallery_3col_full/contact_sheet_t0.png)

Full 21-frame animation at view 0: [`runs_aux/gallery_3col_full/gallery_v0.gif`](runs_aux/gallery_3col_full/gallery_v0.gif)

---

## Results table

### Phase 1 — Within-framework extension (CVCG + frequency curriculum)

| Split | Method | PSNR | # Gaussians | 3D-consistency |
|---|---|---:|---:|---:|
| View-split | vanilla SC-GS | 13.10 | 96,309 | 47.71% |
| View-split | **+C3+CVCG (fast)** | **13.61** (+0.51) | **43,664** (½×) | **49.43%** (+1.72 pp) |
| Temporal-split | vanilla SC-GS | 25.75 | — | — |
| Temporal-split | **+C3+CVCG (slow)** | **27.33** (+1.58) | — | — |

### Phase 2 — Structure / motion decoupled (the headline contribution)

| Method | PSNR | DOF | Train time | Notes |
|---|---:|---:|---:|---|
| Static canonical (no motion) | 15.91 | 0 | — | baseline floor |
| Part-rigid v1 (hard ID) | 18.03 | 126 | 35 s | original |
| LBS (soft per-Gaussian weights) | 17.97 | 126 | 35 s | one-arm |
| Hier K=3 (3 sub-parts, ARAP) | 17.98 | 378 | 92 s | after LBS bug fix |
| Hier K=10 (no smart photo) | 17.14 | 1,260 | 132 s | over-fragments |
| Hier K=3 + smart photo (v5-filter) | 18.28 | 378 | 100 s | smart filter unlocks gain |
| Hier K=10 + smart photo 3× | 18.56 | 1,260 | 110 s | smart photo rescues K=10 |
| Hier K=50 + smart photo 3× | 18.82 | 6,300 | 210 s | |
| **Hier K=100 + smart photo 3× (final)** | **18.89** 🥇 | **12,600** | **347 s** | **+0.86 over baseline** |
| Vanilla SC-GS (16 M deform-MLP) | 25.75 | 16,000,000 | ~900 s | reference upper |

**Engineering finding**: a one-line LeakyReLU patch to SC-GS's deform-MLP
recovers training on sparse-time multi-view data (PSNR 17.39 → 31.79).

### Diagnostic — VGM cross-render to clean reference (§5.5)

| Fix | PSNR ceiling | Δ |
|---|---:|---:|
| Naive matched-fid | 12.19 dB | — |
| + baseplate Gaussian removal | 12.35 | +0.16 |
| + per-frame shift correction | 12.12 | −0.22 |

→ Clean D-NeRF 4D-GS rendered at our SV4D cameras **cannot serve as quantitative
GT**. SV4D rewrites both the world transform and the animation timing
(monotonic D-NeRF arc vs V-shape SV4D cycle).

---

## Key visualizations (all embedded inline)

### Per-view VGM artifact heatmap (§3)

![VGM per-view residual heatmap](runs_aux/vgm_artifact/heatmap.png)

View 1 is systematically 2× harder to fit. All 9 worst-residual cells originate
from view 1. Boundary-localized.

### Part assignment after LBS (§5.1)

![Canonical Gaussians colored by LBS weight (red=arm, blue=body, purple=boundary)](runs_aux/part_assignment_anim/canonical_part_assignment_contact_sheet.png)

5 views at t=0. **Red = arm (w > 0.9)**, **blue = body (w < 0.1)**, **purple =
LBS boundary** (29,917 of 54,475 Gaussians sit in boundary regime — only 33
fully snap to arm-rigid).

### Clean-ref temporal alignment (§5.5)

![Matched fid curves + PSNR ceiling](runs_aux/alignment_A/matched_curves.png)

Left: matched fid is **V-shaped** (1.0 → 0.4 → 1.0) — confirms SV4D animation
runs cyclically opposite to D-NeRF's monotonic arc.
Right: best-match PSNR plateaus at ~12 dB across all views.

### Static vs full-FG PSNR (§5.5)

![Static-region PSNR vs full-FG](runs_aux/static_region_C/static_vs_full_bar.png)

Restricting evaluation to the static body region (cabin + treads, masked from
temporal-variance heatmap) lifts our part-rigid PSNR by **+2.8 dB**
(13.28 → 16.06) — confirming ~3 dB of "structural fuzziness" was actually
arm pose mismatch.

---

## Method (one diagram)

```
SV4D 5-view mp4
   ↓
[multiview_videos_to_dnerf]   →  D-NeRF format
[sam2_seg_multiview]          →  RGBA with SAM-2 mask
[split_train_test / temporal] →  splits
   ↓
Stage A: train static 3DGS on t=0 only       → frozen canonical (54,475 G, PSNR 39.4)
Stage B: per-pixel temporal-variance         → motion mask (~30% arm)
Stage C: 5-view voting + K-means             → per-Gaussian part / sub-part id
Stage D: 2D centroid → DLT triangulation     → 3D arm trajectory (T, 3), conf 0.94
Stage E: train SE(3) for K sub-parts         → 126–1260 DOF, NO raw RGB loss
         L = λ_silh · BCE+IoU
           + λ_traj · ||centroid - target||² · conf
           + λ_smooth · ||Δtrans||² + ||Δrot||²
           + λ_arap · ||trans_k - trans_neighbor||²
```

---

## Current progress (2026-05-29)

### ✅ Completed
- **Engineering**: dying-ReLU patch (`third_party/SC-GS/utils/time_utils.py:418`)
- **Phase 1**: CVCG + C3 wired into motionprior framework; full ablation (4 runs × 2 splits)
- **Phase 2 method**: Stages A–E end-to-end on scene00 (5 views × 21 frames)
- **8-variant photometric ablation** (refutes original "no-photometric" framing — bottleneck is capacity, not VGM noise)
- **Hierarchical part-rigid** (K=3 with proper ARAP matches single-arm baseline; K=10 still over-fragments)
- **Clean-ref cross-render diagnostic** (§5.5: V-shape + 12 dB ceiling + baseplate fix)
- **Visualizations**: 3-col gallery (105 frames), part-assignment animation, contact sheets
- **LBS deform bug fix**: canonical fallback for sub-unity weights (regression 14.78 → 17.98 dB)

### 🚧 In progress
- Per-cluster trajectory targets for K=10+ (further constrain sub-parts)
- DINOv2 / foundation-feature loss (replace photometric entirely)
- Per-time Gaussian scale (let canonical Gaussians stretch as arm rotates)

### ✅ Just completed
- **Smart photometric loss with v5-canonical-residual filter** (+0.30 to +1.42 dB depending on K)
- **K-scaling ablation** (K=1 → 100, diminishing returns from K=50)
- **LBS deform bug fix** — canonical fallback for sub-unity weights
- **Headline visual** ([`runs_aux/final_comparison/`](runs_aux/final_comparison/))

### ⏳ Deferred
- Part-rigid data leak fix (training on full scene00 vs eval on split_t test)
- 4D-SH appearance model beyond global color tint
- Learnable skinning weights

---

## Reproduce

All commands assume the conda env `scgs` is active
(`/home/cthsu/miniconda3/envs/scgs/bin/python`).

### Data pipeline (one-shot)

```bash
python scripts/multiview_videos_to_dnerf.py \
    --src_dir /mnt/HDD_1/cthsu/multiview_videos \
    --out_dir data/custom/scene00
python scripts/sam2_seg_multiview.py \
    --src_dir /mnt/HDD_1/cthsu/multiview_videos \
    --orig_scene_dir data/custom/scene00 \
    --out_dir data/custom/scene00_masked
python scripts/split_temporal.py \
    --src_scene_dir data/custom/scene00_masked \
    --out_dir data/custom/scene00_split_t
```

### Phase 1 (CVCG/C3, SC-GS)

```bash
MOTIONPRIOR_C3_BANDS=10 MOTIONPRIOR_CVCG_BETA0=1.0 \
    python third_party/SC-GS/train_gui.py \
    -s data/custom/scene00_split_t -m outputs/custom/scene00_split_t_node \
    --eval --is_blender --iterations 30000 ...
```

### Phase 2 (structure/motion decoupled)

```bash
# Stage A: frozen canonical
python scripts/build_frame0_subset.py
python third_party/SC-GS/train_gui.py \
    -s data/custom/scene00_frame0 -m outputs/custom/canonical_static_node \
    --eval --is_blender --iterations 5000

# Stage B-D: motion mask + part assignment + trajectory
python scripts/motion_parts.py
python scripts/motion_parts_temporal.py
python scripts/build_part_assignments_and_trajectory.py
python scripts/build_part_lbs_weights.py

# Stage E: part-rigid training (LBS variant)
python scripts/train_partrigid_lbs.py --label lbs_photo1 --iterations 5000

# Hierarchical (current best capacity expansion)
python scripts/train_partrigid_hier.py --label hier_K3_fixed \
    --k_arm 3 --lbs_K 2 --lam_arap 1.0 --iterations 5000
python scripts/eval_partrigid_hier.py --label hier_K3_fixed --save_renders
```

### Diagnostics + visualizations

```bash
python scripts/characterize_vgm_artifact.py        # §3 per-view residual
python scripts/gaussian_mv_consistency.py          # GT-free 3D-consistency
python scripts/render_part_assignment_animation.py # part-color animation
python scripts/build_3col_full_gallery.py          # clean | GT | ours

# Clean-ref cross-render diagnostic (§5.5)
python scripts/render_clean_ref_fine_grid.py --n_fid 100
python scripts/match_sv4d_to_clean_ref.py
python scripts/static_region_psnr.py
python scripts/diagnose_clean_ref_align.py
python scripts/render_clean_ref_fine_grid_nobase.py --z_min -0.15
```

---

## Repository layout

```
EV_Final_Project/
├── motionprior/
│   ├── losses/cross_view_consistency.py    # CVCG module (§4)
│   ├── integration/scgs_hook.py            # patched hook (CVCG + C3)
│   └── ...                                  # other framework pieces
├── scripts/
│   ├── multiview_videos_to_dnerf.py        # mp4 → D-NeRF format
│   ├── sam2_seg_multiview.py               # SAM-2 mask
│   ├── characterize_vgm_artifact.py        # §3 measurement
│   ├── build_part_*.py                     # Stages B–D
│   ├── train_partrigid*.py                 # Stage E (3 variants)
│   ├── eval_partrigid*.py                  # evaluators
│   ├── render_clean_ref_*.py               # §5.5 cross-render
│   ├── build_3col_full_gallery.py          # final visual
│   └── ...
├── third_party/SC-GS/                       # vendored + patched
│   └── utils/time_utils.py:418             # LeakyReLU patch
├── outputs/custom/                          # trained models (gitignored)
│   ├── scene00_v5_node/                    # fits-all v5 (PSNR 31.79)
│   ├── canonical_static_node/              # frozen canonical (PSNR 39.4)
│   ├── partrigid_lbs_photo1/               # baseline (17.97)
│   ├── partrigid_hier_K3_fixed/            # hier K=3 (17.98)
│   └── lego_clean_ref/                     # clean D-NeRF ref (PSNR 25.23)
├── runs_aux/                                # all figures + diagnostics (gitignored)
│   ├── vgm_artifact/                       # §3 measurement
│   ├── gallery_3col_full/                  # 3-col visual gallery
│   ├── alignment_A{,_nobase}/              # §5.5 temporal alignment
│   ├── clean_ref_aligned{,_nobase}/        # §5.5 spatial alignment
│   ├── static_region_C/                    # §5.5 static-region PSNR
│   └── part_assignment_anim/               # part-colored animation
├── tests/
│   ├── test_cross_view_consistency.py      # 11 tests, all pass
│   └── test_scgs_hook.py                   # 24 tests, all pass
└── docs/
    ├── reports/2026-05-29_final_report.md  # full report (9 sections)
    ├── planning/2026-05-29_final_slides.md # slide outline (20 slides)
    ├── design/motion_design.md             # method design doc
    └── ...
```

---

## Acknowledgements

- SC-GS (Sparse Controlled Gaussian Splatting) — Yi-Hua Huang et al., CVPR 2024
- SV4D 2.0 — Stability AI
- SAM 2 — Meta AI Research
- D-NeRF lego scene — Pumarola et al., CVPR 2021
- Method design feedback from `Gemini Pro` (4 capacity-expansion options)

## License

MIT — see [`LICENSE`](LICENSE).
