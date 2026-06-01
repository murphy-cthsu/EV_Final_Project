# EV Final Project — Structure / Motion Decoupled 4D Gaussian Reconstruction from VGM Supervision

> NTU 113-2 EV Final Project · last updated 2026-06-01
> Latest PI checkpoint: [`meetTW_checkpoint_0601/checkpoint_0601.md`](meetTW_checkpoint_0601/checkpoint_0601.md)
> Method survey (all 24 attempts): [`docs/reports/2026-06-01_method_survey.md`](docs/reports/2026-06-01_method_survey.md)
> Fairness experiments: [`docs/reports/2026-06-01_fairness_experiments.md`](docs/reports/2026-06-01_fairness_experiments.md)

---

## 🏆 Headline (lego_v2, eval vs an independent d-3dgs clean GT)

| Method | PSNR vs clean GT | Δ |
|---|---:|---:|
| Vanilla SC-GS (16M deform-MLP, joint train) | 11.43 dB | — |
| **Ours (Phase 2, A1)** | **20.40 dB** | **+8.97** |
| Architecture ceiling (ours trained on clean GT) | 20.96 dB | +0.56 over ours |

→ We are at **96 % of the architectural ceiling** under noisy VGM supervision.

---

## 1. Core mechanism (read this first)

The problem: when SC-GS is supervised with **SV4D 2.0** multi-view video (a video generative model's output), joint structure + motion training breaks — the Gaussians absorb VGM hallucination noise into both, producing black-spike "explosions" at silhouette boundaries.

Our fix in one sentence: **freeze a clean canonical 3DGS and only learn motion on top of it, with a per-pixel "smart photometric" loss that automatically excludes VGM-hallucinated pixels.**

The full method is a 5-stage decoupled pipeline:

```
Monocular video ── SV4D 2.0 ──► Multi-view video {I_{v,t}}
                                          │
Clean canonical 3DGS ────────────────────►│ Stage A  freeze (xyz, scale, rot, SH, opacity)
(from D-NeRF / static scan)               │
                                          │ Stage B  motion mask per (v,t)
                                          │        (temporal variance + Otsu)
                                          │
                                          │ Stage C  per-Gaussian part assignment
                                          │        (multi-view voting → arm/body)
                                          │
                                          │ Stage D  3-D arm trajectory
                                          │        (DLT triangulation of 2-D centroids)
                                          │
                                          │ Stage E  train motion module
                                          │        K=100 cluster SE(3)  +  LBS (K_lbs=6)
                                          │        + per-time per-cluster 3D-scale
                                          │        + per-Gaussian XYZ residual
                                          │        loss = silh + smart-photo + ARAP + smooth
                                          ▼
                                  Deformed 3DGS  G(t),  renderable from any view
```

Key design choices and why they exist:

| Choice | Why |
|---|---|
| **Freeze canonical structure** (Stage A) | Joint training absorbs noise → explosion. Empirically tested with `_scale/_rot/_features` fine-tune (lr 1e-4) → **−1.11 dB**. Even microscopic drift hurts. |
| **Part-rigid cluster SE(3) instead of free deform-MLP** | 885 K params (vs 16 M for SC-GS DeformModel) but **+8.51 dB** higher PSNR on the same canonical (`F2` fairness experiment). The inductive bias of structured cluster motion is doing the work, not the DOF. |
| **Smart photometric filter**: `w = exp(-α · |I_sv4d − I_clean_ref|)` | Per-pixel VGM-noise confidence weighting. Pixels where SV4D disagrees with a known-clean reference get near-zero weight. Sharpness sweep: α=16 wins (20.40); α=8 baseline (20.28); α=4 worse (20.15). |
| **LBS with canonical fallback for sub-unity weights** | Boundary Gaussians with `Σw < 1` keep their canonical position rather than collapsing. Bug fix yielded +3.20 dB. |
| **8 k iterations, not more** | Loss plateaus by 8 k; 24 k overfits noise (**−0.34 dB**). |

For the full design rationale (24 variants tried, all winners and all failures with one-line "why"), read [`docs/reports/2026-06-01_method_survey.md`](docs/reports/2026-06-01_method_survey.md).

---

## 2. Environment

```bash
# conda env (assumes already created — full bootstrap below)
conda activate scgs   # /home/cthsu/miniconda3/envs/scgs/bin/python

# vendored third-party (one-shot)
bash scripts/bootstrap_third_party.sh

# SAM-2 mask generation uses a separate env that has hydra+sam2
conda activate motionprior
```

Hardware: any single GPU with ≥ 16 GB (we use 1× RTX 3090). One full training run is ~4 min for lego_v2 / ~9 min for the 57-view datasets at 8 k iterations.

---

## 3. Data layout

```
data/custom/
├── lego_v2/                       # 5 views × 21 frames (sparse, elev≈0°)
│   ├── transforms_train.json
│   ├── transforms_test.json
│   └── {train,test}/r_{flat:05d}.png   # flat = v_idx × 21 + t
├── lego_v3/                       # 57 views × 21 frames (7 elev × 9 azim, elev 0–30°)
├── lego_v3_elev0/                 # 9 views × 21 frames (elev=0 subset of v3)
├── hellwarrior/                   # 57 views × 21 frames
└── hellwarrior_elev0/             # 9 views × 21 frames

outputs/custom/                    # all training outputs (gitignored)
├── lego_v2_canonical/             # frozen canonical 3DGS (114 k Gaussians)
├── lego_v2_d3dgs_ref/renders/     # clean d-3dgs reference renders (eval-only)
├── lego_v3_d3dgs_ref/renders/
├── hellwarrior_d3dgs_ref/renders/
└── partrigid_<label>/             # one dir per training run

runs_aux/                          # diagnostics, figures, part-assignment data (gitignored)
├── part_assignment_<dataset>/     # Stage C output (Gaussian → arm/body id)
├── parts_motion_<dataset>/        # Stage B intermediate (motion masks)
├── lego_v2_eval/<label>/          # eval output (PSNR JSON, optionally tile renders)
└── method_animations/             # GIFs for reports
```

---

## 4. Reproduce the headline (lego_v2, ~5 min on one GPU)

We assume the lego_v2 dataset, the canonical 3DGS, and d-3dgs reference renders are already on disk (they are on the lab machine). If you need to re-create them, see §6.

```bash
# (1) Stage B + C + D — motion mask, part assignment, arm trajectory.
#     Reads SV4D mp4 + canonical PLY; writes runs_aux/part_assignment_lego_v2/
python scripts/motion_parts_generic.py \
    --dataset lego_v2 \
    --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply

# (2) Stage E — train motion module (A1 best config).  ~4 min.
python scripts/train_partrigid_hier.py \
    --label lego_v2_A1 \
    --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply \
    --part_dir runs_aux/part_assignment_lego_v2 \
    --scene_dir data/custom/lego_v2 \
    --v5_render_dir outputs/custom/lego_v2_d3dgs_ref/renders \
    --use_test_too --k_arm 100 --lbs_K 6 --lam_arap 1.0 \
    --lam_photo_smart 3.0 --photo_smart_alpha 16.0 \
    --use_per_time_scale --use_xyz_residual --iterations 8000

# (3) Eval — PSNR vs SV4D + vs clean d-3dgs GT.  Add --save_renders for view-0 tiles.
python scripts/eval_lego_v2_hier.py \
    --label lego_v2_A1 --scene lego_v2 \
    --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply
```

Expected output of (3):

```
[eval-v2] vs SV4D (supervision data) : mean=??.??  median=??.??
[eval-v2] vs d-3dgs (CLEAN GT)        : mean=20.40  median=20.??
[eval-v2] gap (d3dgs - sv4d): +?.??? dB
  → d-3dgs higher means we're predicting closer to clean GT than to noisy training data
```

For the 57-view extensions, see §5.

---

## 5. Run on the 57-view datasets (lego_v3 / hellwarrior)

These datasets stress test generality. Latest finding (2026-06-01): canonical–view alignment matters more than view count — using only the 9 elev=0° views beats all 57 views by **+3.85 dB** on lego_v3. See §3.7 of the [PI checkpoint](meetTW_checkpoint_0601/checkpoint_0601.md).

### Convert SV4D output → D-NeRF format (one-shot)

```bash
# Reads /mnt/HDD_1/cthsu/<dataset>/ (mp4 + transforms_sv4d2_math.json) and writes data/custom/<dataset>/
python scripts/lego_v3_hellwarrior_to_dnerf.py --dataset lego_v3
python scripts/lego_v3_hellwarrior_to_dnerf.py --dataset hellwarrior

# Extract clean d-3dgs reference renders (eval GT)
python scripts/extract_d3dgs_renders_v3.py --dataset lego_v3
python scripts/extract_d3dgs_renders_v3.py --dataset hellwarrior
```

### Build the elev=0 subset (the current best variant)

```bash
# Symlink elev=0 views (view_idx 0..8 are the elev=0 row in our flat indexing)
python scripts/build_scene_dataset.py \
    --src data/custom/lego_v3 --dst data/custom/lego_v3_elev0 --views 0,1,2,3,4,5,6,7,8

python scripts/build_scene_dataset.py \
    --src data/custom/hellwarrior --dst data/custom/hellwarrior_elev0 --views 0,1,2,3,4,5,6,7,8

# Symlink the d-3dgs reference so eval can find it
ln -sfn $(realpath outputs/custom/lego_v3_d3dgs_ref) outputs/custom/lego_v3_elev0_d3dgs_ref
ln -sfn $(realpath outputs/custom/hellwarrior_d3dgs_ref) outputs/custom/hellwarrior_elev0_d3dgs_ref
```

### SAM-2 mask refinement (optional but helps silhouette quality)

```bash
# Needs the motionprior env (has hydra + sam2)
conda activate motionprior
python scripts/sam2_mask_legov3.py --dataset lego_v3_elev0
conda activate scgs
```

### Train + eval

```bash
# Same A1 config as lego_v2, just point at the new dataset
python scripts/motion_parts_generic.py \
    --dataset lego_v3_elev0 \
    --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply

python scripts/train_partrigid_hier.py \
    --label lego_v3_elev0_A1 \
    --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply \
    --part_dir runs_aux/part_assignment_lego_v3_elev0 \
    --scene_dir data/custom/lego_v3_elev0 \
    --v5_render_dir outputs/custom/lego_v3_d3dgs_ref/renders \
    --use_test_too --k_arm 100 --lbs_K 6 --lam_arap 1.0 \
    --lam_photo_smart 3.0 --photo_smart_alpha 16.0 \
    --use_per_time_scale --use_xyz_residual --iterations 8000

python scripts/eval_lego_v2_hier.py \
    --label lego_v3_elev0_A1 --scene lego_v3_elev0 \
    --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply \
    --save_renders
```

Latest numbers:

| Dataset | full 57-view | **elev=0 (9 views)** |
|---|---:|---:|
| lego_v3 | 15.82 dB | **19.67 dB** (+3.85) |
| hellwarrior | 15.48 dB | 15.04 dB (−0.44, canonical-quality limited) |

---

## 6. Re-creating Stage A canonical from scratch (only if needed)

The lego canonical comes from D-NeRF clean training; it's already on disk as
`outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply`. To
retrain for a new scene:

```bash
# Frame 0 subset
python scripts/build_frame0_subset.py --dataset hellwarrior --out_dir data/custom/hellwarrior_frame0

# Train static 3DGS on t=0 only
python third_party/SC-GS/train_gui.py \
    -s data/custom/hellwarrior_frame0 \
    -m outputs/custom/hellwarrior_canonical_node \
    --eval --is_blender --iterations 20000

# Optional: hull prune to remove floaters
python scripts/prune_canonical_hull.py \
    --in_ply outputs/custom/hellwarrior_canonical_node/point_cloud/iteration_20000/point_cloud.ply \
    --out_ply outputs/custom/hellwarrior_canonical_pruned/point_cloud/iteration_20000/point_cloud.ply
```

Warning: canonical quality is the binding constraint. The hellwarrior canonical trained from SV4D-only (no clean source) regressed by −4.92 dB vs the D-NeRF-derived lego canonical. **If you don't have a clean source for the canonical, expect lower numbers**, and that's the open research question.

---

## 7. Script cheat sheet

| Script | What it does |
|---|---|
| `lego_v3_hellwarrior_to_dnerf.py` | SV4D mp4 + transforms_sv4d2_math.json → D-NeRF format (57-view variant) |
| `lego_v2_to_dnerf.py` | Same as above but for the original 5-view lego_v2 layout |
| `build_scene_dataset.py` | Subset a dataset by view list (used for elev=0 filter) |
| `extract_d3dgs_renders_v3.py` | Pull d-3dgs reference video frames out of SV4D mp4s |
| `sam2_mask_legov3.py` / `sam2_mask_lego_v2.py` | Refine alpha with SAM-2 video predictor |
| `motion_parts_generic.py` | Stages B + C + D (motion mask, voting, DLT trajectory) for any dataset |
| `train_partrigid_hier.py` | Stage E — the main training script |
| `train_scgs_deform_frozen_canon.py` | F2 fairness baseline: SC-GS deform-MLP on top of our frozen canonical |
| `eval_lego_v2_hier.py` | Eval against SV4D supervision **and** independent d-3dgs clean GT |
| `eval_multi_metric.py` / `eval_motion_metrics.py` | LPIPS / DINOv2 / FG-IoU / motion-region IoU |
| `eval_static_canonical.py` | Sanity: render the frozen canonical at our cameras (no motion) |
| `measure_vgm_pollution.py` | Quantify how much of SV4D's output is hallucinated noise |
| `build_method_animations.py` | Per-view 21-frame GIFs (vanilla vs ours, 3-col) |
| `build_3col_gallery.py` | 3-col contact sheet + GIF (SV4D \| d-3dgs \| ours) |

---

## 8. Latest results table (lego_v2)

| Method | PSNR vs clean GT | DOF | Notes |
|---|---:|---:|---|
| Vanilla SC-GS (random init) | 11.43 | 16 M | broken (Gaussian explosion) |
| F1: vanilla warm-start (clean init, not frozen) | 11.55 | 16 M | clean init alone doesn't help |
| F2: SC-GS DeformModel + frozen canon | 11.89 | 16 M | shows DOF isn't the bottleneck |
| Static canonical only (no motion) | 15.91 | 0 | baseline floor |
| Part-rigid hard ID, K=1 arm | 18.03 | 126 | "single rigid arm" |
| LBS K=3 + ARAP | 17.98 | 378 | |
| K=10 hier (no smart photo) | 17.14 | 1 260 | over-fragments without per-pixel signal |
| K=10 hier + smart photo 3× | 18.56 | 1 260 | smart photo rescues K=10 (+1.42) |
| K=100 hier + smart photo + per-time scale | 19.11 | 18 900 | |
| **K=100 hier + smart α=16 + per-time scale + xyz residual (A1)** | **20.40** 🥇 | 885 K | **headline** |
| Path 1: new t=0 canonical from d-3dgs + SAM | 14.07 | — | **−6.33, FAILED** — lost baseplate |
| Path 2: mild canonical fine-tune (lr 1e-4) | 19.29 | — | **−1.11, FAILED** — drift |
| Architecture ceiling (A1 trained on clean GT) | 20.96 | 885 K | hard upper bound |

Multi-metric agreement (ours vs vanilla, 105 frames):

| Metric | Vanilla | Ours | Δ |
|---|---:|---:|---:|
| PSNR ↑ | 11.43 | **20.40** | **+8.97** |
| LPIPS-alex ↓ | 0.412 | **0.230** | −0.18 |
| FG-IoU ↑ | 0.383 | **0.762** | +0.38 |
| Motion-region IoU ↑ | 0.234 | **0.528** | +0.29 |
| Motion magnitude correlation ↑ | 0.382 | **0.793** | +0.41 |

---

## 9. Repository layout

```
EV_Final_Project/
├── motionprior/                          # CVCG + curriculum framework (Phase 1, vendored)
├── scripts/                              # the runnable pipeline (see §7)
├── third_party/                          # SC-GS + SAM-2 vendored (bootstrap script)
├── data/custom/                          # datasets (gitignored)
├── outputs/custom/                       # trained models + d-3dgs renders (gitignored)
├── runs_aux/                             # part assignment, eval, figures (gitignored)
├── docs/
│   ├── reports/2026-06-01_method_survey.md         # 24-method survey
│   ├── reports/2026-06-01_fairness_experiments.md  # F1 + F2
│   ├── reports/2026-06-01_path1_path2_postmortem.md
│   └── reports/2026-06-01_checkpoint2_final.md
├── meetTW_checkpoint_0601/               # PI checkpoint (gitignored — local-only)
└── README.md
```

---

## 10. Acknowledgements

- SC-GS — Yi-Hua Huang et al., CVPR 2024
- Deformable-3DGS — Yang et al., CVPR 2024
- SV4D 2.0 — Stability AI
- SAM 2 — Meta AI Research
- D-NeRF lego scene — Pumarola et al., CVPR 2021

## License

MIT — see [`LICENSE`](LICENSE).
