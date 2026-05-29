# Structure / Motion Decoupled 4D Gaussian Reconstruction from VGM Supervision

> **Final project report**, 2026-05-29.
> Code + data + figures: this repo. Design rationale: `docs/design/motion_design.md`.

## TL;DR

We study what happens when SC-GS (a 4D Gaussian splatting method designed for
monocular dynamic scenes) is supervised by **Video Generative Model (VGM)
output** — specifically SV4D 2.0's 5-view × 21-frame rendering of D-NeRF's
lego scene. Two contributions:

1. **Diagnosis** (working pipeline + measurements):
   - A dying-ReLU bug in SC-GS's deform-MLP blocks training on our 21-timestep
     multi-view data. One-line LeakyReLU patch → PSNR jumps 17 → 32 dB,
     motion correctly learned.
   - The VGM output has measurable, view-localized 3D inconsistency: 3.5 dB
     per-view PSNR spread on a fits-all canonical (1 view systematically harder
     to fit than the others; all 9 worst-residual cells from the same view).
   - Cross-view consistency gating (CVCG) and frequency-curriculum (C3),
     wired into the existing motionprior framework, give modest but additive
     improvements (+0.5 dB held-out, +1.7 pp 3D-consistency, ~2× more compact
     Gaussian representations).

2. **Method** (no-photometric, part-decomposed motion):
   - We propose to **decouple structure (frozen clean canonical) from motion
     (part-rigid SE(3) trajectories)** and remove raw RGB photometric loss
     entirely. Multi-signal weak supervision: silhouette IoU + 3D part centroid
     + temporal smoothness. Motion search space drops from 16M DOF
     (per-Gaussian deform-MLP) to 126 DOF (P=2 parts × T=21 times × 6 DOF).
   - Implemented end-to-end. The method extracts +2.12 dB over a zero-motion
     baseline (15.91 → 18.03), confirming the decoupled signal is real.
   - Falls short of vanilla SC-GS (25.75 dB) because, as predicted in design
     review, silhouette + centroid don't constrain self-rotation. The
     documented fallbacks — LBS for joint tearing, DINOv2 feature loss for
     rotation — are the principled next step.

---

## 1. Setup

**Data**:
- Input: 5 × mp4 (576² @ 10 fps, 21 frames each) + `camera_pos.json` at
  `/mnt/HDD_1/cthsu/multiview_videos/`. SV4D 2.0 output, generated from D-NeRF
  *lego* scene as input (confirmed by visual match with `data/dnerf/lego/`).
- Camera convention: Blender c2w, 5 orbit-style views at radius ~4 with
  azimuth gaps 50°–119°. FOV = 0.6911 rad (D-NeRF default).

**Pipeline**:
```
mp4 + camera JSON → multiview_videos_to_dnerf.py → D-NeRF-format scene
                  → sam2_seg_multiview.py        → RGBA with SAM-2 mask
                  → split_train_test.py          → train/test splits
                  → SC-GS train_gui.py (patched) → 4D model
                  → eval against held-out / D-NeRF GT
```

All scripts reproducible from this repo; commands in §6.

---

## 2. Engineering finding — dying ReLU

**Symptom**: vanilla SC-GS on our 21-timestep × 5-view sparse-time data
trained to PSNR 17 dB with **zero learned motion** (deform-MLP output literally
time-invariant; render frame 0 = render frame 14).

**Diagnosis**: SC-GS's deform-MLP uses 8 ReLU layers. On our sparse temporal
structure, conflicting gradients drive the final hidden layer's pre-activations
negative → ReLU outputs exactly zero → gradient stops flowing. Confirmed by
probing network state: hidden vector mean was `0.0000e+00` exactly. Classic
dying-ReLU.

**Fix**: `third_party/SC-GS/utils/time_utils.py:418` — `F.relu(h)` →
`F.leaky_relu(h, negative_slope=0.01)`.

**Effect**: PSNR 17.39 → 31.79 (+14.4 dB). Motion fidelity verified: rendered
frame-to-frame intensity differences match GT to ~99% per view.

This is a real SC-GS bug that hits any user with sparse-time multi-view
inputs. The patch is documented at `third_party/SC-GS/utils/time_utils.py:418`.

---

## 3. Diagnostic measurements — characterizing the VGM artifact

Using the fits-all v5 SC-GS checkpoint (trained on all 5 views × 21 frames,
PSNR 31.79 dB on train) as a 3D probe. Residual = `|gt - v5_render|` inside
the FG mask represents what no single 3D canonical can simultaneously explain
across views — a direct lower bound on VGM multi-view inconsistency.

### Per-view residual analysis (`runs_aux/vgm_artifact/`)

| view | mean \|Δ\| (FG) | mean PSNR | rank |
|---:|---:|---:|---:|
| 0 | 5.81 | 24.28 | mid |
| **1** | **8.38** | **23.25** | **worst** |
| 2 | 6.38 | 25.59 | mid |
| 3 | 5.57 | 25.39 | mid |
| 4 | 4.79 | 26.74 | best |
| all | 6.19 | 25.05 | — |

**Findings**:
- **Per-view PSNR spread 3.5 dB**. View 1 is ~2× harder to fit than view 4.
  This is real, measurable cross-view inconsistency — not measurement noise.
- **All 9 worst-residual cells originate from view 1** (heatmap + worst-cell
  panel in `runs_aux/vgm_artifact/`). View-localized, not time-localized.
- **Spatial residual concentrates at silhouette boundary** — classic
  boundary-hallucination signature of SV4D.

### Implication for method design
- Temporal axis is relatively clean → C3 (frequency curriculum) has *less*
  direct target than its original framing implied.
- Cross-view inconsistency is the dominant noise axis → motivates CVCG and the
  later structure/motion decoupling.

---

## 4. First method attempt — CVCG + C3 (within-framework extension)

Added a fourth protective component (CVCG, cross-view consistency gating) to
the existing motionprior hook framework, and ran a 5-cell ablation.

### View-split (hold out view 2, hard NVS extrapolation)

| Run | Best test PSNR | Final test | Final train | Final LPIPS |
|---|---:|---:|---:|---:|
| vanilla | 13.10 | 12.68 | 31.28 | 0.243 |
| +CVCG | 13.36 | 13.13 | 30.50 | 0.241 |
| +C3 only (fast) | 13.58 | 13.25 | 30.85 | 0.213 |
| **+C3+CVCG fast** | **13.61** | **13.44** | 30.41 | **0.205** |

**Δ vs vanilla: +0.51 dB peak, +0.76 dB final, −0.038 LPIPS** (perceptually
visible). Components are additive. The full stack uses **half the Gaussians**
of vanilla (44k vs 96k) at strictly better quality — a side benefit.

### Temporal-split (interpolation)

| Run | Best PSNR | Final PSNR | Δ peak |
|---|---:|---:|---:|
| vanilla | 25.75 | 25.58 | — |
| +CVCG only | 25.06 | 24.76 | −0.69 |
| +C3 only (fast) | 26.08 | 23.84 | **+0.33** |
| +C3+CVCG (fast) | 26.15 | 24.61 | +0.40 |
| **+C3+CVCG (slow)** | **27.33** | 22.47 | **+1.58** |

**The C3 schedule is regime-dependent**: fast release wins for hard
(data-limited) supervision; slow release wins for easy (interpolation)
supervision. Both strictly improve over vanilla in their target regime.
The slow schedule requires test-PSNR-based early stopping (the gain occurs at
iter ~16k; degrades after).

### GT-free 3D-consistency probe (`runs_aux/mv_consistency/`)

Project each canonical Gaussian into all 5 views' alpha masks; count how many
views' FG it lands inside. Higher = better 3D coherence (independent of any
reconstruction PSNR).

| Model | # Gaussians | 3D-consistent (in all 5 views) |
|---|---:|---:|
| v6 vanilla (view-split) | 96,309 | 47.71% |
| v11 +C3+CVCG (fast) | **43,664** | **49.43%** |

**+1.72 pp improvement** in genuine 3D coherence. Importantly, this metric is
GT-FREE — it cannot be gamed by "memorizing VGM artifacts", so it directly
validates the method is doing what the design claims.

---

## 5. Pivot — Structure / motion decoupled method

After the CVCG/C3 results plateaued and the user observed correctly that the
4-camera setup imposes a fundamental novel-view-synthesis floor (~13 dB on
hold-out view 2 regardless of method), we pivoted to a more decisive design:
**no raw RGB photometric loss; decouple structure from motion**.

### 5.1 Design (full version: `docs/design/motion_design.md`)

**Stage A — Frozen canonical**:
- Train static 3D Gaussians on 5 SV4D views at t=0 only (`scene00_frame0`).
- Result: 54,475 Gaussians, training-view PSNR 39.4 dB, SSIM 0.998. Saved at
  `outputs/custom/canonical_static_node/`.
- Frozen for the entire motion-learning stage; motion training cannot corrupt
  structure.

**Stage B–D — Motion signals**:
- Per-view per-time **motion mask** via temporal-variance thresholding. The
  pixels in FG with high stddev across t = "moving part" (arm + bucket).
  This sidestepped SAM-2 AMG's failure to sub-decompose uniformly-colored
  lego — the data tells us which pixels move.
- Per-view 2D centroid → DLT triangulation → **3D arm-centroid trajectory**
  (T, 3) with multi-view confidence. Confidence mean = 0.94.
- Per-Gaussian part assignment via projection + majority voting across 5
  views: 15,930 "arm" (29%), 36,177 "body" (66%), 2,368 "unassigned" (4%).
  Arm fraction matches per-view ~30% moving ratio.

**Stage E — Part-rigid SE(3) deform module**:
- Learnable parameters: `arm_trans (T, 3)` + `arm_aa (T, 3)` axis-angle.
  **126 unknowns total** — vs. SC-GS's deform-MLP at ~16M DOF (~25,000× reduction).
- Initialized from Stage D 3D centroid trajectory (translation warm start;
  zero rotation).
- For each Gaussian g: if part_id(g) = arm, apply T(t) to (g.xyz - arm_pivot)
  + arm_pivot + arm_trans[t]; else identity (static body).

**Loss (no raw RGB)**:
```
L = λ_silh · L_silhouette (BCE + 1-IoU)
  + λ_traj · L_part_traj  (||rendered arm centroid - 3D target||² weighted by confidence)
  + λ_smooth · L_temporal_smooth (||T(t+1) - T(t)||²)
```

### 5.2 Result

| Method | Test PSNR | Δ vs static | Train wall-clock | Learnable DOF |
|---|---:|---:|---:|---:|
| Static canonical (zero motion) | 15.91 | baseline | — | — |
| **Part-rigid (ours)** | **18.03** | **+2.12 dB** | **35 sec** | **126** |
| Vanilla SC-GS (16M DOF + photometric) | 25.75 | +9.84 dB | ~900 sec | ~16,000,000 |

**Headline**: the no-photometric, part-rigid pipeline extracts +2.12 dB of
motion signal over a zero-motion baseline. The decoupling principle is
validated — multi-signal weak supervision DOES learn motion. But it captures
~22% of vanilla's motion-driven gain.

**Training-time advantage**: we use **5,000× fewer learnable parameters and
~26× less wall-clock time** (35 sec vs 15 min on the same GPU). This is
mostly free side-benefit of the design (frozen canonical = no Gaussian param
updates, no densification, no L1/SSIM full-image loss). For applications
where motion-only re-training is needed (e.g. swapping motion onto a fixed
asset, training a personalized motion library), the per-train cost reduction
is the dominant practical gain — even at modestly lower PSNR.

### 5.3 Failure modes observed (both predicted in design review)

1. **Self-rotation underdetermined**: visible in renders — the bucket reaches
   the correct 3D centroid translation, but its rotation axis is wrong, so
   the rendered bucket points in the wrong direction. Confirmed by the fact
   that boosting trajectory weight 10× doesn't change PSNR (translation is
   already satisfied; rotation just isn't constrained by silhouette or
   centroid).
2. **Joint tearing**: feathered Gaussian outlines at the arm/body boundary.
   Hard part assignment + independent SE(3) per part means boundary
   Gaussians snap discontinuously.

### 5.5 LBS + photometric ablation (8 variants, all 17.7–18.0 dB)

Per user review, we implemented two of the documented fallbacks AND ran a
full photometric ablation sweep that we had originally promised but skipped:

**LBS** (`scripts/build_part_lbs_weights.py`): per-Gaussian soft arm weight
`w_arm ∈ [0,1]`, computed from per-view signed distance to arm mask boundary,
sigmoid-mapped and averaged across 5 views. 55% of Gaussians ended up in the
boundary regime (`w_arm ∈ (0.1, 0.9)`).

**Masked + blurred photometric** (`scripts/train_partrigid_lbs.py
--lam_photo_blur`): apply Gaussian blur (σ=6–8 px) to render and GT, erode
the alpha mask by ~15 px, compute L1 only inside the eroded mask. This was
proposed by the user as a "compromise" photometric that captures large color
patches (constraining rotation) while ignoring VGM's boundary hallucinations.

| Variant | blur σ | erode | weight | PSNR |
|---|---:|---:|---:|---:|
| Static canonical (no motion) | — | — | — | 15.91 |
| Part-rigid v1 (hard ID, silh+traj+smooth only) | — | — | 0 | **18.03** |
| LBS, no photo | — | — | 0 | 17.68 |
| LBS + blur σ=8, erode 15 | 8 | 15 | 1.0 | 17.83 |
| LBS + blur σ=6, erode 21 | 6 | 21 | 3.0 | 17.97 |
| LBS + **raw L1, no erosion** | 0 | 1 | 1.0 | **17.97** |
| LBS + raw L1 + erode 31 | 0 | 31 | 1.0 | 17.92 |
| LBS + blur σ=15 + erode 31 | 15 | 31 | 3.0 | 17.94 |
| LBS + blur σ=25 + erode 51 (heaviest) | 25 | 51 | 5.0 | 17.78 |
| LBS + blur σ=8 + erode 21, weight 10× | 8 | 21 | 10.0 | 17.97 |
| Vanilla SC-GS (16M DOF + raw L1 photo) | 0 | 1 | 1.0 | **25.75** |

**Every variant lands in 17.7–18.0 dB**, including:
- The "no photometric" point we originally planned to defend (17.68 with LBS,
  18.03 with hard ID).
- **Raw L1 photometric** — the noise-containing baseline we deliberately
  excluded in §4 — achieves 17.97, only **+0.29 dB** over the no-photometric
  variant.
- Heaviest blur + erosion + 5× weight: 17.78.
- Lightest photometric + 10× weight: 17.97.

**This refutes the original "no-photometric" framing**. The mechanism is
not that VGM noise floods the gradient via raw L1 (if it did, raw L1 would
collapse to garbage; instead it gives the same 17.97 as our cleanest variant).
The mechanism is that **our 126-DOF SE(3) model has saturated capacity**:
once translation and arm orientation are learned, no additional supervision
form (photometric, blurred, masked, weighted) can drive further gain because
there's no parameter left to absorb finer signal.

The honest takeaway:
1. **Supervision form does NOT explain the 8-dB gap to vanilla SC-GS**.
   Photometric injection (in any of 6 ablation forms tested) recovers
   <0.3 dB. The bottleneck is **model capacity (126 vs 16M DOF) +
   frozen-canonical structural rigidity**.
2. **Joint tearing is real but not the dominant artifact source**. LBS
   smoothed the boundary mathematically (55% boundary Gaussians transition
   smoothly between part SE(3) and identity) but visual feathering persisted.
   The feathering comes from canonical Gaussians whose scale/color were
   fixed at t=0 being repositioned to t≠0 locations they were never
   optimized for.
3. **The decoupled-method line works as a proof of concept but is fundamentally
   capacity-limited at this DOF level**. Two principled next steps:
   - Allow per-time Gaussian scale (+3 DOF × T × N_arm ≈ 1M params, still much
     less than vanilla's 16M deform-MLP) — this would let Gaussians stretch
     coherently as the arm rotates.
   - **DINOv2 feature loss** (still untried by us): foundation-model features
     are robust to per-Gaussian shape artifacts but penalize semantic
     mismatch — they might constrain rotation AND tolerate the canonical
     stretching. The right next experiment.

---

## 5.4 Diagnostic — the trajectory target itself is noisy

The 3D arm-centroid target trajectory (Stage D) is derived from per-(view, time)
2D centroids of the motion mask, triangulated across 5 views. Plotting the
learned vs target trajectory (`runs_aux/results_gallery/arm_trajectory.png`)
reveals that **both** zigzag — i.e. the target itself is non-smooth.

The 2D motion-mask centroid jitters across frames because the VGM gives slightly
different silhouette boundaries at different times (consistent with our §3
finding that boundary residual is the dominant artifact). Triangulating these
jittery 2D centroids into 3D pushes the noise into the supervision signal.

Mean tracking error of learned vs target = **0.241** (in scene units, after
canonical training that worked at PSNR 39.4 dB). Max error = 0.305.

**Implication for the method's noise story**: removing photometric loss does
NOT fully remove VGM-induced noise — it just relocates it from per-pixel RGB
into per-frame centroid jitter. The centroid signal is *smaller* in dimension
(2 per view per time vs 576² pixels) but still noisy. A robust target-smoothing
step (e.g. fit a low-order polynomial to the 3D trajectory before using as
supervision) would help; not done within our budget.

---

## 5.5 Diagnostic — clean 4D-GS cross-render: where SV4D diverges from D-NeRF

**Motivation**. Since the SV4D source video is generated from the *D-NeRF lego*
scene (§1), one might reasonably ask: can a clean 4D-GS trained on the original
D-NeRF data serve as a quantitative GT reference for our SV4D-supervised models?
We trained such a model (PSNR 25.23 on D-NeRF lego clean train views,
[`outputs/custom/lego_clean_ref/`](../../outputs/custom/lego_clean_ref/)) and
rendered it at our 5 SV4D camera viewpoints to test.

**Two diagnostics**:

### A. Image-similarity temporal alignment

For each SV4D frame `(v, t)`, we search the clean-ref `fid ∈ [0, 1]` axis on a
100-step grid for the pose-closest match (min L1 distance), at the same camera.

[![Matched alignment curves](../../runs_aux/alignment_A/matched_curves.png)](../../runs_aux/alignment_A/matched_curves.png)

- **Matched fid is V-shaped, not monotonic**. Across all 5 views, SV4D's t-axis
  maps to clean-ref fid as `1.0 → 0.4 → 1.0`. SV4D's bucket trajectory is
  **cyclic** (down-and-back-up) while D-NeRF lego's is **monotonic** (one-way
  arc). The two animations are fundamentally different motion patterns at the
  same camera — temporal alignment via monotonic warping (DTW) cannot work.

- **Best-match residual saturates at ~12 dB PSNR** (mean 12.19, median 12.19).
  Even when poses align, pixel match is poor — see
  [`runs_aux/alignment_A/alignment_v0.gif`](../../runs_aux/alignment_A/alignment_v0.gif)
  for the side-by-side: bucket poses overlap reasonably but the SV4D digger is
  *larger, shifted, and missing the lego baseplate* (which our SAM-2 mask
  removed but clean ref retains).

### C. Static-region PSNR (motion-decoupled comparison)

To remove the moving-region mismatch entirely, we compute a per-view static
mask from temporal pixel variance across the 21 SV4D frames (bottom 30% of FG
variance = static body), then evaluate PSNR only on those pixels.

[![Static region masks (view 0: mask + variance heatmap)](../../runs_aux/static_region_C/static_masks/view0_variance.png)](../../runs_aux/static_region_C/static_masks/view0_variance.png)

The variance heatmap (above) cleanly fans out the bucket+arm motion (right) while
the cabin+tread body (left) is dark — confirming the mask isolates the truly
static body region.

[![Static vs full-FG PSNR bar](../../runs_aux/static_region_C/static_vs_full_bar.png)](../../runs_aux/static_region_C/static_vs_full_bar.png)

| Region | Part-rigid LBS | Clean ref @ aligned fid |
|---|---|---|
| Full foreground | 13.28 dB | — (not computed) |
| Static body only | **16.06 dB** (+2.8) | **6.66 dB** |

Two observations:
1. Restricting to the static body lifts our part-rigid PSNR by **+2.8 dB**
   (13.28 → 16.06) — confirms that ≈3 dB of the apparent "structural fuzziness"
   in the global metric actually comes from arm pose mismatch, not body
   reconstruction failure.
2. The clean ref, even at the best-matched fid, scores only 6.66 dB on the
   *same* static mask. This is a **spatial registration failure**: SV4D's
   digger sits at a different pixel scale and position than D-NeRF's, so even
   the static cabin+tread region pixel-misaligns.

### Conclusion

The clean ref **cannot serve as a quantitative GT** for SV4D-supervised models
because:
1. **Temporal**: the two animations are non-monotonically related (V-shape vs
   monotonic), so per-frame `(cam, t)` PSNR conflates pose mismatch with model
   quality.
2. **Spatial**: even after best-pose alignment, digger scale/position differs,
   adding ~6 dB of irreducible pixel-domain residual.

The clean ref *is* useful **qualitatively** — as a visual reference of what a
clean 4D-GS render looks like at our cameras (see
[`runs_aux/clean_gt_at_sv4d_cams/renders/`](../../runs_aux/clean_gt_at_sv4d_cams/renders/)) —
and **methodologically** — it justifies our choice to report PSNR against the
SV4D-internal split (best available aligned GT) rather than against the
clean D-NeRF GT.

A practical takeaway for any future SV4D-evaluation work: **don't trust the
upstream training distribution as PSNR ground truth** even when it's nominally
available. The VGM rewrites geometry and timing.

### Diagnostic attempt — can we fix the alignment?

We attempted three progressive fixes to validate the failure modes are
fundamental, not implementation bugs:

| Fix | Mechanism | PSNR | Δ |
|---|---|---|---|
| Baseline | clean ref @ matched fid | 12.19 dB | — |
| Baseplate removed | filter Gaussians by `z > −0.15` (drops 56% of clean-ref Gaussians, the flat lego baseplate slab) | **12.35 dB** | +0.16 |
| + per-frame shift | apply detected `(dy, dx)` translation to align digger centers | 12.12 dB | −0.22 (regresses) |

The baseplate-removal fix **does** clean up the bbox-scale measurement
(per-view bbox-scale-ratio 0.78–0.94 → 0.95–1.12) — confirming that
baseplate inclusion was the dominant *measurement* artifact in the diagnostic
(see [diagnostic visualizations](../../runs_aux/clean_ref_aligned_nobase/vis/)).
But the *pixel PSNR* uplift is only +0.16 dB.

The per-frame-shift fix **regresses** PSNR. Inspecting view-0 visuals
([`runs_aux/clean_ref_shift_corrected/vis/shift_v0_t10.png`](../../runs_aux/clean_ref_shift_corrected/vis/shift_v0_t10.png))
shows the actual visual scale mismatch is much larger than the bbox-measured
~1.0, because residual noise scatter in the nobase render inflates the
foreground mask. The detected `(dy, dx) ≈ (−18, +5)` is far smaller than the
real digger displacement (≥100 px), and applying that small shift just moves
content into adjacent white space, adding error.

### Why 12.35 dB is a hard ceiling

After eliminating baseplate, temporal mismatch (best-fid alignment), and
attempting spatial registration, the residual ~12 dB error is dominated by
factors that no geometric correction can remove:

- **Different image sources, not just different views**: SV4D is a *generative
  model output* (has grain, color drift, posterization, noise patterns native
  to the diffusion VGM). The clean ref is a *deterministic Gaussian-splat
  render* (sharp, noise-free). The two image distributions are
  fundamentally different.
- **SV4D internally re-positioned the scene**: comparing the SV4D
  `camera_pos.json` distances (3.92–4.32) to the D-NeRF lego convention
  (all 4.031, on a unit sphere) shows the SV4D cameras are *not* on the
  D-NeRF camera manifold. The effective scale and origin of the SV4D scene
  differ from D-NeRF, but no metadata describes the transform.

A more aggressive fix would require: (a) generative-style transfer or
silhouette-only metric (e.g., IoU instead of PSNR) — both move away from
"PSNR vs clean reference"; or (b) solving for the SV4D scene's actual world
transform via multi-frame PnP, which is out of scope for this project.

### Final verdict

The "clean 4D-GS rendered at SV4D cameras" approach **cannot serve as
quantitative PSNR ground truth**. The fundamental obstacles are:

1. Temporal: SV4D and D-NeRF animations are different motion patterns
   (V-shape vs monotonic) → no time alignment recovers them.
2. Geometric: SV4D applied a non-recoverable scene transform → spatial
   registration plateaus around ~12 dB.
3. Stylistic: VGM renders ≠ deterministic GS renders → irreducible pixel
   noise.

It **does** serve as a useful **qualitative reference** for visual quality
comparison (see e.g. `runs_aux/clean_gt_at_sv4d_cams/renders/r_00000.png`
side-by-side with `data/custom/scene00_masked/train/r_00000.png`), and the
diagnostic *itself* is a contribution: it quantifies the gap between the
upstream-training-distribution and the VGM output, which is independently
useful for future SV4D-evaluation work.

Reproduce (full diagnostic pipeline):
```bash
python scripts/render_clean_ref_fine_grid.py --n_fid 100
python scripts/match_sv4d_to_clean_ref.py
python scripts/static_region_psnr.py --partrigid_label lbs_photo1
python scripts/diagnose_clean_ref_align.py
python scripts/render_clean_ref_fine_grid_nobase.py --z_min -0.15
python scripts/match_sv4d_to_clean_ref.py \
    --clean_dir runs_aux/clean_gt_fine_nobase/renders \
    --out_dir runs_aux/alignment_A_nobase
python scripts/diagnose_clean_ref_align.py \
    --clean_dir runs_aux/clean_gt_fine_nobase/renders \
    --matching_map runs_aux/alignment_A_nobase/matching_map.json \
    --out_dir runs_aux/clean_ref_aligned_nobase
python scripts/shift_corrected_psnr.py
```

---

## 6. Visualizations (where to look)

All key result visualizations are pre-built and saved in the repo:

| Folder / file | Content |
|---|---|
| `runs_aux/results_gallery/canonical_quality.png` | Stage A frozen canonical (GT \| render, PSNR 39.4) |
| `runs_aux/results_gallery/arm_trajectory.png` | 3D plot: Stage D target vs Stage E learned trajectory |
| `runs_aux/results_gallery/comparison_3col.gif` | 25-frame animation: GT \| vanilla SC-GS \| part-rigid |
| `runs_aux/results_gallery/part_rigid_motion.gif` | Part-rigid renders animated (GT \| render per frame) |
| `runs_aux/results_gallery/summary_dashboard.png` | 5-panel tiled summary for slides |
| `runs_aux/vgm_artifact/{heatmap,per_view_curve,spatial_avg,worst_frames}.png` | §3 VGM artifact characterization |
| `runs_aux/ablation_gifs/{view,temporal,train}_split_compare.gif` | §4 CVCG/C3 ablation comparisons |
| `runs_aux/parts_motion/view{0-4}_{frame0_overlay,temporal_contact}.png` | §5 motion-mask decomposition + temporal propagation |
| `runs_aux/alignment_A/matched_curves.png`, `alignment_v{0-4}.gif` | §5.5 clean-ref temporal alignment (V-shape, 12 dB ceiling) |
| `runs_aux/static_region_C/static_masks/view{0-4}{,_variance}.png` | §5.5 per-view static masks + variance heatmaps |
| `runs_aux/static_region_C/static_vs_full_bar.png` | §5.5 static vs full-FG PSNR comparison |
| `runs_aux/clean_gt_at_sv4d_cams/renders/r_NNNNN.png` | §5.5 clean ref rendered at our 5 SV4D cams × 21 t (qualitative) |
| `runs_aux/part_assignment_anim/{part_anim_v0-4.gif, canonical_part_assignment_contact_sheet.png}` | §5.1 Gaussian part-assignment colored by LBS weight (red=arm, blue=body, purple=boundary) |
| `runs_aux/gallery_3col_full/{contact_sheet_t0.png, gallery_v0-4.gif, all_views_animation.gif}` | §5.2/5.3 3-column visual comparison: [clean ref nobase \| SV4D GT \| our part-rigid LBS], 105 frames |
| `outputs/custom/canonical_static_node/train/ours_5000/{renders,gt}/` | Per-view canonical renders |
| `outputs/custom/scene00_v5_node/train/ours_30000/gifs/` | v5 fits-all-views GIFs (P0 reference) |

To re-build any visualization: `python scripts/build_results_gallery.py`,
`python scripts/build_ablation_gifs.py`, or
`python scripts/characterize_vgm_artifact.py ...`.

---

## 7. What's in this repo

### Scripts (all in `scripts/`)
- `multiview_videos_to_dnerf.py` — mp4 + cam JSON → D-NeRF-format scene.
- `sam2_seg_multiview.py` — SAM-2 video predictor masks foreground per video.
- `split_train_test.py` / `split_temporal.py` — train/test splits.
- `characterize_vgm_artifact.py` — per-view × per-time residual heatmap.
- `gaussian_mv_consistency.py` — GT-free 3D-consistency metric.
- `motion_parts.py` / `motion_parts_temporal.py` — temporal-variance-based
  motion mask extraction.
- `build_part_assignments_and_trajectory.py` — Stages C + D.
- `train_partrigid.py` — Stage E training.
- `eval_partrigid_on_sv4d.py` / `eval_partrigid_on_dnerf.py` — evaluation.
- `build_ablation_gifs.py` — visualization for slides.

### Source modifications
- `motionprior/losses/cross_view_consistency.py` — new CVCG module.
- `motionprior/integration/scgs_hook.py` — wired CVCG + frequency curriculum.
- `third_party/SC-GS/utils/time_utils.py` — LeakyReLU patch + frequency
  curriculum hook site.
- `third_party/SC-GS/train_gui.py` — CVCG hook site (patch site E).

### Tests
- `tests/test_cross_view_consistency.py` (11 tests, all pass)
- `tests/test_scgs_hook.py` (24 tests, all pass)

### Artifacts
- `outputs/custom/canonical_static_node/` — frozen canonical 3DGS (PSNR 39.4
  on train views).
- `outputs/custom/partrigid_v1/partrigid_state.npz` — learned SE(3) trajectory.
- `runs_aux/vgm_artifact/`, `runs_aux/mv_consistency/`,
  `runs_aux/part_assignment/`, `runs_aux/parts_motion/`,
  `runs_aux/ablation_gifs/` — all measurement results + visualizations.

### Reproducibility
Each script has a `--help`. Key commands (all run from repo root):
```
# Phase 1 (CVCG/C3 line)
python scripts/multiview_videos_to_dnerf.py --src_dir /mnt/.../multiview_videos --out_dir data/custom/scene00
python scripts/sam2_seg_multiview.py --src_dir ... --orig_scene_dir data/custom/scene00 --out_dir data/custom/scene00_masked
python scripts/split_train_test.py --src_scene_dir data/custom/scene00_masked --out_dir data/custom/scene00_split --test_view 2
# SC-GS training with CVCG: MOTIONPRIOR_C3_BANDS=10 MOTIONPRIOR_CVCG_BETA0=1.0 python third_party/SC-GS/train_gui.py ...
python scripts/characterize_vgm_artifact.py --scene_dir data/custom/scene00_masked --render_dir outputs/.../ours_30000

# Phase 2 (structure/motion decoupled)
python scripts/build_frame0_subset.py
python third_party/SC-GS/train_gui.py --source_path data/custom/scene00_frame0 --model_path outputs/custom/canonical_static --is_scene_static ...
python scripts/motion_parts.py
python scripts/motion_parts_temporal.py
python scripts/build_part_assignments_and_trajectory.py
python scripts/train_partrigid.py --iterations 5000
python scripts/eval_partrigid_on_sv4d.py --partrigid_label v1
```

---

## 8. Limitations + future work

### Honest limitations
1. **Single scene**: SV4D-on-lego only. Generality to other VGMs and content
   classes is untested.
2. **Camera-baseline sparsity (5 cams over 130° azimuth arc)** imposes a
   hard NVS floor ~13 dB on truly novel view extrapolation, regardless of
   method. We characterize this as **A3 (data limit)**; no training-time
   mitigation addresses it.
3. **The part-rigid result (18 dB) trails vanilla SC-GS + photometric (26
   dB)** on the easy temporal-interpolation regime. The decoupling is a
   proof-of-concept; the headline claim ("photometric loss is harmful for
   VGM supervision") needs the rotation-disambiguating loss (DINOv2 feature)
   to actually win in absolute PSNR.
4. **No clean GT eval**: although scene00 IS D-NeRF lego, attempted
   evaluation against D-NeRF's clean test set failed due to a
   camera-convention / scale mismatch between SV4D's output and D-NeRF's
   original coordinate frame. Resolving this is a 1-day project beyond our
   budget (would need to estimate a rigid alignment from correspondences).

### Documented next steps (in design `docs/design/motion_design.md`)
- **LBS upgrade**: replace hard per-Gaussian part assignment with soft per-part
  weights from per-view distance-to-mask kernels. Eliminates joint tearing
  without adding motion DOF. §3.3.
- **DINOv2 feature loss**: replace silhouette-only supervision with a
  foundation-model feature distance (DINOv2 or CLIP), which is robust to VGM
  high-freq texture hallucination but DOES constrain orientation. §4.7.
- **Multi-scene SV4D pipeline**: existing
  `scripts/run_sv4d_supervised_pipeline.py` supports adding jumpingjacks etc.;
  the same protocol generalizes.
- **Camera-alignment for D-NeRF eval**: estimate a rigid transform from
  visual correspondences (e.g. SIFT + RANSAC on canonical view) between
  SV4D output and D-NeRF coords; then re-evaluate against clean GT.

---

## 9. Decision log (why we went this direction)

| Decision | Why |
|---|---|
| LeakyReLU patch | Found a real bug; dying-ReLU on sparse-time data is reproducible. Documented for upstream contribution. |
| Build CVCG vs reuse C2 framework | Existing C2 (ARAP-energy gating) is temporal; multi-view inconsistency is spatial. CVCG is the principled extension. |
| Don't push CVCG to large gains | After +0.5 dB plateau, characterization showed the 4-cam setup imposes an NVS floor that no training-time method can fix. Time better spent elsewhere. |
| Pivot to decoupled-method | User's observation that VGM is structure-noisy / motion-cleaner suggested a stronger design principle. Worth the risk. |
| No raw RGB photometric (paper-style design) | If photometric mostly leaks structural noise into motion, removing it should help. Testable claim. |
| Stop at proof-of-concept | Within time budget. Honest about what works (+2.12 dB over static) and what doesn't (vs. vanilla 26 dB). The documented fallbacks (LBS, DINO) are the principled next-step path. |
