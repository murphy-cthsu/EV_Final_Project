# Design — Structure / Motion Decoupled 4D Reconstruction with VGM Supervision

> Status: **DRAFT for user review**, 2026-05-29 (v2: corrected `scene00` provenance).
> Supersedes the CVCG/C3 ablation direction (see `docs/reports/2026-05-29_checkpoint.md`).
> Evaluation: **scene00 = SV4D 2.0 output on D-NeRF *lego* scene; D-NeRF lego test set has the clean GT** we evaluate against. No scene switch needed.

## 0.5 Data provenance — scene00 IS D-NeRF lego (SV4D-supervised version)

The 5 mp4 at `/mnt/HDD_1/cthsu/multiview_videos/` are SV4D 2.0's output when
fed D-NeRF *lego*'s scene as input — same toy digger, same camera angle, same
motion (visually verified by inspecting `data/dnerf/lego/test/r_000.png` vs
`scene00_masked/train/r_00000.png`).

This gives us a clean 3-layer setup:

| Asset | Source | Role |
|---|---|---|
| `data/dnerf/lego/train/` (50 monocular frames, t∈[0,1]) | Original D-NeRF synthetic | Clean reference for **canonical** (Stage A) |
| `data/dnerf/lego/test/` (20 frames at novel cam+time pairs) | Original D-NeRF synthetic | **Clean held-out GT** for evaluation |
| `scene00_masked/train/` (5 views × 21 frames) | SV4D 2.0 on D-NeRF lego | **Noisy multi-view supervision** for motion training |

The model trains on SV4D's noisy supervision; renders at the D-NeRF test camera
+ time pairs; PSNR/SSIM/LPIPS computed against D-NeRF's clean GT. This is
exactly the workflow `scripts/sv4d_to_dnerf.py` was designed for — we just
need to add D-NeRF lego's `transforms_test.json` + `test/` to scene00.

---

## 0. Problem framing (one paragraph)

VGMs (SV4D 2.0 etc.) produce multi-view dynamic videos whose **3D structure is
noisy** (per-view hallucination at silhouettes, cross-view disagreement on
texture and shape) but whose **2D motion within each view is comparatively
clean** (per-view temporal coherence is the easy axis for VGMs). Vanilla SC-GS
jointly fits structure (canonical Gaussians) and motion (deform-MLP) from the
same noisy supervision — both channels are dragged down. We propose to
**decouple**: source structure from a clean canonical, and learn motion from
the VGM video using **only noise-robust supervision signals** (silhouette,
part-trajectories, ARAP rigidity, temporal smoothness) and a **dramatically
constrained motion parameterization** (per-part SE(3) instead of per-Gaussian
deform-MLP). **No raw-RGB photometric loss** — it would inject the VGM's
structural noise into motion training.

---

## 1. Why this design (the four claims)

| Claim | Why we believe it | Evidence we have |
|---|---|---|
| **C1**: VGM noise concentrates in structure, not motion. | Each VGM frame is generated from a temporal-consistency prior (sliding context); cross-view consistency is a weaker prior. So per-pixel per-frame appearance flickers across views, but a single view's frame-to-frame motion is smooth. | P0 measurement: 3.5 dB per-view PSNR spread (view 1 outlier); temporal axis (per-time mean residual) flat. |
| **C2**: Photometric loss on noisy RGB drags the motion learner toward mimicking artifacts. | The loss has no way to distinguish "correct motion produced wrong pixel" from "wrong motion produced wrong pixel"; gradient flows back into deform-MLP either way. | CVCG ablation: improvement is modest because photometric loss dominates and CVCG only dampens it ~10% at peak. |
| **C3**: Reducing motion search space by 25,000× shrinks the noise floor proportionally. | 84k Gaussians × 9 DOF × 21 t = 16M unknowns vs. 5 parts × 6 DOF × 21 t = 630 unknowns. With fewer DOF, noise has fewer dimensions to leak into. | Standard optimization reasoning; not directly measured, but the reduction is geometric. |
| **C4**: Multi-source weak supervision averages out individual signal noise. | Silhouette, trajectory, flow, ARAP each fail in different directions; their consensus is more robust. | Standard ML reasoning (analogous to noisy-label literature). |

C1 is empirically supported by our data. C2 is hypothesis we'll test. C3 and C4
are design choices justified by reasoning.

---

## 2. Pipeline overview

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                       INPUT (per scene)                          │
   │                                                                  │
   │   Multi-view video         (VGM output OR re-rendered D-NeRF)   │
   │   = V views × T frames × (RGB, alpha)                            │
   │   Camera poses             (Blender-convention c2w per view)     │
   │   FOV                       (1 number, scalar)                   │
   └──────────────────────────────┬───────────────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
        ▼                                                   ▼
  ┌─────────────────────┐                      ┌──────────────────────┐
  │ STAGE A             │                      │ STAGE B              │
  │ Clean canonical     │                      │ Motion signals       │
  │ (structure prior)   │                      │ (from VGM video)     │
  │ §3.1                │                      │ §3.2                 │
  └──────────┬──────────┘                      └──────────┬───────────┘
             │                                            │
             ▼                                            ▼
  ┌─────────────────────┐                      ┌──────────────────────┐
  │ STAGE C             │                      │ STAGE D              │
  │ Per-Gaussian part   │                      │ Per-part trajectory  │
  │ labels              │                      │ targets              │
  │ §3.3                │                      │ §3.4                 │
  └──────────┬──────────┘                      └──────────┬───────────┘
             │                                            │
             └──────────────────┬─────────────────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │ STAGE E                    │
                  │ Part-rigid deform module   │
                  │ + multi-signal loss        │
                  │ §3.5, §4                   │
                  └────────────┬───────────────┘
                               │
                               ▼
                  ┌────────────────────────────┐
                  │ STAGE F                    │
                  │ Evaluation (D-NeRF GT)     │
                  │ §5                         │
                  └────────────────────────────┘
```

---

## 3. Stage-by-stage I/O + purpose

### 3.1 Stage A — Clean canonical 3DGS

**Purpose**: produce a canonical 3D Gaussian set whose positions/colors/scales
encode object structure cleanly, ahead of any motion learning. This canonical
is FROZEN during all later training — motion learning cannot corrupt structure.

**Input**:
- For D-NeRF eval: D-NeRF's train images at t=0 only (clean GT, single
  timestep, multi-view via D-NeRF's many cameras with t≈0).
- For VGM scene00: 5 views' frame 0 RGBA (SAM-2-masked). All views share t=0.

**Output**:
- `canonical_gaussians.ply` — frozen 3D Gaussian state (xyz, color, scale,
  rotation, opacity). Roughly the v5 model's canonical, but trained on a
  single-timestep multi-view subset so it represents an *uncorrupted-by-motion*
  structure.

**Method** (two variants, A1 chosen for fastest path):

- **A1 (simplest, baseline)**: train vanilla 3D Gaussian Splatting (no
  deformation) on the t=0 frames of all V views for ~5000 iters. Standard SfM
  / canonical setup. Output: high-quality static 3DGS.
- **A2 (alternative)**: use AnySplat or image-to-3D model from a single high
  quality view. Existing project integration: `motionprior/integration/vgm.py`.
- **A3 (lazy)**: reuse v5's canonical Gaussians (from the fits-all-views run).
  Cheapest, but canonical was trained on motion-blurred multi-view; some
  motion-induced drift may have leaked into structure.

**Unusual choice (worth flagging)**: we *freeze* the canonical for the entire
motion-learning stage. Most 4D-GS work jointly optimizes both. We argue the
joint optimization is precisely what's harming us under noisy supervision.

---

### 3.2 Stage B — Motion signals from VGM video

**Purpose**: extract robust motion-related signals from the noisy VGM video
that do NOT include raw RGB photometric. Each signal is a per-(view, time)
tensor that motion training can pull from.

**Input**:
- The V views × T frames RGBA from STAGE A's input.
- The canonical Gaussians from STAGE A (for projection-based signals).

**Output** (per (view, time) cell):

| Signal | Shape | What it is | Method |
|---|---|---|---|
| `silhouette_mask` | (H, W) bool | per-frame foreground mask | SAM-2 video predictor on each view, propagate from a seed point on frame 0 |
| `part_masks` | (P, H, W) bool | per-frame per-part mask | SAM-2 hierarchical on frame 0 → propagated per-part through the video using SAM-2 video predictor |
| `optical_flow` | (H, W, 2) float | per-pixel 2D flow to next frame | RAFT (per view, frame-to-frame) |
| `flow_mask` | (H, W) bool | flow validity (cycle-consistency check) | standard RAFT post-processing |
| `texture_keypoints` | (K, 2) float | sparse keypoints stable across time per view | SIFT or LightGlue (optional, for §3.4) |

**Note**: per-part masks are obtained by running SAM-2 hierarchical on frame 0
(produces ~5-10 part masks for typical objects), then using SAM-2 video
predictor independently per part to propagate through the video.

**Unusual choice**: we do NOT compute photometric features (gradient, LPIPS).
We deliberately keep only signals that are either (a) silhouette/mask based or
(b) flow / sparse keypoint based — they're less affected by VGM's specific
texture hallucination.

---

### 3.3 Stage C — Per-Gaussian part labels

**Purpose**: assign each canonical Gaussian a part ID (0, 1, ..., P-1) so that
its motion is governed by its part's SE(3) trajectory in Stage E.

**Input**:
- Canonical Gaussians from Stage A.
- SAM-2 hierarchical part masks from Stage B (specifically: frame 0 per-view
  part masks, P parts).
- Camera poses.

**Output**:
- `gaussian_part_ids` — int tensor of shape (N_gaussians,) with values in
  {0, ..., P-1, -1 for static-background}.

**Method**:
1. For each Gaussian g, project to each view at t=0 using the V cameras (we
   already have `gaussian_mv_consistency.py` projection code).
2. For each view, look up which part mask the projected pixel falls inside.
3. Vote across views: g's part_id = majority vote, ties broken by alpha-weighted vote.
4. Gaussians that project outside all part masks in all views → label -1
   (treated as static background, no deformation applied).

**Unusual choice**: voting across V views resolves cross-view part disagreement
robustly. A Gaussian needs majority-view support to be assigned a part — single
view artifacts can't drag a Gaussian into the wrong part.

**Fallback for tearing at joint boundaries (LBS)**: hard assignment + per-part
SE(3) will produce visible tearing where parts meet (e.g. digger arm/cabin
junction, jumpingjacks shoulder). If P5 shows joint-region artifacts, upgrade
to **Linear Blend Skinning**:
- Output of Stage C becomes `gaussian_part_weights ∈ R^{N×P}` (soft, sums to 1
  per Gaussian), not a hard integer.
- Weights derived from per-view part-mask distance: pixel-distance-to-each-mask
  → soft weights (Gaussian kernel over distance), averaged across views.
- Stage E uses LBS: `g.xyz(t) = Σ_p weight(g, p) · T(p, t) · g.xyz_canonical`.
- Adds zero new DOF to motion (still P×T×6 unknowns); only Stage C output
  changes from integer to softmax-like float weights.

---

### 3.4 Stage D — Per-part trajectory targets

**Purpose**: convert the per-(view, time) part masks into a 3D centroid
trajectory per part — a clean supervision target for motion that's much less
noisy than per-pixel RGB.

**Input**:
- Per-view per-part 2D masks from Stage B.
- Camera poses.

**Output**:
- `part_centroid_3d` — float tensor of shape (P, T, 3) — for each (part, time),
  a 3D centroid in world coords.
- `part_centroid_confidence` — float tensor of shape (P, T) in [0, 1] — high
  when multi-view triangulation is well-conditioned, low when only one view
  sees this part.

**Method**:
1. For each (part p, time t, view v): compute 2D centroid of the part mask.
2. Lift to a per-view ray (one ray per (p, t, v)).
3. Triangulate across V views using DLT (or robust SVD if rays disagree).
4. Confidence = 1 / (1 + reprojection RMSE).

**Unusual choice**: working at the *centroid* level (one 3D point per
(part, time)), not per-pixel. This is robust because per-pixel matching is what
makes photometric noisy, while centroid averaging cancels noise.

---

### 3.5 Stage E — Part-rigid deform module + training

**Purpose**: learn the SE(3) trajectory T(p, t) for each part p over time t,
using the multi-signal loss. The canonical Gaussians stay frozen.

**Input**:
- Frozen canonical Gaussians (from A).
- Per-Gaussian part IDs (from C).
- Per-part 3D centroid trajectories + confidences (from D).
- Per-(view, time) silhouette + part masks (from B).
- Optionally optical flow (from B).
- Camera poses.

**Output**:
- `part_trajectories.pt` — learned T(p, t) for all parts and times, encoded as
  per-part-per-time 6 DOF (translation + axis-angle rotation) or quaternion.
- A renderable model: given canonical Gaussians + part IDs + T(p, t), produce
  deformed Gaussians at time t.

**Method**:
- Parameterization: a small MLP `f(p, t) -> SE(3)` that takes part ID (one-hot
  or embedded) + time (positional encoded), outputs translation + rotation.
  Alternative: per-(p, t) free parameters (P×T×6 numbers), no MLP at all,
  trained with smoothness regularization.
- For Gaussian g with part_id p_g: `g.xyz(t) = T(p_g, t) · g.xyz_canonical`
  (similarly for rotation; scale is unchanged).
- Train with the multi-signal loss (§4).

**Unusual choice**: the part-rigid parameterization is fundamentally different
from SC-GS's deform-MLP. SC-GS's per-Gaussian deformation has 16M+ DOF; ours
has ~600. The reduction makes optimization much easier and noise-robust, but
**at the cost of expressivity**: only piecewise-rigid motion can be
represented. For most articulated objects (humans, animals, toys with rigid
parts) this is exactly the right inductive bias. For squishy / continuously
deformable objects it would be wrong.

---

## 4. Loss design

```python
L_total = λ_silh    * L_silhouette        # alpha mask match per view
        + λ_traj    * L_part_traj         # part centroid match
        + λ_flow    * L_optical_flow      # 3D-lifted flow direction (optional)
        + λ_arap    * L_arap_articulated  # within-part rigid (auto-satisfied by parameterization;
                                          #   may keep for stability)
        + λ_smooth  * L_temporal_smooth   # T(p, t) ≈ T(p, t+1)
        + λ_xview   * L_cross_view        # same-3D-pt → same render color across views
        # explicitly: NO raw-RGB photometric L1
```

### 4.1 L_silhouette
Render the deformed Gaussians at time t from each view → get rendered alpha
mask `M_render(v, t)`. Compare to SAM-2 silhouette `M_gt(v, t)`:
```
L_silhouette = mean over (v, t) of  binary_cross_entropy(M_render, M_gt)
             + IoU loss (for sharper boundary)
```

### 4.2 L_part_traj
For each (part p, time t), compute the rendered Gaussian centroid in 3D:
```
C_pred(p, t) = mean over g in part p of  g.xyz(t)
L_part_traj = mean over (p, t) of  confidence(p, t) * ||C_pred(p, t) - C_gt(p, t)||²
```

### 4.3 L_optical_flow (optional)
For each (view, t, pixel), the rendered Gaussian motion projects to a 2D motion
vector at that pixel. Compare to RAFT flow:
```
L_optical_flow = mean over (v, t, p) of  cosine_distance(render_flow(v,t,p), raft_flow(v,t,p))
```
We use cosine distance (direction match), not magnitude — flow magnitude in VGM
is often miscalibrated.

### 4.4 L_arap_articulated
Per-edge ARAP between K-NN Gaussians with anisotropic weights:
```
λ_intra (same part) = 1.0, λ_inter (different part) = 0.05
```
This is already implemented in `motionprior/losses/arap_articulated.py`. Even
though the part-rigid parameterization auto-satisfies within-part rigidity, we
keep this loss for cross-part flexibility / stability.

### 4.5 L_temporal_smooth
```
L_temporal_smooth = mean over (p, t) of  ||T(p, t+1) - T(p, t)||²
```

### 4.6 L_cross_view
Render canonical (deformed) state from each view at time t. For each 3D point
visible in multiple views, the rendered RGBs should agree:
```
L_cross_view = mean over (3D_pt, view_pair) of  ||render_color(v1) - render_color(v2)||²
```
This is the CVCG idea repurposed as a self-consistency loss rather than a gate.

### 4.7 Fallback: L_feature (DINOv2 / CLIP), NOT raw RGB

**Risk to address**: silhouette + centroid losses are nearly invariant to
self-rotation around the part's axis (a cylinder or sphere can spin without
moving its centroid or changing its silhouette). If P5 shows correct-position
but wrong-orientation parts (digger bucket facing the wrong way at the end of
the swing), we add **semantic feature consistency**, NOT raw RGB.

```
L_feature = mean over (v, t) of  ||DINOv2(rendered) - DINOv2(gt)||²
```

Why DINOv2/CLIP instead of raw RGB:
- VGM noise is mostly high-frequency texture hallucination. Foundation-model
  features encode shape + semantics — they're robust to per-pixel hallucination.
- Empirically (in the wider literature: DreamFusion, ProlificDreamer, etc.),
  DINO feature loss tolerates noisy supervision much better than L1/L2.
- Applied at low resolution (DINOv2 ViT input = 224×224), much fewer DOF for
  noise to leak through than 576×576 pixel L1.

Defer to P6+. Skipped in P1–P5 to test the "no raw photometric" claim cleanly.

---

## 5. Evaluation protocol

### 5.1 Evaluation against clean GT (no scene switch needed)

Per §0.5, scene00 IS D-NeRF lego. Its clean GT lives in `data/dnerf/lego/test/`
(20 frames at novel cam+time pairs). The evaluation protocol:

1. Train on `scene00_masked/train/` (5 SV4D views × 21 frames, noisy).
2. At eval time, render at the camera+time pairs listed in
   `data/dnerf/lego/transforms_test.json`.
3. Compute PSNR/SSIM/LPIPS against `data/dnerf/lego/test/r_NNN.png` (clean GT).

This is the protocol `scripts/sv4d_to_dnerf.py` was designed for. Our existing
`scene00_masked` lacks the D-NeRF test linkage — we'll fix this in **P1.5** (a
quick scene-rebuild step that adds `transforms_test.json` + a symlink to
`data/dnerf/lego/test/` into `scene00_masked`).

**One subtlety**: D-NeRF test frames are RGB on a black background (no alpha).
SV4D output (post-SAM-2) is RGBA with white background. We need to either
(a) composite our render onto black at eval time, or (b) re-mask the D-NeRF
test images. (a) is simpler. The evaluation script will handle this.

**Multi-scene extension**: same protocol applies if we add jumpingjacks etc.
We'd run SV4D 2.0 on jumpingjacks to produce a `scene_jumpingjacks_sv4d/`,
then evaluate against `data/dnerf/jumpingjacks/test/`. Existing project
infrastructure (`run_sv4d_supervised_pipeline.py`) supports this; we'd hit
SV4D inference time (~10 min/scene) but no other roadblocks.

### 5.2 Metrics

| Metric | What it measures | Where used |
|---|---|---|
| Held-out test PSNR/SSIM/LPIPS on D-NeRF GT | True reconstruction quality vs clean reference | Headline; can't compute on scene00 |
| GT-free 3D-consistency (Gaussian mask-projection) | 3D coherence regardless of GT availability | Cross-scene; the only metric we can compute on both D-NeRF + scene00 |
| Per-part trajectory error vs GT trajectories | Motion accuracy specifically | D-NeRF only (we don't have GT trajectories for scene00) |
| Number of canonical Gaussians used | Representation compactness | Both |
| Wall-clock training time | Engineering | Both |

### 5.3 Comparisons (the ablation table we want)

All rows use D-NeRF lego test set (20 frames at novel cam+time) as eval. Train
sets differ by row. **PSNR/SSIM/LPIPS against clean D-NeRF GT in every row.**

| Row | Train data | Method | What it tests |
|---|---|---|---|
| 0 | D-NeRF lego train (clean monocular) | Vanilla SC-GS | Upper-bound reference: best PSNR achievable on this scene with clean data |
| 1 | scene00 (SV4D-noisy multi-view) | Vanilla SC-GS | Baseline: how much VGM noise hurts (gap from row 0 = noise penalty) |
| 2 | scene00 | Vanilla SC-GS + LeakyReLU + SAM mask | Reproduces our v6 baseline |
| 3 | scene00 | + Our part-rigid + multi-signal (no photometric) | **HEADLINE**: gap closed back toward row 0? |
| 4 | scene00 | + Our method WITH photometric L1 retained | Ablation: does removing photometric actually help? |
| 5 | scene00 | + Our method, free SE(3) (no MLP smoothing) | Ablation: simple parameterization sufficient? |

Row 3's gap-closed-vs-row-1 is the main number. Row 4 isolates the
"no-photometric" claim (most novel). Row 5 isolates the parameterization
choice.

If multiple D-NeRF scenes (jumpingjacks etc.) are run, the same rows are
replicated per scene; we report **mean across scenes** as the headline.

---

## 6. Phased implementation plan

| Phase | Tasks | Time | Risk |
|---|---|---|---|
| **P1** Stage A canonical | Train static 3DGS on D-NeRF lego at t=0 (use train frames near t=0; clean structure) → canonical .ply | 1–2 hr | Low (vanilla 3DGS) |
| **P1.5** Eval scene rebuild | Add D-NeRF lego transforms_test.json + symlinks to `scene00_masked` so SC-GS can eval against clean GT | 20 min | Trivial |
| **P2** Stages B–D | SAM-2 hierarchical on canonical view; SAM-2 video propagate per-part on each of 5 mp4 → trajectories; project labels to canonical Gaussians | 3–4 hr | Medium |
| **P3** Stage E reparam | Part-rigid SE(3) module + replace SC-GS deform | 4 hr | Medium-high |
| **P4** Losses | Wire L_silhouette + L_part_traj + L_smooth + L_arap | 3 hr | Medium |
| **P5** Run rows 0–3 + eval | Row 0 (D-NeRF clean upper bound), Row 1 (scene00 vanilla), Row 2 (scene00 v6 baseline already done), Row 3 (our method) → eval against D-NeRF lego test | 6–8 hr | Medium (training time mostly) |
| **P6** Ablations (rows 4–5) | Re-run row 3 with photometric retained / with free SE(3) | 3–4 hr | Low (re-runs) |
| **P7** scene00 qualitative | Build comparison GIFs for slides | 2 hr | Low |
| **P8** Report + slides + README | Write up | 1 day | Low |

Total: 4–5 days for P1–P5 (minimal viable headline) + P6 + P7–P8.

---

## 7. Risks + mitigations

| Risk | Mitigation |
|---|---|
| **Part-rigid is wrong inductive bias for our scene** (e.g. the digger's tracks deform continuously) | Run on multiple D-NeRF scenes; jumpingjacks (articulated body) should fit. If our digger fails but jumpingjacks works, frame as "method targets articulated objects." |
| **Stage A canonical is bad** (insufficient multi-view triangulation from 5 cams at t=0) | Use D-NeRF first where multi-view is dense; fall back to v5 canonical as A3 variant. |
| **Stage D triangulation is noisy** (SAM-2 mask centroids disagree across views) | Confidence weighting; visualize per-(p, t) error; downweight low-confidence cells. |
| **Optimization gets stuck** at identity (T(p, t) = identity for all t) | Initialize T from Stage D trajectories (warm start); don't trust pure-random init. |
| **L_silhouette alone is too weak** (no texture supervision → wrong rotations) | Always combine with L_part_traj; if both fail, add L_optical_flow. |
| **D-NeRF supervision-mismatch** (SV4D's azimuths ≠ D-NeRF train cameras) | Existing `sv4d_to_dnerf.py` handles this; we don't need to compare to D-NeRF training views, only its TEST views. |
| **Joint-region tearing** from hard part assignment + independent SE(3) per part — most obvious at digger arm/cabin junction, jumpingjacks shoulder | Built-in upgrade path: §3.3 documents the LBS variant. Triggered if P5 visual inspection shows joint artifacts. Adds zero motion DOF (still P×T×6), only Stage C output changes from int → softmax weights. |
| **Self-rotation underdetermined** by silhouette + centroid alone (a cylinder can spin without changing either) — manifests as parts in correct 3D position but wrong rotation | Add `L_feature` (DINOv2/CLIP semantic feature distance) in P6+, **NOT raw RGB L1**. DINO features are robust to VGM's high-freq hallucination because they encode shape/semantics. See §4.7. |

---

## 8. Open questions the user should weigh in on

1. **Canonical source**: A1 (train static 3DGS from t=0), A2 (AnySplat), or A3
   (reuse v5)? My recommendation: A1 for D-NeRF (clean data, fast), then A3
   for scene00 (we already have v5).
2. **SE(3) parameterization**: free per-(p, t) params (simpler, fewer
   hyperparams) vs MLP `f(p, t)` (smoother, more compact). My recommendation:
   start with free params + smoothness loss; if it works, no need for MLP.
3. **Number of parts**: SAM-2 hierarchical produces ~5–10 parts depending on
   the object. Take all of them? Merge small ones into "other"? My
   recommendation: take all, no merging, but track per-part trajectory
   confidence; auto-prune parts with low cross-view support.
4. **Optical flow signal**: include from the start, or only if base method
   underperforms? My recommendation: skip in P1–P5; add in P6 only if
   needed.

---

## 9. What this design replaces

This design **deprecates** our earlier CVCG/C3-ablation direction (see
`docs/reports/2026-05-29_checkpoint.md` §5–6). That work stays in the repo as engineering
contribution (dying ReLU fix; characterization of VGM artifacts) but is no
longer the methodological headline. The new method is more decisive:

| Aspect | Old (CVCG/C3) | New (this design) |
|---|---|---|
| Supervision | Photometric L1 (noisy) + gating | Silhouette + part-trajectory + flow (denoised) |
| Motion DOF | 16M (per-Gaussian deform-MLP) | 630 (per-part SE(3) over time) |
| Structure-motion coupling | Jointly trained | Decoupled — canonical frozen |
| Gain | +0.5 dB peak, +1.7 pp 3D-consistency | TBD — designed to be bigger |
| Headline claim | "CVCG mitigates VGM multi-view inconsistency" | "Photometric loss is harmful; replace with part-decomposed multi-signal supervision" |

---

## 10. Decisions you need to make for me to start

| Decision | Default | If you want different |
|---|---|---|
| Approve this design overall? | YES → I start P1 | Tell me which section to revise |
| Scene? | **scene00 = SV4D-on-D-NeRF-lego** (we already have it) | Also add jumpingjacks etc. — incurs SV4D inference cost (~10 min/scene) |
| Canonical source (A1/A2/A3)? | **A1**: train static 3DGS on D-NeRF lego near t=0 (clean structure from clean source) | A3 (reuse v5 canonical) if we want to save 1-2 hr but accept some motion-induced drift in canonical |
| SE(3) parameterization? | Free per-(p, t) params + smoothness loss | MLP `f(p, t) → SE(3)` if you want |
| Include optical flow in P1? | No (defer to P6) | Yes if you want full signal set from start |
| First report draft date? | After P5 finishes (~3 days) | Specify if you want sooner / progress-only checkpoint |
