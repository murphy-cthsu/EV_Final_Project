# Final presentation — slide outline (2026-05-29)

> **Format**: 8 sections, ~20–25 min total. Visuals over bullets — slides exist
> to **project the evidence**, talking explains it.
> **Companion**: [`../reports/2026-05-29_final_report.md`](../reports/2026-05-29_final_report.md)
> **Asset roots**: `runs_aux/*` (all figures), `outputs/custom/*` (all renders).

---

## Time budget

| § | Section | Slides | Time | Role |
|---|---|---:|---:|---|
| 1 | Hook + setup | 2 | 2 min | What problem, what data |
| 2 | Diagnosis 1: dying-ReLU engineering finding | 2 | 2 min | A real SC-GS bug |
| 3 | Diagnosis 2: VGM artifact characterization | 2 | 2.5 min | View-localized 3D inconsistency |
| 4 | Phase 1 method: CVCG + C3 within-framework | 3 | 3.5 min | Framework extension, modest gain |
| 5 | Phase 2 method (headline): structure/motion decoupled | 4 | 5 min | The contribution |
| 6 | Results + visual quality | 3 | 3 min | Numbers + 3-col gallery |
| 7 | Diagnostic: clean-ref cross-render limits | 2 | 2 min | Why VGM-as-GT, the alignment limit |
| 8 | Limitations + path forward | 2 | 2 min | Honest closing |
| | **Total** | **20** | **22 min** | leaves 3 min buffer |

---

## Section 1 — Hook + Setup (2 slides, 2 min)

### Slide 1 — Title + headline visual (1 min)

- **Title**: *"Structure / Motion Decoupled 4D Gaussian Reconstruction from VGM Supervision"*
- **Subtitle**: Final project · NTU 113-2 EV · 2026-05-29
- **Headline visual** (full-slide):
  [`runs_aux/gallery_3col_full/contact_sheet_t0.png`](../../runs_aux/gallery_3col_full/contact_sheet_t0.png)
  — 5 views × t=0, three columns [clean ref nobase | SV4D GT | our part-rigid LBS].
- **Verbal pitch (30 s)**: "We train a 4D-GS on Video-Generative-Model output.
  Two contributions: (1) we diagnose why naive SC-GS fails on this data, and
  (2) we propose a decoupled method that learns motion with 5,000× fewer
  parameters and no RGB photometric loss."

### Slide 2 — Data + pipeline (1 min)

- **Visual** (left half): screenshot of `data/custom/scene00_masked/train/` — 5
  views × 21 frames RGBA tiles, white-on-white showing SAM-2 masking.
- **Visual** (right half): pipeline diagram
  ```
  mp4 + cam.json  →  to_dnerf format  →  SAM-2 mask  →  SC-GS / our method
  ```
- **One-liner**: "5 views × 21 frames at 576², SV4D 2.0 output of D-NeRF
  *lego* scene. Input to SC-GS-style 4D pipeline."

---

## Section 2 — Diagnosis 1: Dying ReLU (2 slides, 2 min)

### Slide 3 — The symptom (1 min)

- **Title**: *"Vanilla SC-GS: PSNR 17.39 dB, ZERO learned motion."*
- **Visual**: side-by-side GIF before-fix vs after-fix from
  `outputs/custom/scene00_v5_node/train/ours_30000/gifs/`.
- **Bullet**:
  - Deform-MLP output time-invariant
  - Hidden vector probe: `mean = 0.0000e+00` exactly
  - Classic dying-ReLU

### Slide 4 — The fix (1 min)

- **Title**: *"One-line LeakyReLU patch → +14.4 dB"*
- **Code diff**:
  ```python
  # third_party/SC-GS/utils/time_utils.py:418
  -    h = F.relu(h)
  +    h = F.leaky_relu(h, negative_slope=0.01)
  ```
- **Bar chart**: 17.39 → 31.79 dB.
- **Closing line**: "This is a real SC-GS bug on sparse-time multi-view inputs.
  The patch is documented in the repo."

---

## Section 3 — Diagnosis 2: VGM Artifact (2 slides, 2.5 min)

### Slide 5 — Per-view residual heatmap (1.5 min)

- **Visual**: `runs_aux/vgm_artifact/heatmap.png`
- **Per-view PSNR table** (overlay or right panel):
  | view | mean PSNR | rank |
  |---|---:|---|
  | 0 | 24.28 | mid |
  | **1** | **23.25** | **worst** |
  | 2 | 25.59 | mid |
  | 3 | 25.39 | mid |
  | 4 | 26.74 | best |
- **One-liner**: "3.5 dB per-view spread; **all 9 worst-residual cells from
  view 1**. View-localized inconsistency, not random."

### Slide 6 — Boundary residual signature (1 min)

- **Visual**: `runs_aux/vgm_artifact/spatial_avg.png` + `worst_frames.png`.
- **Closing line**: "Residual concentrates at silhouette boundary — classic
  SV4D boundary-hallucination signature. This motivates the protective
  components (Phase 1) AND the no-photometric design (Phase 2)."

---

## Section 4 — Phase 1: CVCG + C3 (3 slides, 3.5 min)

### Slide 7 — CVCG (cross-view consistency gating) (1 min)

- **Title**: *"Component (PROTECTIVE): down-weight motion gradient at Gaussians
  whose 5-view photometric agreement is inconsistent."*
- **Diagram**: per-Gaussian projection → 5-view photo color → variance →
  sigmoid gate → multiplied into temporal-PE gradient.
- **One-liner**: "Identity-prefixed PE means gate skips static channels —
  preserves stable 3D structure while suppressing inconsistent motion."

### Slide 8 — C3 (frequency curriculum) (1 min)

- **Title**: *"Component (PROTECTIVE): release temporal-PE bands progressively."*
- **Visual**: animation/schedule of which PE bands are active over iterations.
- **One-liner**: "Force macro-motion to be learned before high-frequency
  jitter. Regime-dependent: fast release for hard regimes, slow for easy."

### Slide 9 — Phase 1 results (1.5 min)

- **Title**: *"+0.51 dB peak, +1.72 pp 3D-consistency, half the Gaussians."*
- **Table**:
  | Split | Method | PSNR | # Gaussians | 3D-consistency |
  |---|---|---:|---:|---:|
  | View-split | vanilla | 13.10 | 96,309 | 47.71% |
  | View-split | +C3+CVCG (fast) | **13.61** (+0.51) | **43,664** | **49.43%** (+1.72 pp) |
  | Temporal-split | vanilla | 25.75 | — | — |
  | Temporal-split | +C3+CVCG (slow) | **27.33** (+1.58) | — | — |
- **Pivot line**: "Phase 1 works but plateaus. The bigger gain needs
  rethinking the supervision form — not just gating it."

---

## Section 5 — Phase 2: Structure/Motion Decoupled (4 slides, 5 min)

### Slide 10 — Design rationale (1 min)

- **Title**: *"What if we never use raw RGB?"*
- **Three-step rationale** (animated bullets):
  1. VGM noise concentrates in pixel boundary (Section 3) — don't use pixel loss
  2. Structure should be CLEAN (from t=0 only) — freeze it
  3. Motion is mostly rigid (arm + body) — parameterize with SE(3)
- **One-liner closing**: "16M DOF → 126 DOF. 5,000× reduction."

### Slide 11 — Pipeline (1.5 min)

- **Title**: *"Five stages: A frozen canonical → B-D weak signals → E SE(3) learning."*
- **5-stage diagram** (horizontal flow):
  ```
  [A] static 3DGS         [B] motion mask          [C] part assignment       [D] 3D trajectory       [E] SE(3) learning
  (t=0 only, frozen)   →  (temporal variance)  →   (5-view voting)        →   (DLT triangulation) →  (126 DOF, 35 sec)
  PSNR 39.4 dB              29% arm pixels            13,756 arm Gaussians   confidence 0.94          NO photometric loss
  ```
- **Bottom bar**: visual of motion mask propagated across t for one view —
  `runs_aux/parts_motion/view0_temporal_contact.png`.

### Slide 12 — Part assignment visualization (HEADLINE VISUAL) (1.5 min)

- **Visual**: `runs_aux/part_assignment_anim/canonical_part_assignment_contact_sheet.png`
  (5 views, Gaussians colored by LBS weight: **red=arm, blue=body, purple=boundary**)
- **Stats overlay**:
  - 33 pure-arm (w > 0.9, red)
  - 24,525 pure-body (w < 0.1, blue)
  - **29,917 LBS boundary** (purple)
- **Animation gif inline (per slide config)**:
  `runs_aux/part_assignment_anim/part_anim_v0.gif`
- **Closing line**: "The arm cluster (red) moves coherently across t while body
  stays static — exactly the rigid-decomposition prior we encoded."

### Slide 13 — Loss + training (1 min)

- **Title**: *"Multi-signal weak supervision replaces L1 photometric."*
- **Loss formula**:
  ```
  L = λ_silh   · L_silhouette  (BCE + 1-IoU on rendered alpha)
    + λ_traj   · L_part_traj   (||rendered arm centroid - 3D target||² × confidence)
    + λ_smooth · L_temporal_smooth (||T(t+1) - T(t)||²)
  ```
- **Bullets**:
  - 126 learnable DOF (vs 16M for SC-GS deform-MLP)
  - 35 seconds wall-clock (vs ~15 min vanilla)
  - **No RGB pixel comparison anywhere**

---

## Section 6 — Results + Visual Quality (3 slides, 3 min)

### Slide 14 — Headline number (1 min)

- **Title**: *"+2.12 dB over zero-motion, 5,000× fewer DOF."*
- **Bar chart**:
  | Method | PSNR | DOF | Time |
  |---|---:|---:|---:|
  | Static (no motion) | 15.91 | 0 | — |
  | **Ours (Part-rigid LBS)** | **18.03** | **126** | 35 s |
  | Vanilla SC-GS | 25.75 | 16,000,000 | ~900 s |
- **Two takeaways**:
  - **Proof of concept**: weak signals DO learn motion (+2.12 over static)
  - **Capacity-limited**: 22% of vanilla's gain — bottleneck is DOF, not signal

### Slide 15 — Visual gallery (1.5 min) — HEADLINE QUALITATIVE

- **Visual**: `runs_aux/gallery_3col_full/all_views_animation.gif`
  (5 views stacked, 3-col [clean | SV4D | ours], 21 frames)
- **OR**: individual `gallery_v0.gif` for one focused view
- **3 failure modes annotated**:
  1. Self-rotation underdetermined → bucket pointing wrong direction
  2. Joint tearing/feathering → arm/body boundary fuzz
  3. Canonical scale fixed at t=0 → Gaussians can't deform with arm

### Slide 16 — Photometric ablation refutes original framing (30 s)

- **Title**: *"Adding raw L1 photometric: +0.29 dB. Heaviest blur+erode: -0.25 dB."*
- **Mini-table** (8 variants):
  | Variant | PSNR |
  |---|---:|
  | No photo (Part-rigid v1, hard ID) | 18.03 |
  | LBS, no photo | 17.68 |
  | LBS + raw L1 | 17.97 |
  | LBS + blur σ=25 + erode 51 (heaviest) | 17.78 |
- **One-liner**: "**All 8 variants land in 17.7–18.0 dB**. Supervision form
  does not explain the gap to vanilla. The bottleneck is **126 DOF saturated**."

---

## Section 7 — Diagnostic: Clean-Ref Cross-Render (2 slides, 2 min)

### Slide 17 — Why SV4D isn't D-NeRF (1 min)

- **Title**: *"We tried using a clean D-NeRF 4D-GS as quantitative GT. It can't."*
- **Visual**: `runs_aux/alignment_A_nobase/matched_curves.png`
  (V-shape matched fid + 12 dB PSNR ceiling)
- **Three findings**:
  1. **Temporal**: SV4D V-shape vs D-NeRF monotonic → DTW impossible
  2. **Geometric**: SV4D cameras off the unit sphere (3.92–4.32 vs 4.031)
  3. **Stylistic**: VGM grain vs deterministic GS render → 12 dB hard ceiling

### Slide 18 — Diagnostic + fix attempt (1 min)

- **Visual**: `runs_aux/clean_ref_aligned_nobase/vis/diag_v0_t00.png`
  (bbox + scale visualization)
- **Fix attempts table**:
  | Fix | PSNR | Δ |
  |---|---:|---:|
  | Baseline | 12.19 | — |
  | Baseplate Gaussian removal | 12.35 | +0.16 |
  | + Per-frame shift | 12.12 | −0.22 (regresses) |
- **Closing line**: "SV4D rewrites world transform AND animation timing. The
  upstream training distribution is not a quantitative GT — only a qualitative
  reference."

---

## Section 8 — Limitations + Path Forward (2 slides, 2 min)

### Slide 19 — Honest limitations (1 min)

- **Title**: *"What's broken in our method, and why."*
- **Two-column**:
  - **Capacity-limited**: 126 DOF saturated; no parameter to absorb finer signal
  - **Rotation underdetermined**: silhouette + 3D centroid don't constrain
    self-rotation (bucket faces wrong direction)
  - **Canonical rigidity**: Gaussian scale/color fixed at t=0; can't stretch
    coherently as arm rotates
  - **Trajectory target itself is noisy**: 0.241 mean tracking error from VGM
    centroid jitter (Section 5.4)

### Slide 20 — Path forward (1 min)

- **Title**: *"Three principled next steps."*
- **Three cards**:
  1. **Hierarchical parts** (~378 DOF, +252 from current): K-means sub-decompose
     arm into 3 sub-parts; expand DOF where it's actually saturated.
     `train_partrigid_hier.py` written, not yet run.
  2. **DINOv2 feature loss**: foundation-model features tolerate per-Gaussian
     shape artifacts but penalize semantic mismatch — should constrain
     rotation. Right next experiment.
  3. **Per-time Gaussian scale** (~1M params, still 16× less than vanilla):
     allow canonical Gaussians to stretch as arm rotates. Mid-term.

- **Closing line** (30 s):
  > "We diagnosed two real problems in this regime (the dying-ReLU bug and the
  > VGM artifact structure), built two complementary methods (CVCG/C3 within
  > the framework, and structure/motion decoupling that rethinks supervision
  > form), and produced honest evidence for what works, what doesn't, and why.
  > The decoupled method's 5,000× parameter reduction is a real result; closing
  > the 8 dB gap to vanilla is well-scoped future work with three concrete
  > paths."

---

## Backup slides (Q&A only)

| B# | Title | Source |
|---|---|---|
| B1 | Full ablation table (Phase 1, all 4 runs × 2 splits) | `report §4` |
| B2 | 8-variant photometric table | `report §5.5` |
| B3 | Trajectory target noise diagnostic | `runs_aux/results_gallery/arm_trajectory.png` |
| B4 | 3D-consistency metric definition + math | `report §4` |
| B5 | Stage A canonical PSNR 39.4 | `runs_aux/results_gallery/canonical_quality.png` |
| B6 | LBS weight distribution histogram | inline derive from `partrigid_lbs_photo1/partrigid_state.npz` |
| B7 | Full clean-ref alignment GIFs per view | `runs_aux/alignment_A_nobase/alignment_v{0-4}.gif` |
| B8 | Static-region PSNR + variance heatmap | `runs_aux/static_region_C/static_vs_full_bar.png` |
| B9 | Pipeline code links | `scripts/{multiview_videos_to_dnerf, sam2_seg_multiview, train_partrigid_lbs}.py` |

---

## Talking-point tips

### Where to slow down
- **Slide 1 (headline visual)**: 30 s of silence before talking — let the
  3-col comparison speak.
- **Slide 12 (part assignment)**: the red/blue cluster contact sheet is the
  single most informative slide; spend 60+ s on it.
- **Slide 15 (visual gallery animation)**: let the GIF cycle 2× before
  annotating failure modes.

### Likely Q&A and prepared responses

- **"Why not just train vanilla SC-GS and accept the 25.75 dB?"** → "We did
  (Section 4). The contribution isn't the number — it's: (a) characterizing
  *why* VGM data is hard (Section 3), (b) showing the no-photometric design
  works as a proof of concept (Section 5), (c) the 5,000× parameter
  reduction + 26× training time speedup is independently valuable for
  motion-only re-training applications."

- **"126 DOF seems too few — why not start higher?"** → "Started minimal to
  isolate the supervision-form variable (Section 5.5). The 8-variant
  photometric ablation showed supervision form is NOT the bottleneck.
  Hierarchical parts (~378 DOF) and per-time scale (~1M params) are the
  scoped capacity expansions — both written, scoping the next term."

- **"Clean-ref alignment to 12 dB — is that a method failure?"** → "It's a
  *measurement* finding, not method failure. SV4D produces images at a
  different scene scale and stylistic distribution than D-NeRF. The
  finding has independent value for any future SV4D-evaluation pipeline."

- **"What about RigGS / VideoArtGS / 4D-Fly?"** → "Closest works in §1 of
  the report. Our decoupled formulation differs in that we use *zero*
  photometric supervision and parameterize motion explicitly as part-SE(3)
  rather than relying on ARAP or skeleton extraction. The 8-variant
  ablation shows photometric removal is the genuine forcing variable."

---

## Practical prep

- All GIFs pre-load (Slack/Drive) — avoid runtime decoder hangs.
- Backup PSNR table on phone (Slide 14) — recover if numbers are
  challenged live.
- Reproduce commands (any visual) → [`docs/runbooks/demo_runbook.md`](../runbooks/demo_runbook.md).
- Full report → [`docs/reports/2026-05-29_final_report.md`](../reports/2026-05-29_final_report.md).
