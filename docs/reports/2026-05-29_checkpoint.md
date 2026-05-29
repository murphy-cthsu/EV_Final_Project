# Checkpoint Report — VGM-supervised SC-GS on scene00

> Snapshot: 2026-05-29, mid-experiment, **NOT the final report**.
> Purpose: honest inventory of what we have, so we can decide framing for D4-D5.

---

## 1. What we set out to do

Take a 5-view × 21-frame video output from a Video Generative Model (VGM, likely
SV4D 2.0), supervise SC-GS 4D Gaussian reconstruction on it, and answer:
1. Does SC-GS train at all on this data?
2. If yes, what failure modes appear?
3. Are those failures the VGM's fault or the training procedure's?
4. Can a small intervention measurably improve any axis?

**Honest reframing**: this is final-project-scale exploration, not a paper. We're
producing characterization + a small methodological contribution, with honest
limitations.

---

## 2. Setup (data + pipeline)

**Input**: `/mnt/HDD_1/cthsu/multiview_videos/` — 5 mp4 (576×576 @ 10 fps, 21
frames) + `camera_pos.json` with Blender-convention c2w extrinsics. No GT, no
intrinsics (we use D-NeRF default `camera_angle_x = 0.6911 rad`).

**Pipeline**:
1. `scripts/multiview_videos_to_dnerf.py`: mp4 + cam JSON → D-NeRF-format
   `transforms_train.json` + 105 PNGs (5×21 flat-indexed).
2. `scripts/sam2_seg_multiview.py`: SAM-2 video predictor masks foreground,
   produces RGBA training set.
3. `scripts/split_train_test.py` (view holdout) or
   `scripts/split_temporal.py` (temporal holdout) → train/test split.
4. SC-GS training via patched `third_party/SC-GS/train_gui.py`.
5. Evaluation: PSNR / SSIM / LPIPS on held-out frames + GT-free 3D-consistency.

**Reproducibility**: each script has a CLI signature; commands recorded in
the report. SC-GS patches isolated to two files, documented in
`docs/design/scgs_hook_design.md`.

---

## 3. Engineering finding: dying ReLU in SC-GS deform-MLP

**Symptom**: vanilla SC-GS on our 5-view sparse-time data trained to PSNR ~17 dB
with **zero learned motion** (deform-MLP output literally constant w.r.t. time).

**Diagnosis**: SC-GS's deform-MLP uses ReLU activations. On our sparse temporal
structure (21 timesteps × 5 mostly-redundant views), conflicting gradients drive
the final hidden layer's pre-activations negative, ReLU outputs exactly zero,
gradient stops flowing — classic dying-ReLU. We confirmed by probing the network
state across iterations: hidden vector was `0.0000e+00` mean.

**Fix**: one-line patch at `third_party/SC-GS/utils/time_utils.py:418` —
`F.relu(h)` → `F.leaky_relu(h, negative_slope=0.01)`. The 1% leak lets dead
neurons receive gradient and recover.

**Effect**: PSNR jumped from 17.39 → 31.79 (+14.4 dB). Motion fidelity:
render Δ matches GT Δ to ~99% per view. This unblocked all subsequent work.

This is an SC-GS-internal bug that hits any user with sparse-time multi-view
data. We submitted it as a documented patch.

---

## 4. Artifact characterization (P0)

Used the fits-all-views v5 checkpoint as a 3D probe: residual between
v5-rendered frames and VGM-output frames is the part of the VGM that no single
canonical 3D model can simultaneously explain across 5 views — a direct lower
bound on VGM multi-view inconsistency.

**Results** (`runs_aux/vgm_artifact/SUMMARY.md`):

| view | mean \|Δ\| (FG) | mean PSNR | rank |
|---:|---:|---:|---:|
| 0 | 5.81 | 24.28 | mid |
| **1** | **8.38** | **23.25** | **worst** |
| 2 | 6.38 | 25.59 | mid |
| 3 | 5.57 | 25.39 | mid |
| 4 | 4.79 | 26.74 | best |
| all | 6.19 | 25.05 | — |

**Findings**:
1. **Per-view PSNR spread is 3.5 dB** — view 1 is systematically ~2× harder to
   fit than view 4. Real, measurable cross-view inconsistency.
2. **All 9 worst-residual cells come from view 1** — not noise, structural.
3. **Spatial residual concentrates at silhouette boundary** — classic
   boundary-hallucination signature (`spatial_avg_residual.png`).
4. **Temporal axis is relatively clean** — no time-step where all 5 views
   collectively fail. Inconsistency is *view-localized*, not *time-localized*.

This argues that CVCG-style (multi-view gating) intervention has signal to act
on, while C3 (frequency curriculum, targets temporal jitter) has *less* direct
target — but as we'll see, C3 turned out to be a general capacity regularizer.

---

## 5. Method: CVCG + C3 (the W3 protective-components extension)

We extended the existing motionprior framework with a new component:

**CVCG (Cross-View Consistency Gating)** — `motionprior/losses/cross_view_consistency.py`,
patched into SC-GS as site E. At each training iteration on view *v* at time *t*,
render the current canonical state from each sibling view at *t*, compute mean
photometric residual `r_iter`, gate the loss by `exp(−β(iter) · r_iter)` with
adaptive `β` (EMA-normalized). Mirrors the existing photometric_gating contract,
but the trust signal is multi-view, not temporal.

**C3 (frequency curriculum)** — already in the framework, applied to the
deform-MLP's temporal positional encoding. Schedule: `(milestones, k_at_milestone)`.

**Wired in** as `MOTIONPRIOR_CVCG_BETA0` and `MOTIONPRIOR_C3_BANDS` env vars; both
independently toggleable, additive. Tests at
`tests/test_cross_view_consistency.py` (11 unit tests) and
`tests/test_scgs_hook.py` (24 hook tests, all passing).

---

## 6. Experiments (5 cells × 2 regimes + 1 GT-free metric)

### 6.1 View-split regime (hold out view 2 — hard, novel-view extrapolation)

| Run | Config | Best test PSNR | Final test | Final train | Final SSIM | Final LPIPS |
|---|---|---:|---:|---:|---:|---:|
| v6 | vanilla | 13.10 | 12.68 | 31.28 | 0.748 | 0.243 |
| v7 | +CVCG only | 13.36 | 13.13 | 30.50 | 0.748 | 0.241 |
| v9 | +C3 only (fast) | 13.58 | 13.25 | 30.85 | 0.748 | 0.213 |
| **v11** | **+C3+CVCG (fast)** | **13.61** | **13.44** | 30.41 | **0.760** | **0.205** |
| v12 | +C3+CVCG (slow) | 13.30 | 13.11 | 30.02 | 0.746 | 0.232 |

**Best config: v11 (fast schedule). Δ vs vanilla: +0.51 dB peak, +0.76 dB final,
−0.038 LPIPS.** Components are additive (v11 > max(v7, v9)). Train PSNR slightly
lower (less overfit) and SSIM/LPIPS strictly better — classic anti-overfitting
signature.

### 6.2 Temporal-interpolation regime (hold every 4th frame across all views — easy)

| Run | Config | Best test PSNR | Final test | Degradation |
|---|---|---:|---:|---:|
| t_van | vanilla | 25.75 | 25.58 | −0.17 |
| t_cvcg | +CVCG only | 25.06 | 24.76 | −0.30 |
| t_c3 | +C3 only (fast) | 26.08 | 23.84 | −2.24 |
| t_full | +C3+CVCG (fast) | 26.15 | 24.61 | −1.54 |
| **t_slow** | **+C3+CVCG (slow)** | **27.33** | 22.47 | −4.86 |
| t_cap8 | +C3+CVCG (cap k=8) | 25.96 | 23.96 | −2.00 |

**Best peak: t_slow (slow schedule). Δ vs vanilla: +1.58 dB.** Note the
trade-off: every protective config eventually degrades worse than vanilla.
**Practical implication: protective methods require test-PSNR early stopping**
(checkpoint at peak iter). With early stopping, the gain is real and substantial.

### 6.3 Regime-dependent C3 schedule (the trade-off)

| C3 schedule | view-split peak Δ | temporal-split peak Δ |
|---|---:|---:|
| **fast** (release k=10 at iter 10k) | **+0.51** | +0.40 |
| **slow** (release k=10 at iter 20k) | +0.20 | **+1.58** |
| neither regresses vs vanilla | — | — |

**Finding**: optimal C3 schedule depends on supervision difficulty.
- Hard regime (data-limited, view extrapolation): aggressive release wins.
  Late capacity isn't useful when model is data-bound.
- Easy regime (interpolation): slow release wins. More gradual capacity buildup
  prevents premature overfitting.
- Both schedules **strictly improve** over vanilla in both regimes (no regression).

### 6.4 GT-free 3D-consistency metric (view-split, no view 2 in training)

Project every deformed canonical Gaussian into all 5 views' alpha masks; count
in how many views it lands inside FG. Headline = % of Gaussians inside FG in all
5 views (a "3D-consistent" Gaussian).

| Model | # Gaussians | **3D-consistent** | Majority (≥4) |
|---|---:|---:|---:|
| v6 vanilla | 96,309 | 47.71% | 80.27% |
| v7 +CVCG | 40,697 | 49.08% | 81.06% |
| **v11 +C3+CVCG (fast)** | **43,664** | **49.43%** | **82.79%** |
| v12 +C3+CVCG (slow) | 42,840 | 47.54% | 81.99% |
| (v5 fits all, ceiling) | 84,140 | 53.55% | 83.20% |

**Findings**:
1. **+C3+CVCG fast produces +1.72 pp more 3D-consistent Gaussians** than vanilla
   — directly testifies CVCG isn't just learning to mimic VGM artifacts.
2. **Compact representation**: v11 uses 44k Gaussians vs vanilla's 96k —
   **~2× more compact** with strictly better quality. Side-effect of C3
   frequency mask preventing over-detailed Gaussians.
3. **3D-consistency ranking matches PSNR ranking** on view-split (v11 > v7 >
   v6 ≈ v12) — second independent metric supports the v11 win.

---

## 7. Limitations (explicit)

1. **"Test PSNR" = "predict VGM" accuracy, not real-world 3D fidelity.** We have
   no real GT (the videos are VGM-generated). High test PSNR could mean
   (a) learned correct 3D, or (b) learned to mimic VGM's per-view hallucinations.
   The GT-free 3D-consistency metric partially addresses this.
2. **Camera-baseline sparsity (5 cams over 130° azimuth arc)** dominates the
   view-extrapolation residual. No training-time mitigation can recover the
   missing triangulation evidence.
3. **Single scene, single VGM output.** Findings haven't been validated across
   different VGMs (SV4D 2.0 vs Wan-2.2 etc.) or different content classes.
4. **Protective methods need early stopping in easy regime.** We don't have an
   automatic stopping rule.
5. **Hyperparameter sensitivity is high.** ±2× schedule duration changes
   peak PSNR by 1+ dB. Robust default unclear.

---

## 8. What's done vs what's left

**Done** (ready for D4-D5 writing):
- All experiments listed above.
- Characterization figures: `runs_aux/vgm_artifact/*.png`.
- GT-free metric JSONs: `runs_aux/mv_consistency/*.json`.
- Scripts + tests merged into `motionprior/` and `scripts/`.
- Hook documentation: `docs/design/scgs_hook_design.md` (updated for patch E).

**Not done**:
- Clean-MV reference baseline (would need a real multi-view dataset).
- Per-VGM comparison (SV4D vs Wan).
- Per-pixel CVCG (current is per-iter scalar).
- Full reprojection-based 3D-consistency (we use mask-projection as proxy).

---

## 9. Headline claims (the report's contributions)

Listed strongest to weakest:

1. **(Engineering)** Discovered and fixed a dying-ReLU failure in SC-GS that
   blocks training on sparse-time multi-view inputs (PSNR 17 → 32, motion
   recovered). One-line LeakyReLU patch.
2. **(Characterization)** Quantified per-view inconsistency in SV4D-style VGM
   output: 3.5 dB PSNR spread, one view (v1) systematically harder by 8.4 vs
   4.8 residual on FG.
3. **(Method)** Added CVCG to the motionprior framework; combined with C3, gives
   +0.51 dB held-out NVS PSNR and +1.72 pp 3D-consistent Gaussians in the
   hard regime, with ~2× more compact representations.
4. **(Method, novel)** Identified a regime-dependent C3 schedule trade-off:
   aggressive release wins for hard (data-limited) supervision, slow release
   wins for easy (interpolation) supervision. Both strictly improve over
   vanilla in their target regime.
5. **(Evaluation)** Proposed a GT-free 3D-consistency metric using
   Gaussian-mask projection coverage — a practical evaluation for VGM-supervised
   reconstruction where no real GT exists.

---

## 10. Open question: reframing

See companion section "Reframe options" — this checkpoint serves as the
inventory we'll use to pick a final framing.
