# Experiment Design — MotionPrior-4DGS

> Owner: Murphy. Last updated: 2026-05-11.
> Positioning: articulation-aware ARAP as headline contribution; see [Survey](../../MP_Obsidian_Notes/wiki/research/Image_to_4D_Survey_May2026.md) for landscape and [MotionPrior4DGS wiki page](../../MP_Obsidian_Notes/wiki/research/MotionPrior4DGS.md) for full method.

---

## 1. The load-bearing experiment

**Q:** Does articulation-aware piecewise-rigid ARAP recover joint kinematics that uniform ARAP smears out?

**Why this is the headline:** every monocular → 4DGS method we surveyed applies uniform local-rigidity (or none). 4D-Fly (CVPR 2025) names articulated motion as a failure mode by name. If we win this comparison on D-NeRF articulated synthetic scenes, the paper writes itself. If we lose, the project pivots to Option B/C from the survey.

**Setup:** D-NeRF `jumpingjacks` and `hellwarrior` (both have explicit articulated skeletons in their synthetic models). Train SC-GS-default + our articulation-aware variant from the same starting state. Compare on held-out novel views and times.

**Decision rule:** if **inter-part angular consistency** (defined §5) improves by ≥ 0.5 std on `jumpingjacks` while PSNR/SSIM/LPIPS hold within ±0.1 dB / 0.005 / 0.005, articulation is validated as the contribution. Below that bar, fall back to selective-trust framing and bet on the gating signal differentiation vs ViDAR.

**Compute:** 8 runs × ~2 GPU-hr = 16 H100-hr. Cheap enough that we can re-run if anything looks off.

---

## 2. Datasets

| Dataset | # scenes | Type | Why | Input format |
|---|---|---|---|---|
| **D-NeRF** | 8 synthetic | known camera + GT skeleton | Headline articulation benchmark; clean | Multi-view rendered video |
| **D-NeRF articulated subset** | 4 (`jumpingjacks`, `hellwarrior`, `bouncingballs`, `standup`) | strong articulation | The discriminating subset | Use first-frame-only or full video per scene |
| **DyCheck (iPhone)** | 7 real | monocular handheld | ViDAR's primary benchmark — we must report on it | Monocular video + GT held-out view |
| **HyperNeRF** | 5 (vrig subset) | real-world deformable | Standard secondary; tests generalization | Monocular video |
| **DAVIS articulated subset** | 5 (TBD: pick on day 1 — `dog`, `dance-twirl`, others) | real, no held-out GT | Qualitative-only; covers in-the-wild articulation | Monocular video |

**Scope rule (1-month deadline):** D-NeRF + DyCheck are mandatory. HyperNeRF is mandatory if Week 4 timeline holds. DAVIS qualitative is week 5 nice-to-have.

**Total scene budget:** 8 D-NeRF + 7 DyCheck + 5 HyperNeRF = **20 scenes** quantitative. With 6 ablation rows: 120 runs at ~2 GPU-hr each = **240 H100-hr**. Within budget (rented H100 at $2/hr ≈ $480 total — feasible).

---

## 3. Ablation matrix (the discriminating table)

The matrix the paper must contain. Each row is a full training run; each scene is a column.

| # | Variant | Gating | Curriculum | Articulation ARAP | Rest L2 | Backbone |
|---|---|---|---|---|---|---|
| 1 | **SC-GS default** | — | — | uniform | — | SC-GS |
| 2 | **MoSca** | — | classical priors | scaffold | — | MoSca |
| 3 | **Shape of Motion** | — | — | SE(3) bases (soft) | — | SoM |
| 4 | **ViDAR** | dyn-region mask | — | uniform | — | ViDAR |
| 5 | + Articulation only | — | — | **piecewise-rigid** | — | SC-GS |
| 6 | + Gating only | **ARAP-energy** | — | uniform | — | SC-GS |
| 7 | + Curriculum only | — | **freq-band** | uniform | — | SC-GS |
| 8 | + Gating + Curriculum | ✓ | ✓ | uniform | small | SC-GS |
| 9 | **Ours (full)** | ✓ | ✓ | **piecewise-rigid** | small | SC-GS |
| 10 | Ours − articulation | ✓ | ✓ | uniform | small | SC-GS |

Rows 1–4 are baselines (off-the-shelf implementations). Rows 5–10 are our method.

**Critical pair:** rows 9 vs 10 isolate the articulation contribution. **Rows 5 vs 1 isolate articulation alone.** Rows 6 and 7 isolate gating and curriculum individually.

**Mandatory comparison:** row 9 vs row 4 (Ours vs ViDAR) on DyCheck. This is the head-to-head against the closest prior work.

---

## 4. Metrics

### Mandatory (every run reports)
- **PSNR / SSIM / LPIPS** on held-out novel views and held-out time. Standard.
- **Inter-part angular consistency** (new, §5). Headline articulation metric.
- **Floater count** — Gaussians outside the convex hull of COLMAP point cloud by margin > X. Define X day 1.
- **Training time + memory** — for fair speed comparison.

### Diagnostic (logged, only highlighted when relevant)
- **Per-frame ARAP energy** (our gating signal) over training — visualize what's being gated.
- **Frequency-band activation** over training — visualize what curriculum unlocks when.
- **Part-confidence histogram** — for scenes where SAM 2 segmentation is borderline.

### Inter-part angular consistency — formal definition

For a scene with K segmented parts, at each frame t:
1. Per part k, collect deformed Gaussian positions `{x_i^t : part(i) = k}`.
2. Compute the principal axis `a_k^t = first eigenvector of cov({x_i^t})`.
3. For each pair (k, k') of adjacent parts, compute the angle `θ_{k,k'}^t = arccos(a_k^t · a_{k'}^t)`.
4. Smoothness score: `1 - std_t(d θ_{k,k'}/dt) / mean_t(|d θ_{k,k'}/dt|)`. Range [0, 1].

A piecewise-rigid joint produces smooth angular trajectories (smoothness → 1). An elastic-bend reconstruction produces jittery, non-monotonic angles (smoothness → 0).

**Implementation:** `motionprior/metrics/articulation.py`. CPU-testable. Built in Task 21.

---

## 5. Compute budget

| Resource | Spend | Where |
|---|---|---|
| H100 hours | ~240 | Main ablation table (120 runs × 2 hr) |
| H100 hours | ~60 | Baseline reproductions (ViDAR, MoSca on shared scenes) |
| H100 hours | ~40 | Debugging, failed runs, re-runs |
| **Total H100** | **~340 hr** | ≈ $680 at RunPod spot pricing |
| RTX 6000 Ada (if needed) | optional | Wan-2.2 if single-image input is added |
| Storage | ~200 GB | Datasets + checkpoints + supervision videos |

**Headroom:** budgeting 30% buffer. If we go over, drop HyperNeRF or DAVIS first.

---

## 6. Timeline (4 weeks remaining from 2026-05-11)

| Week | Goal | Gate |
|---|---|---|
| **W1 (2026-05-12 to 2026-05-18)** | SC-GS reproduced on D-NeRF; articulation ARAP plugged in; first jumpingjacks run | Row 1 vs Row 5 result on jumpingjacks |
| **W2 (2026-05-19 to 2026-05-25)** | Gating + curriculum + rest-state hooked; full ablation matrix on D-NeRF synth | Rows 5–10 on all 8 D-NeRF scenes |
| **W3 (2026-05-26 to 2026-06-01)** | DyCheck pipeline; ViDAR baseline reproduction; ours vs ViDAR head-to-head | Row 9 vs Row 4 numbers on DyCheck |
| **W4 (2026-06-02 to 2026-06-08)** | HyperNeRF extension; writing; figures; final ablation polish | Submission-ready draft |

**Scope-cut decisions baked in:**
- If articulation ARAP underperforms uniform ARAP on jumpingjacks by end of W1: pivot to Option B (single-image + DIFF4SPLAT comparison) or Option C (refinement-of-feed-forward).
- If ViDAR reproduction doesn't work by mid-W3: ship the D-NeRF + HyperNeRF results without head-to-head, note ViDAR comparison as future work.
- If HyperNeRF runs slip past W4 start: drop them from the submission, keep as supplementary.

---

## 7. Per-week deliverable definition

**W1 success criteria** (the riskiest week — if this fails, the whole project pivots):
- `motionprior.integration.scgs_hook` imports and runs inside SC-GS's training loop without crashing
- D-NeRF `jumpingjacks` trains end-to-end with SC-GS-default and produces sane PSNR (within 1 dB of paper)
- Same scene trains with `articulation_aware_arap=True` and produces a different deformation field (visible in saved gaussians)
- Inter-part angular consistency metric runs on both outputs and produces a number
- One side-by-side rendered video for visual comparison

**W2 success criteria:**
- Full row 9 of the ablation table runs on all 8 D-NeRF scenes
- Rows 5, 8, 10 also run (the critical ablation isolations)
- Results spreadsheet auto-populated by an eval script
- Loss curves logged to wandb for all runs

**W3 success criteria:**
- DyCheck data preprocessed (camera poses extracted, monocular video aligned)
- ViDAR repo cloned, conda env built, reproduces their DyCheck PSNR-m within 0.5 dB
- Ours on DyCheck reported with same metric protocol as ViDAR

**W4 success criteria:**
- All quantitative tables filled
- All qualitative figures rendered
- Method section drafted (using wiki MotionPrior4DGS.md as basis)
- Related work section drafted (using survey as basis)
- Submission-ready by 2026-06-08

---

## 8. Baselines — implementation plan

| Baseline | Source | Effort | Owner |
|---|---|---|---|
| SC-GS default | github.com/yihua7/SC-GS | install, run with config — ~½ day | Member B |
| MoSca | github.com/JiahuiLei/MoSca | install, run on D-NeRF + DyCheck — ~1 day | Member B |
| Shape of Motion | github.com/vye16/shape-of-motion | install, run — ~½ day | Member B |
| ViDAR | github.com/vidar-4d/ViDAR | install, DyCheck reproduction — ~2 days | Member B |
| 4D-Fly | TBD whether code released | optional, skip if blocked | — |
| USplat4D | TBD whether code released | optional, skip if blocked | — |

If any baseline reproduction blocks for >3 days, switch to reporting their *published* numbers and label as "ours numbers vs paper-reported." Reviewers accept this if the dataset/metric protocol is the same.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Articulation ARAP doesn't help — joints are too small | Medium | W1 gate; pivot to Option B/C if so |
| SAM 2 part segmentation is unreliable on real DyCheck scenes | Medium-high | Hierarchical SAM 2 + manual prompts for the 7 scenes; if still unreliable, fall back to whole-object segmentation on DyCheck and lean D-NeRF for articulation results |
| ViDAR doesn't reproduce in our env | High | Use their reported numbers as comparison; document protocol differences |
| H100 budget overrun | Low | Drop HyperNeRF/DAVIS first |
| SC-GS rasterizer breaks on RunPod H100 (CUDA version mismatch) | Medium | Pin SC-GS CUDA 11.8 env; have AnySplat env as fallback |
| Frequency curriculum's PE assumption doesn't match SC-GS's actual PE | Medium | W1 first action: confirm SC-GS's temporal encoding by reading source. Fix with explicit Fourier basis if needed |

---

## 10. Decisions still open (block W1 start)

1. **Front-end input**: monocular video (assumed in this doc) vs single-image + SV4D 2.0. Lean: monocular video for simplicity and ViDAR direct comparison. Decide by EOD 2026-05-12.
2. **D-NeRF input mode**: full multi-view rendered video (uses all cameras), or first-camera-only (simulates monocular)? The "fair vs ViDAR" choice is monocular. The "fair vs SC-GS" choice is multi-view. Decide alongside #1.
3. **Floater margin X**: settle in W1 day 1 (look at a few SC-GS-default renders, pick a sane threshold).
4. **wandb vs tensorboard** for experiment tracking: wandb (recommended; better team viewing).
5. **Adjacency for inter-part angular consistency metric**: how to derive (k, k') adjacency? Options: (a) all pairs; (b) skeleton-based (D-NeRF has it); (c) spatial-proximity-based. Default (c) for generality, override with (b) when GT skeleton is available.

---

## 11. Reproducibility

Every entry in the ablation table must come with:
- A YAML config file under `motionprior/configs/experiments/` (one per row × scene)
- A wandb run ID stored in `outputs/runs.csv`
- A checkpoint at `checkpoints/{config_name}/last.pt`
- Rendered novel-view + novel-time MP4 at `outputs/{config_name}/render.mp4`

`scripts/run_all_ablations.sh` reads `outputs/runs.csv`, iterates missing rows, launches them. Idempotent.
