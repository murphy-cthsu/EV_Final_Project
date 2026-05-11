# Experiment Design — MotionPrior-4DGS

> Owner: Murphy. Last updated: 2026-05-12 (Option C: VGM-supervised articulation-aware 4DGS from single image; VWM perception module framing).
> Positioning: articulation × video-diffusion-supervision × single-image input. See [`docs/vwm_framing.md`](vwm_framing.md), [Survey](../../MP_Obsidian_Notes/wiki/research/Image_to_4D_Survey_May2026.md), [MotionPrior4DGS wiki](../../MP_Obsidian_Notes/wiki/research/MotionPrior4DGS.md).

> **[2026-05-12 update]** Articulation-cluster audit found that RigGS (CVPR 2025) already does ARAP + articulated + monocular video; VideoArtGS (Sep 2025) does articulated monocular 4D with hybrid center-grid parts. Surviving novelty is the intersection: articulation + diffusion supervision + single-image input. Baseline list updated; experiment matrix below adds RigGS as a critical comparison.

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

## 5. Compute budget — two-tier (2026-05-12 update)

**Lab A4500 × 3 server** (free; default for dev + W1/W2):

| Workload | Estimate |
|---|---|
| Day-1 setup + SC-GS smoke + patch verification | ~4 hr (interactive) |
| W1: jumpingjacks ablation gate (rows 1, 5, 9 × 1 scene) | ~6 hr (3 runs × 2 hr) |
| W2: D-NeRF articulated subset (rows 1, 5, 6, 7, 8, 9, 10 × 4 scenes = 28 runs) | ~56 hr wall clock (run 3 in parallel across 3 cards → ~20 hr) |
| Baseline reproductions: RigGS, MoSca, Shape of Motion on D-NeRF | ~12 hr total |
| **Lab total** | **~50 hr active wall time, free** |

**RunPod H100** (rented; W3-W4 scaling):

| Workload | Estimate | $ |
|---|---|---|
| DyCheck + HyperNeRF runs (rows 1, 5, 9 × 12 scenes = 36 runs × 2 hr) | ~72 H100-hr | ~$145 |
| ViDAR baseline reproduction on DyCheck | ~16 H100-hr | ~$32 |
| Full ablation completion (rows 6, 7, 8, 10 on real scenes if needed) | ~48 H100-hr | ~$96 |
| Debug + re-run buffer (~30%) | ~40 H100-hr | ~$80 |
| **RunPod total** | **~176 H100-hr** | **~$355** |

**Combined: ~$355 cash + ~50 hr lab box time**, vs. ~$680 if RunPod were the default. The lab box pays for the dev phase entirely.

Storage: ~200 GB on lab box (datasets + per-run checkpoints + supervision videos); ~50 GB on RunPod (W3-W4 outputs only).

**Headroom:** budgeting 30% buffer on RunPod. If we go over, drop HyperNeRF or DAVIS first.

---

## 6. Timeline (4 weeks remaining from 2026-05-11)

| Week | Goal | Gate |
|---|---|---|
| **W1 (2026-05-12 to 2026-05-18)** — **Lab A4500** | SC-GS reproduced; articulation ARAP plugged in; first jumpingjacks run | Row 1 vs Row 5 on jumpingjacks (free iteration; cheap to re-run) |
| **W2 (2026-05-19 to 2026-05-25)** — **Lab A4500** | Gating + curriculum + rest-state hooked; full D-NeRF ablation matrix; RigGS/MoSca/SoM baselines | Rows 5-10 on all 8 D-NeRF scenes (~50 lab-hr; 3-way parallel on the 3 cards) |
| **W3 (2026-05-26 to 2026-06-01)** — **RunPod H100** | DyCheck pipeline; ViDAR head-to-head; simulator-import demo (Genesis/PyBullet) | Row 9 vs Row 4 on DyCheck; URDF runs in Genesis IK probe |
| **W4 (2026-06-02 to 2026-06-08)** — **Mix** | HyperNeRF extension; writing; figures; final ablation polish | Submission-ready draft |

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

| Baseline | Source | Effort | Owner | Required? |
|---|---|---|---|---|
| SC-GS default | github.com/yihua7/SC-GS | install, run with config — ~½ day | Member B | yes |
| **RigGS** (CVPR 2025) — articulation comparator | search arxiv 2503.16822 for code | ~2 days; if no code, cite their published D-NeRF PSNR 40.82 | Member B | **yes — direct articulation competitor** |
| MoSca | github.com/JiahuiLei/MoSca | install, run on D-NeRF + DyCheck — ~1 day | Member B | yes |
| Shape of Motion | github.com/vye16/shape-of-motion | install, run — ~½ day | Member B | yes |
| ViDAR | github.com/vidar-4d/ViDAR | install, DyCheck reproduction — ~2 days | Member B | yes — direct diffusion-supervision competitor |
| DIFF4SPLAT | TBD (CVPR 2026; check release) | ~1 day if code; else cite | Member B | yes — single-image-input competitor |
| VideoArtGS | TBD | optional; PartNet-Mobility protocol differs | — | no |
| 4D-Fly | TBD | optional, skip if blocked | — | no |
| USplat4D | TBD (ICLR 2026; check release) | optional, skip if blocked | — | no |

The new mandatory: **RigGS**. Without RigGS as a baseline, our articulation claim is not supportable — they did articulation + monocular video + ARAP before us. Get their D-NeRF numbers (published, not reproduced) at minimum.

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

1. **Front-end input** — **DECIDED 2026-05-12**: single-image input + VGM front-end (Option C). Primary VGM: SV4D 2.0 (commercial license, multi-view-aware). Fallback: Wan-2.2 I2V. The articulation × diffusion-supervision × single-image intersection is the surviving novelty after the audit.
2. **D-NeRF input mode**: use the GT first frame as the static input; run SV4D 2.0 to generate the supervision video; held-out novel-view + novel-time evaluation against D-NeRF's GT rendering. This matches the single-image-input deployment regime.
3. **Floater margin X**: settle W1 day 1 (look at a few SC-GS-default renders).
4. **wandb** for experiment tracking.
5. **Adjacency for inter-part angular consistency**: spatial-proximity-based by default; skeleton-based when D-NeRF GT skeleton is available.
6. **Simulator-import demo target**: Genesis (recommended — modern, fast, articulated bodies) or PyBullet (more mature). Pick before W4.
7. **SV4D 2.0 license terms**: verify commercial / research-only / NeRF-acceptable. If blocked, fall back to Wan-2.2 I2V (Apache-2.0) plus a multi-view step (AnySplat does the lifting).

---

## 11. Reproducibility

Every entry in the ablation table must come with:
- A YAML config file under `motionprior/configs/experiments/` (one per row × scene)
- A wandb run ID stored in `outputs/runs.csv`
- A checkpoint at `checkpoints/{config_name}/last.pt`
- Rendered novel-view + novel-time MP4 at `outputs/{config_name}/render.mp4`

`scripts/run_all_ablations.sh` reads `outputs/runs.csv`, iterates missing rows, launches them. Idempotent.
