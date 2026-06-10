# Project Handover — Structure/Motion-Decoupled 4D-GS & VGM-Inconsistency Diagnosis

> Last updated 2026-06-01. Audience: teammate syncing on the project. Read this top-to-bottom once; then use the codebase map + 3-day plan.
> **Most current strategic state lives in `meetTW_checkpoint_0601/`** (deck.md, framework.md, autonomous_session_summary.md). This file is the stable overview.

---

## 0. TL;DR (read this first)

- **Course final project → deliverable is a POSTER** (not a top-conference paper). Scope/contribution can be modest. ~3 days, 2 people.
- We started building a **method**: reconstruct a clean 4D Gaussian scene from a video-generation model's (SV4D 2.0) noisy multi-view video, by **freezing a clean static canonical 3DGS and learning only motion** (part-rigid SE(3) + LBS). Result: **+8.6 dB** over the vanilla SC-GS baseline on the same noisy supervision.
- After PI feedback we **pivoted the framing**: rather than compete with CAT4D on reconstruction quality (we lose — they have a better generator + joint optimization), we use our decoupled setup as an **instrument to diagnose where/how the VGM is spatio-temporally inconsistent**.
- **Two clean results to put on the poster:**
  1. **Reconstruction** (method works): decoupling + part-rigid motion = +8.6 dB, 18× fewer params than the deform-MLP baseline.
  2. **Diagnosis** (the pivot): SV4D has a **"reliability cone"** — fidelity is high near the input view and degrades off-axis (spatially + temporally); and our **fit-residual probe** measures this using our own model, GT-free, reproducing the cone on 2 scenes.

---

## 1. What the project is

**Setting.** Input = one monocular video. An off-the-shelf video generative model (**SV4D 2.0**) turns it into **object-centric multi-view video** (novel views at controllable elevation/azimuth; we use a 57-view grid = 7 elevations × 9 azimuths, 21 frames). These generated views are **spatially/temporally inconsistent** (VGMs aren't physically faithful). We also assume a **clean static canonical 3DGS** of the object at t=0 (in our experiments it comes from a clean D-NeRF/Deformable-3DGS pre-train; in a real deployment it would be a quick static scan).

**Two threads:**
- **(A) Reconstruction method** — freeze the canonical, learn motion → clean, renderable, identity-preserving 4D.
- **(B) Diagnosis (current primary framing)** — use the decoupled setup to *measure* the VGM's inconsistency: where (viewpoint, image region), in what form (spatial / temporal / pose / geometry).

**Why the pivot.** Freezing the canonical loses at reconstruction (structure can't absorb generation noise → fuzzy output) but is exactly what a *measurement instrument* needs (a fixed, known-correct reference). The thing that hurts reconstruction makes us a valid probe. Full argument: `meetTW_checkpoint_0601/formulation_justification.md` and `cat4d_comparison.md`.

---

## 2. Method — reconstruction pipeline (thread A)

Given a frozen canonical 3DGS + SV4D multi-view video, a **5-stage** pipeline. Stages A–D are **preprocessing with zero learnable parameters**; only Stage E learns.

| Stage | What | Output |
|---|---|---|
| **A. Frozen canonical** | A clean static 3DGS (e.g. 114k Gaussians). All attributes `requires_grad=False`. | clean structure, noise-immune |
| **B. Motion mask** | Per-view per-pixel **temporal variance** + Otsu threshold → which pixels move. | `m_v` masks |
| **C. Part assignment** | Project canonical Gaussians into all views, **multi-view vote** by `m_v` → {arm, body, unassigned}. | per-Gaussian part id |
| **D. Arm trajectory** | DLT-triangulate the 2-D motion-mask centroids across views → 3-D motion path (init for Stage E). | `(T,3)` trajectory |
| **E. Motion module (LEARN)** | K=100 K-means clusters on arm Gaussians; per-cluster per-time **SE(3)** (rotation+translation); **LBS** blend over K_lbs=6 nearest clusters; per-time per-cluster 3-D scale; per-Gaussian XYZ residual. ~885K–2.1M params. | deformed 3DGS G(t) |

**Loss (Stage E):** `silhouette (SAM-2) + smart-photo + ARAP (cluster rigidity) + temporal smoothness + trajectory-init`.

**Smart-photometric loss** (the key VGM-supervision trick): per-pixel confidence weight
`w = exp(-α·|I_sv4d − I_ref|)`, then `L = Σ|I_pred − I_sv4d|·w·alpha_mask / Σw`.
- Pixels where SV4D disagrees with a clean reference get ~0 weight (filters hallucination). α=16.
- **IMPORTANT (leakage history):** the original `I_ref` was the d-3dgs render = the eval GT → soft leakage. **Fixed** to the **motion-gated canonical-static render** (Stage A' = render frozen canonical at each view; in static regions compare to it, in moving regions weight=1). Leak-free headline = **20.03 dB** (was 20.40 leaky). Always report leak-free.

**Key design rationale (for the poster):**
- Freeze vs joint: vanilla joint SC-GS = 11.43 dB (geometry explosion); ours frozen = 20.03.
- Structured SE(3)+LBS vs free deform-MLP: same canonical, swap motion module → SC-GS 16M deform-MLP = 11.89 dB vs ours 885K = 20.03 → **+8.14 dB; inductive bias, not DOF, is what matters.**

---

## 3. Method — diagnosis (thread B, current primary)

We characterize SV4D's inconsistency against a clean reference (d-3dgs) and, crucially, with **our own method as the instrument**.

### 3a. Descriptive diagnostics (D-series) — SV4D vs clean d-3dgs
- **D5 spatial fidelity vs viewpoint** → the **"reliability cone"**: PSNR(SV4D, clean) = 37.5 dB at the input azimuth → ~19 dB on the far side (18 dB range); −0.77 dB per 10° elevation.
- **D3 temporal flicker** → 37.5% of provably-static pixels are falsely "moving" in SV4D; 5.7% at input view → up to 57% off-axis. Same azimuthal cone.
- **D6 pose drift** → generated object centroid drifts +0.57 px/° off the requested geometry (matches SV4D 2.0's own stated failure mode).
- **D2/D7 localization** → hallucination is 1.4× stronger at silhouette boundary; error is 81% "wrong appearance", 19% "invented content", 0% "missed".
- **Generality** → the cone replicates on hellwarrior (articulated) — rigid scenes fail spatially, articulated scenes flicker temporally (9.2×).

### 3b. ★ The novel measurement — capacity-controlled fit-residual probe (VALIDATED)
**Idea:** a physically-consistent multi-view video can be explained by (clean static canonical) + (low-DOF physical motion). **VGM inconsistency = the part no physical 4D scene can explain.** Fit our part-rigid model to the VGM views; high per-view fit-residual = physical inconsistency.
- **GT-free:** `R_vgm = |our_render − SV4D|` per view reproduces the cone with **no clean reference at all** (lego_v3 Spearman 0.82, hellwarrior 0.87).
- **Capacity-controlled:** optionally fit the clean reference too (`R_clean`, uniform across views) to confirm the per-view variation is inconsistency, not model-capacity. Floor needed only ONCE, not per measurement.
- **Bypasses registration:** fitting absorbs global misalignment (a naive static-canonical comparison failed here — silhouette IoU 0.28 on lego_v3).
- **Positioning:** complementary to MEt3R (pairwise static, reference-free) and FV4D (aggregate scalar) — ours is *joint multi-view + temporal + physical-motion* consistency, localizable and attributable.

---

## 4. Codebase map (what to run)

Environment: conda env **`scgs`** (`/home/cthsu/miniconda3/envs/scgs/bin/python`). Vendored SC-GS in `third_party/SC-GS`.
**GPU GOTCHA:** GPU 1 on this shared box is contended by another user (renders went 17s/frame vs 0.01s on GPU 0/2). **Always `CUDA_VISIBLE_DEVICES=0` or `2`.** Also set `OMP_NUM_THREADS=6` (24-core box → numpy oversubscribes otherwise). Don't `pkill -f <scriptname>` — it kills your own running jobs.

### Core scripts
| Script | Purpose |
|---|---|
| `scripts/train_partrigid_hier.py` | **Stage E training** (the method). Flags: `--canon_ply --part_dir --scene_dir --v5_render_dir --k_arm 100 --lbs_K 6 --lam_photo_smart 3 --photo_smart_alpha 16 --use_per_time_scale --use_xyz_residual --use_test_too --iterations 8000`. Add `--motion_gated_smart_photo` for leak-free. |
| `scripts/motion_parts_generic.py` | **Stages B+C+D** for any dataset: `--dataset --canon_ply`. Writes `runs_aux/part_assignment_<ds>/`. |
| `scripts/eval_lego_v2_hier.py` | Eval a trained model: `--label --scene --canon_ply [--save_renders]`. Prints PSNR vs SV4D and vs d-3dgs. |
| `scripts/fit_residual_probe.py` | **The diagnosis probe (generic, GPU-native)**: `--scene --canon --label_vgm --label_floor --tag`. Outputs azimuth+elevation cones, corr(gap,raw), per-pixel map → `runs_aux/fit_residual_<tag>.npz`. |
| `scripts/diagnose_vgm_inconsistency.py` | D5 (spatial cone) + D6 (pose drift), SV4D vs d-3dgs. |
| `scripts/diagnose_vgm_temporal_flicker.py` | D3 (temporal flicker in static regions). |
| `scripts/diagnose_vgm_hallucination.py` | D2 (boundary vs interior) + D7 (error-type), `--scene`. |
| `scripts/diagnose_cone_generality.py` | cone replication across scenes. |
| `scripts/render_canonical_57x21.py` | render the canonical at a 57-view grid (for smart-photo ref / static comparison). |
| `scripts/train_scgs_deform_frozen_canon.py` | the F2 fairness baseline (SC-GS deform-MLP on our frozen canonical). |

### Typical end-to-end (reconstruction, one scene)
```bash
# 1. preprocessing (B+C+D)
python scripts/motion_parts_generic.py --dataset lego_v3 --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply
# 2. train (GPU 0)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 python scripts/train_partrigid_hier.py --label lego_v3_A1 \
  --canon_ply <canon.ply> --part_dir runs_aux/part_assignment_lego_v3 --scene_dir data/custom/lego_v3 \
  --v5_render_dir outputs/custom/lego_v3_d3dgs_ref/renders --motion_gated_smart_photo \
  --use_test_too --k_arm 100 --lbs_K 6 --lam_arap 1 --lam_photo_smart 3 --photo_smart_alpha 16 \
  --use_per_time_scale --use_xyz_residual --iterations 8000
# 3. eval
CUDA_VISIBLE_DEVICES=0 python scripts/eval_lego_v2_hier.py --label lego_v3_A1 --scene lego_v3 --canon_ply <canon.ply>
```

### Diagnosis probe (one scene)
```bash
# needs a model fit to SV4D (label_vgm) and one fit to clean d-3dgs (label_floor; train with scene_dir=<ds>_d3dgs_sup)
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=6 python scripts/fit_residual_probe.py \
  --scene lego_v3 --canon <canon.ply> --label_vgm lego_v3_A1 --label_floor lego_v3_d3dgs_floor --tag legov3
```

---

## 5. Data & models on disk

**Datasets** (`data/custom/`): `lego_v2` (5 views), `lego_v3` (57 views), `hellwarrior` (57 views), + `*_d3dgs_sup` variants (d-3dgs renders as supervision, for the floor models).
**Clean GT references** (`outputs/custom/*_d3dgs_ref/renders/`): per-(view,time) renders of an independent Deformable-3DGS trained on clean data — used for eval + diagnosis reference. lego_v2/v3 + hellwarrior.
**Trained models** (`outputs/custom/partrigid_*`):
- `partrigid_lego_v2_A1_leakfree_B` — main reconstruction result (20.03 dB).
- `partrigid_lego_v3_A1`, `partrigid_lego_v3_d3dgs_floor` — diagnosis probe (SV4D-fit + clean-floor).
- `partrigid_hellwarrior_cleancanon_A1`, `partrigid_hellwarrior_cleancanon_floor` — hellwarrior probe.
- canonicals: `outputs/custom/lego_v2_canonical/` (clean, used for lego); hellwarrior clean canonical at `/mnt/HDD_1/cthsu/EV_Final_Project/outputs/hellwarrior_scgs_default_node/`.
**Diagnosis results**: `runs_aux/fit_residual_{legov3,hellwarrior}.npz` (+ pixmaps).
**Figures + all strategy docs**: `meetTW_checkpoint_0601/figs/` and `meetTW_checkpoint_0601/*.md`.

---

## 6. Current results (numbers for the poster)

**Reconstruction (lego_v2, vs clean d-3dgs GT):**
| Method | PSNR | DOF |
|---|---:|---:|
| Vanilla SC-GS (joint, 16M deform-MLP) | 11.43 | 16M |
| SC-GS deform-MLP + our frozen canonical (F2) | 11.89 | 16M |
| **Ours (frozen canon + part-rigid, leak-free)** | **20.03** | 885K |
| Oracle ceiling (ours trained on clean GT) | 20.96 | 885K |
| Ours, digger-only (baseplate excluded) | 21.12 | — |

**Diagnosis (SV4D 2.0):**
- Reliability cone: 37.5 dB (input azimuth) → 19.4 dB (far side); −0.77 dB/10° elevation.
- Temporal flicker: 37.5% of static pixels falsely moving; 5.7%→57% off-axis.
- Fit-residual probe (our instrument, GT-free) reproduces the cone: lego_v3 ρ=0.82, hellwarrior ρ=0.87.

---

## 7. What's DONE vs what's LEFT

**Done:**
- Reconstruction method + full ablation (decouple, fairness F1/F2, ablation panel, ceiling, baseplate artefact).
- Leak fix (smart-photo → motion-gated canonical).
- Diagnosis D2/D3/D5/D6/D7 + generality (lego_v3 + hellwarrior).
- Validated GT-free fit-residual measurement probe.
- All figures + the PI deck (`meetTW_checkpoint_0601/deck.md`, 7 slides).

**Left (to make it poster-complete):**
1. **Standard metrics** on the reconstruction comparison: add LPIPS/SSIM (have some) + ideally **DreamSim** and report **vs-SV4D AND vs-clean** ("overfit gap"). PSNR alone is known-unreliable (rewards blur).
2. **MEt3R baseline** (optional, nice-to-have): reference-free multi-view-consistency metric (CVPR'25) as a comparison point next to our fit-residual. Needs DUSt3R setup (install to `/mnt/HDD_1`, not `/home` which is 99% full).
3. **Temporal axis of the fit-residual probe** (optional): currently spatial; add per-frame-Δ residual to show it also tracks flicker.
4. **Poster assembly**: method figure (use `slide2_architecture.drawio`), reconstruction comparison (gallery GIFs / keyframes), diagnosis cone figure (`fit_residual_generality.png`, `vgm_inconsistency_curves.png`).
5. **Lock the narrative**: poster = "Structure/Motion Decoupled 4D-GS, used as a probe to diagnose VGM inconsistency." Reconstruction is the method; diagnosis is the analysis.

**Known caveats (be honest on the poster):**
- Diagnosis correlations are n=9 points/scene, moderate-strong (0.67–0.88).
- The clean canonical assumption; hellwarrior's canonical is pose-mismatched (reconstruction poor there, but the probe still works since fitting absorbs it).
- Single generator (SV4D 2.0); cross-generator is future work (`meetTW_checkpoint_0601/benchmark_plan.md`).

---

## 8. 3-day plan, 2 people (poster scope)

> Split: **P1 = reconstruction/method track**, **P2 = diagnosis track**. Day 3 both on poster. Keep it modest — this is a course poster.

### Day 1 — lock results + fill metric gaps
- **P1**: Add LPIPS + SSIM + DreamSim to `eval_lego_v2_hier.py`; produce the final reconstruction table (vanilla vs ours vs ceiling) **with the overfit gap** (vs-SV4D vs vs-clean). Regenerate the gallery / keyframe figures cleanly. (GPU 0)
- **P2**: Re-confirm the diagnosis figures are final (cone, flicker, fit-residual generality). Add the per-pixel localization figure for hellwarrior too. Write the diagnosis half of the poster text. (GPU 2)
- **Both, EOD**: agree the poster outline (sections + which figures).

### Day 2 — one "completeness" experiment each + draft poster
- **P1**: Pick ONE of: (a) DreamSim/LPIPS-based comparison framed as "ours denoises (small overfit gap), vanilla overfits (large gap)"; (b) a second reconstruction scene cleanly. Finish the method + ablation figures.
- **P2**: Pick ONE of: (a) temporal axis of the fit-residual probe; (b) MEt3R baseline if setup is smooth (timebox to half a day — if DUSt3R install fights you, drop it). Finish the diagnosis figures.
- **Both, EOD**: first full poster draft (all figures placed, captions written).

### Day 3 — assemble + polish
- Morning: integrate, tighten text, make the method schematic clean (export `slide2_architecture.drawio` → PNG via app.diagrams.net — no CLI on the box).
- Afternoon: internal review pass, fix numbers/captions, print check.

### Poster structure (suggested)
1. **Problem/Setting** — VGM gives noisy multi-view video; we reconstruct clean 4D and diagnose the VGM.
2. **Method** — 5-stage decoupled pipeline schematic.
3. **Reconstruction result** — table (+8.6 dB, 18× fewer params) + gallery comparison.
4. **Diagnosis** — the reliability cone (spatial + temporal) + the fit-residual probe (our GT-free instrument).
5. **Takeaways + limitations**.

### Do-NOT (scope guard)
- Don't chase cross-generator benchmarking (too big for 3 days).
- Don't try to beat CAT4D on reconstruction quality (wrong battle).
- Don't add new architecture features — the method is frozen; this is a writing+packaging sprint.
