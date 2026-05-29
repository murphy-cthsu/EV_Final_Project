# Handoff — MotionPrior-4DGS, GPU session

> Cold-start guide for a fresh Claude Code session running on GPU hardware.
> Read top to bottom; everything you need is here or linked.
> Last updated: 2026-05-12.

## 0. Two-tier GPU workflow

The project uses two distinct GPU environments. Match the work to the right one:

| Environment | Use for | Cost |
|---|---|---|
| **Lab A4500 × 3** (60 GB total, default dev box) | Day-1 setup, SC-GS smoke, all four hook components, single-scene experiments, W1/W2 ablation rows, baseline reproductions (RigGS, MoSca, Shape of Motion), iteration on the patch file | Free (lab infra) |
| **RunPod H100** (80 GB single card, rented) | W3/W4 full ablation sweep at scale, longer training runs, large-VRAM VGM at higher resolution, final paper-quality results | ~$2/hr (rented as needed) |

**Default to the lab box.** Move to RunPod only when (a) a run blocks for >12 hours of A4500 time, or (b) you need to parallelize 4+ runs simultaneously and the lab is contended. The first 2 weeks of `docs/design/experiments.md` are entirely doable on the lab box.

**Same setup script** works on both — `scripts/setup_runpod.sh` (badly named; works on any Ubuntu+CUDA GPU box including the lab one). Don't be surprised the name says "runpod"; rename it later if it bothers you.

## 1. The project in three sentences

We build a **perception module for visual world models**: from a single static image, produce an articulation-aware 4D Gaussian Splatting scene that a downstream physics simulator can ingest for IK / contact / manipulation reasoning. The contribution is the intersection of **diffusion-supervised 4D reconstruction** (ViDAR, 4D-Fly, DIFF4SPLAT) and **articulated 4D Gaussian reconstruction** (RigGS, VideoArtGS, Part2GS) — a corner the May-2026 landscape audit confirmed is unoccupied. Three components: SAM-2-grounded piecewise-rigid ARAP, ARAP-energy-based supervision gating, and a frequency-domain motion curriculum.

For the full positioning argument: `docs/design/vwm_framing.md`. For the experiment plan: `docs/design/experiments.md`. For the SC-GS integration design: `docs/design/scgs_hook_design.md`.

## 2. What's already shipped (and tested on CPU)

| Component | Path | Tests |
|---|---|---|
| Physically-gated supervision weights + adaptive α | `motionprior/losses/gating.py` | 8 |
| Articulation-aware per-edge ARAP weights | `motionprior/losses/arap_articulated.py` | 6 |
| Rest-state L2 anchor | `motionprior/losses/rest_state.py` | 5 |
| Frequency-domain curriculum (Fourier-band schedule) | `motionprior/curriculum/frequency.py` | 8 |
| Offline ARAP-prior energy computation | `motionprior/geometry/arap_prior.py` | 6 |
| RAFT-flow → 3D trajectory lifting | `motionprior/geometry/flow_lifting.py` | 11 |
| SAM 2 wrapper + part-label assignment | `motionprior/segmentation/parts.py` | 5 |
| Inter-part angular consistency metric (headline) | `motionprior/metrics/articulation.py` | 10 |
| SC-GS hook (3 patch sites, graceful no-op fallback) | `motionprior/integration/scgs_hook.py` | 12 |
| SC-GS ARAP adapter (monkey-patch ControlNodeWarp.arap_loss) | `motionprior/integration/scgs_arap_adapter.py` | 6 |
| VGM front-end wrappers (SV4D 2.0 primary, Wan-2.2 fallback) | `motionprior/integration/vgm.py` | 8 |
| Simulator-import bridge (URDF + Genesis YAML) | `motionprior/integration/sim_bridge.py` | 7 |
| YAML config loader | `motionprior/configs/__init__.py` | 3 |
| Training entry CLI | `scripts/train.py` | (CPU smoke) |
| Eval script (image + articulation + URDF emit) | `scripts/eval.py` | (CPU smoke) |
| Third-party bootstrap | `scripts/bootstrap_third_party.sh` | — |
| RunPod env setup | `scripts/setup_runpod.sh` | — |
| D-NeRF download | `scripts/download_dnerf.sh` | — |
| SC-GS patch file (3 sites) | `scripts/patches/scgs_hook.patch` | — |

**Total: 95+ tests passing on CPU** as of 2026-05-12.

## 3. Hardware notes

### Lab A4500 × 3 (default dev box; 60 GB total)

| Model | Single A4500 fit? | Lab box (3 cards) plan |
|---|---|---|
| SC-GS training | ✓ yes | GPU 2 (long-running) |
| AnySplat inference | ✓ yes | GPU 1 |
| SAM 2 (hiera-large) | ✓ yes (~6 GB) | shares GPU 1 with AnySplat |
| **SV4D 2.0** at fp16 | ✓ yes (likely; verify day 1) | GPU 0 |
| **Wan-2.2 I2V-A14B** at fp16 | ✗ single-card; needs 2 cards | GPU 0 + GPU 1 via `accelerate device_map='auto'` (this competes with AnySplat; sequence the pipeline rather than running concurrently) |

Recommended use:
- **SV4D 2.0 first** (smaller; fp16 fits on one A4500). Keeps the 3-GPU split clean.
- **Wan-2.2 only if SV4D 2.0 license blocks**. Then run VGM and AnySplat sequentially, not concurrently.

The pipeline is naturally sequential per scene (VGM → AnySplat → SAM 2 → SC-GS), so all four models don't need to be resident simultaneously. The 3 cards just mean we can have 2-3 scenes in different pipeline stages at once.

### RunPod H100 (W3-W4 scaling)

Single 80 GB H100 fits every model comfortably with no quantization gymnastics. Use it when:
- Running the full ablation matrix (120+ runs) — 3-way parallelism on the lab box becomes a bottleneck around the 4th simultaneous training run
- Reproducing ViDAR on DyCheck (their training was done on 8×A100; H100 single-card halves the wall time)
- Final paper renders + figures

Cost estimate per `docs/design/experiments.md`: ~$200-300 on RunPod if we use the lab box for W1/W2, vs. ~$680 originally budgeted assuming everything ran on RunPod.

## 4. The first day on the lab A4500 box (W1 of the experiment timeline)

In strict order:

### 4.1 Environment bootstrap
```bash
cd ~/EV_Final_Project   # (after `git clone https://github.com/murphy-cthsu/EV_Final_Project.git`)
bash scripts/bootstrap_third_party.sh        # clones SC-GS @ 3a9d2ad4, AnySplat @ 5f5e208a
bash scripts/setup_runpod.sh                 # builds 3 conda envs (name notwithstanding, works on lab box)
```

### 4.2 Verify SC-GS source structure (~30 min)
The hook design assumes specific line numbers in `third_party/SC-GS/train_gui.py` and method names in `utils/time_utils.py`. Verify:

```bash
grep -n "Ll1 = l1_loss\|loss = loss + self.deform.reg_loss\|time_input + ast_noise" \
    third_party/SC-GS/train_gui.py
grep -n "def arap_loss\|cal_arap_error" third_party/SC-GS/utils/time_utils.py
```

Expected from the 2026-05-12 read:
- `train_gui.py:1086` — `time_input = fid.unsqueeze(0).expand(N, -1)` (insertion site A)
- `train_gui.py:1088` — `self.deform.step(...)` call
- `train_gui.py:1118` — `Ll1 = l1_loss(image, gt_image)` (replace site B)
- `train_gui.py:1123` — `loss = loss + self.deform.reg_loss` (after-insertion site C)
- `utils/time_utils.py:1080` — `def arap_loss(self, t=None, delta_t=0.05, t_samp_num=2):` (our adapter monkey-patches this)

If any of these have moved: regenerate `scripts/patches/scgs_hook.patch` and update `docs/design/scgs_hook_design.md`.

### 4.3 SC-GS smoke run (~1 hour)
Reproduce SC-GS-default on D-NeRF `jumpingjacks` with no MotionPrior modifications to confirm the env works:

```bash
bash scripts/download_dnerf.sh
conda activate scgs
cd third_party/SC-GS
# Their default training command — see their README
python train_gui.py --source_path ../../data/dnerf/jumpingjacks --eval --is_blender ...
```

Target: PSNR ~43 on jumpingjacks. If you don't get that, fix the env *before* touching the hook.

### 4.4 Apply the patch (~30 min)
```bash
cd third_party/SC-GS
git apply --check ../../scripts/patches/scgs_hook.patch
git apply ../../scripts/patches/scgs_hook.patch
```

If `--check` fails, the line numbers drifted — open the patch, fix offsets, re-apply.

Then add the hook construction lines to `Trainer.__init__` (item D in the patch file — manual, ~10 lines, see hook_design.md for the template).

### 4.5 Run the SC-GS-default ablation row through our hook (~1 hour)
```bash
conda activate scgs
python scripts/train.py \
    --config scenes/dnerf_jumpingjacks \
    --scene_root data/dnerf/jumpingjacks \
    --output_dir outputs/jumpingjacks_scgs_default \
    --ablation scgs_default \
    --iterations 30000
```

`--ablation scgs_default` turns off all four components. The result should match the SC-GS-default smoke run from step 4.3 — the hook degrades gracefully.

### 4.6 First articulation result (~2 hours)
```bash
python scripts/train.py \
    --config scenes/dnerf_jumpingjacks \
    --scene_root data/dnerf/jumpingjacks \
    --output_dir outputs/jumpingjacks_art_only \
    --ablation articulation_only \
    --iterations 30000 \
    --part_labels_path runs_aux/jumpingjacks_part_labels.pt
```

(`part_labels.pt` requires SAM 2 to have run first — see step 4.7.)

### 4.7 SAM 2 part segmentation on jumpingjacks (~30 min)
```bash
conda activate sam2   # or motionprior env if SAM 2 was installed there
python -c "
from motionprior.segmentation.parts import SAM2Segmenter, assign_part_labels
import numpy as np, torch, imageio.v3 as iio
seg = SAM2Segmenter(checkpoint_path='checkpoints/sam2_hiera_large.pt')
img = iio.imread('data/dnerf/jumpingjacks/train/r_000.png')
label_map = seg.segment(img)
# Project to Gaussian-level: requires the canonical 3DGS from AnySplat first.
# For now, save the 2D label map; convert to per-Gaussian labels after AnySplat run.
torch.save(torch.from_numpy(label_map), 'runs_aux/jumpingjacks_label_map_2d.pt')
"
```

### 4.8 Compare and report
```bash
python scripts/eval.py \
    --run_dir outputs/jumpingjacks_scgs_default \
    --scene_root data/dnerf/jumpingjacks \
    --gaussian_trajectory_path outputs/jumpingjacks_scgs_default/positions.pt \
    --part_labels_path runs_aux/jumpingjacks_part_labels.pt \
    --emit_urdf
# Same for jumpingjacks_art_only.
# Diff the two eval.json files. Inter-part angular consistency should be
# noticeably higher for art_only (the W1 gate per docs/design/experiments.md).
```

## 5. Open decisions you need to make on the pod

| Decision | When | What's at stake |
|---|---|---|
| SV4D 2.0 vs Wan-2.2 as VGM front-end | Day 1 | License + VRAM. SV4D 2.0 first; Wan-2.2 fallback. |
| Genesis vs PyBullet vs MuJoCo for the sim-import demo | Before W4 | Genesis recommended (newest, articulated-body-native, fastest). |
| Floater margin X | Day 1 W1 | Run SC-GS-default once, look at outputs, pick a sane threshold. |
| Patch-file line offsets | Day 1 | `git apply --check` will tell you immediately if SC-GS drifted. |

## 6. Critical context you must NOT lose

### 6.1 The articulation novelty is narrow
RigGS (CVPR 2025) already does "ARAP + monocular video + articulated 4DGS" on general objects. Our delta is **diffusion supervision + SAM-2 part labels + single-image input + scene-level**. If a reviewer says "RigGS already did this," the answer is: yes, on real captured video with skeleton extraction; we operate on a hallucinated VGM video with SAM 2 part labels, and the parent settings don't transfer cleanly (their skeleton extraction needs consistent 2D keypoints across frames, which generative video doesn't provide). Be specific about the delta.

### 6.2 The VWM framing is load-bearing
The paper opens with the **perception/dynamics/policy decomposition** and our contribution to the perception layer. The single-image-input requirement is what makes this VWM-relevant — a robot in the wild doesn't have captured video. Without the VWM framing, the paper reads as "yet another monocular-4D paper." `docs/design/vwm_framing.md` is the source of truth.

### 6.3 The simulator-import demo is required, not optional
For W4: load a trained 4DGS into Genesis (or PyBullet) via the URDF we emit; run articulated IK; show that ours works and SC-GS-default fails. This is the evidence that converts "we improved a metric" into "we made the output downstream-usable." Without it, reviewers say "show me articulation matters."

### 6.4 The 4D-Fly stated failure mode is our W1 gate
4D-Fly (CVPR 2025) explicitly names "highly complex articulated motions" as a failure. If our `articulation_only` ablation on jumpingjacks does *not* visibly improve over `scgs_default` on inter-part angular consistency by end of W1, **stop and pivot to Option B/C from the survey before W2**.

## 7. Where everything lives

```
EV_Final_Project/
├── README.md                          # short intro, pipeline diagram
├── HANDOFF.md                         # THIS FILE
├── docs/
│   ├── design/                        # living references (motion, hook, framing, API, experiments)
│   │   ├── motion_design.md
│   │   ├── scgs_hook_design.md        # 3 patch sites + contract
│   │   ├── vwm_framing.md             # paper intro source-of-truth
│   │   ├── sv4d2_api.md
│   │   ├── experiments.md             # ablation matrix, timeline, baselines
│   │   └── pipeline.png
│   ├── runbooks/                      # how-to-run / how-to-reproduce
│   │   ├── demo_runbook.md
│   │   └── sv4d_runbook.md
│   ├── reports/                       # dated progress snapshots (YYYY-MM-DD_*)
│   │   ├── 2026-05-12_progress.md
│   │   ├── 2026-05-13_pipeline_state.md
│   │   ├── 2026-05-29_checkpoint.md
│   │   └── 2026-05-29_final_report.md
│   ├── planning/                      # ownership + slide plans
│   │   ├── ownership.md               # team split
│   │   └── 2026-05-13_slide_plan.md
│   └── superpowers/plans/
│       └── 2026-05-11-codebase-bootstrap.md
├── motionprior/
│   ├── losses/
│   ├── curriculum/
│   ├── geometry/
│   ├── segmentation/
│   ├── metrics/
│   ├── integration/                   # the SC-GS-facing module
│   │   ├── scgs_hook.py               # main hook object
│   │   ├── scgs_arap_adapter.py       # ControlNodeWarp.arap_loss monkey-patch
│   │   ├── vgm.py                     # SV4D 2.0 / Wan-2.2 wrappers
│   │   └── sim_bridge.py              # URDF + Genesis YAML emitter
│   └── configs/
│       ├── default.yaml
│       └── scenes/
│           ├── dnerf_jumpingjacks.yaml
│           ├── dnerf_hellwarrior.yaml
│           ├── dnerf_bouncingballs.yaml
│           └── dnerf_standup.yaml
├── tests/                             # 95+ CPU tests; runs in <1s
├── scripts/
│   ├── train.py                       # main training entry
│   ├── eval.py                        # PSNR/SSIM/LPIPS + articulation + URDF
│   ├── bootstrap_third_party.sh       # clone pinned SC-GS + AnySplat
│   ├── setup_runpod.sh                # 3 conda envs + system deps
│   ├── download_dnerf.sh
│   └── patches/
│       └── scgs_hook.patch            # 3-site patch to SC-GS train_gui.py
├── third_party/                       # gitignored; populated by bootstrap
│   ├── SC-GS/                         # pinned to 3a9d2ad4
│   ├── AnySplat/                      # pinned to 5f5e208a
│   └── README.md                      # role of each upstream
├── pyproject.toml
├── requirements-dev.txt               # CPU; what we used locally
└── requirements-gpu.txt               # CUDA torch + diffusers + transformers
```

## 8. Companion notes (in the Obsidian vault at `~/MP_Obsidian_Notes/wiki/research/`)

- `EV_Project.md` — umbrella project page (VWM perception layer; sister projects)
- `MotionPrior4DGS.md` — full paper-equivalent draft with the Option C positioning
- `Image_to_4D_Survey_May2026.md` — 25+-paper landscape with articulation addendum
- `RigGS.md`, `VideoArtGS.md`, `Part2GS.md`, `GaussianArt.md` — per-paper threat analysis

These are notes; the code repo is the source of truth for what's implemented.

## 9. If something fundamental breaks

1. **SC-GS rasterizer build fails on A4500**: A4500 is Ampere (compute capability 8.6), supported by SC-GS's CUDA extensions. If the build fails, check CUDA toolkit version (need 11.8 to match SC-GS's pin); fall back to a Docker image if the conda CUDA install conflicts.
2. **SV4D 2.0 weights blocked**: switch to Wan-2.2 with 8-bit quantization across 2 GPUs. Update `vgm.py:Wan22I2VAdapter` with the bitsandbytes loading code (~20 LOC).
3. **AnySplat fails on generated video**: known limitation — it's static-trained. Workaround in `docs/design/scgs_hook_design.md` § 4.7: swap AnySplat for DepthAnything V3 per-window if AnySplat assumption breaks.
4. **Articulation result is null on jumpingjacks**: this is the W1 stop-gate. Pivot to Option B (real monocular video, no VGM) from the survey before W2. The codebase still runs end-to-end.

## 10. The single thing to do first

```bash
cd ~/EV_Final_Project
bash scripts/bootstrap_third_party.sh && bash scripts/setup_runpod.sh
conda activate motionprior && pytest -q
```

If pytest reports 95+ passing tests, the codebase made it across the wire intact. From there, follow §4.
