# Live-demo runbook — group discussion 2026-05-13

> Companion to `docs/PROGRESS_2026-05-12.md`. Every command below reproduces a published artifact in <2 min. No GPU needed.

## 0. One-time setup (you've already done this — sanity check)

```bash
which conda                                          # → ~/miniconda3/bin/conda
conda env list | grep scgs                           # → scgs env present
/home/cthsu/miniconda3/envs/scgs/bin/python -c \
    "import numpy, scipy, matplotlib, PIL; print('ok')"
```

If anything errors, the rest will not work.

## 1. "Where does SC-GS fail?" — the headline figure

```bash
cd /home/cthsu/EV_Final_Project
xdg-open outputs/_cross_scene_failure/HEADLINE.png   # or: feh / eog / etc.
```

Single-image summary: 4 scenes × [GT | pred | error | temporal-std] + the radial profile chart + a takeaway box.

If anyone asks "how was this assembled":
```bash
/home/cthsu/miniconda3/envs/scgs/bin/python scripts/build_headline_figure.py
# rebuilds from cached artifacts in ~25 s.
```

## 2. "What's the number?" — the periphery-vs-core analysis

```bash
xdg-open outputs/_cross_scene_failure/core_vs_periphery_cross_scene.png
xdg-open outputs/_cross_scene_failure/radial_profile_cross_scene.png
cat outputs/_cross_scene_failure/spatial_error_summary.json | jq '.summary'
```

Expected stdout:
```json
[
  {"scene": "jumpingjacks",  "core_err": 0.0129, "periphery_err": 0.0294, "periphery_over_core": 2.28},
  {"scene": "bouncingballs", "core_err": 0.0133, "periphery_err": 0.0136, "periphery_over_core": 1.03},
  {"scene": "hellwarrior",   "core_err": 0.0156, "periphery_err": 0.0210, "periphery_over_core": 1.35},
  {"scene": "standup",       "core_err": 0.0136, "periphery_err": 0.0155, "periphery_over_core": 1.14}
]
```

If asked "rerun it fresh":
```bash
/home/cthsu/miniconda3/envs/scgs/bin/python scripts/spatial_error_analysis.py --all_scenes
# ~10 s.
```

## 3. "Show me one scene in detail"

```bash
ls outputs/jumpingjacks_scgs_default_node/inspection/qualitative/
# → worst_frames_contact_sheet.png   ← GT/pred/error for top-4 worst frames
# → joint_zoom_00019.png             ← top-3 high-error patches zoomed
# → temporal_strobe.png              ← time-interp limb-end ghosting
# → core_vs_periphery.png            ← per-scene bar
# → radial_profile.png               ← per-scene radial curve
# → spatial_error.json               ← raw numbers
```

Open `worst_frames_contact_sheet.png` for the qualitative argument, then `temporal_strobe.png` for the "this is pure deformation-field failure" argument (middle/right panel is the money plot).

## 4. "How was the W1 baseline trained?"

```bash
cat HANDOFF.md | sed -n '109,160p'
# OR more concisely:
head -25 outputs/jumpingjacks_scgs_default_node/inspection/REPORT.md
```

The actual command (one of four, parallelized across 3 lab cards):

```bash
CUDA_VISIBLE_DEVICES=0 python third_party/SC-GS/train_gui.py \
    --source_path data/dnerf/jumpingjacks \
    --model_path  outputs/jumpingjacks_scgs_default \
    --deform_type node --node_num 512 --hyper_dim 8 \
    --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
    --resolution 2 --W 800 --H 800 \
    --iterations 30000
```

13m43s on a single A4500.

## 5. "What's the SAM 2 segmentation we'll use in W2?"

```bash
xdg-open runs_aux/jumpingjacks_label_overlay.png
xdg-open runs_aux/hellwarrior_label_overlay.png
# (etc. for bouncingballs, standup)
ls -la runs_aux/                # confirm all 4 *.pt + *.png present
```

Already on disk. 2D label maps tied to `train/r_000.png` per scene, 11–18 parts per scene (hierarchical SAM 2). Per-Gaussian (3D) lifting is W2 day 1 if we go that route, or W2 day 5 otherwise — open for discussion.

## 6. "What does the actual code path look like for W2?"

```bash
cat scripts/patches/scgs_hook.patch | head -50
ls motionprior/integration/
# → scgs_hook.py, scgs_arap_adapter.py
cat docs/scgs_hook_design.md | sed -n '1,60p'
```

The hook degrades gracefully — `--ablation scgs_default` is the no-op path and was verified to match the baseline.

## 7. "If someone challenges the failure claim or the contribution"

§1 and §4 of `PROGRESS_2026-05-12.md` are the prepared responses. The key reframings:

**On the contribution itself:**
- The claim is a **3-component joint system**, not "better ARAP."
- **Component 1** (SAM-2 piecewise-rigid ARAP) *directly* improves articulation. **Components 2 and 3** (ARAP-energy gating, frequency curriculum) are *protective* against generative-supervision drift — they keep Component 1's signal from being corrupted by hallucinated frames.
- The novelty is the intersection. RigGS does Component 1 alone but only because their supervision is clean captured video. The diffusion-supervised cluster has no drift-rejection mechanism. We deliver Component 1 in a regime where it would otherwise collapse.

**On the W1 evidence:**
- D-NeRF is *not* the ill-posedness benchmark — it's the **Tier 1 controlled-comparison benchmark**. SC-GS-default's 40.85 PSNR there is *not* a failure number; the failure shows up in the **structured spatial residual** (×2.28 periphery ratio, 5–6× radial ramp) that scales with articulation complexity.
- Multi-view D-NeRF is the *easiest possible supervision regime* — a lower bound on the failure under SV4D-supervised inputs.
- **Tier 1 measures Component 1 only.** Components 2 and 3 are dormant under clean supervision (no drift to gate against). The bigger headline numbers come from Tier 2 (SV4D-supervised D-NeRF) where the joint system activates.
- Discriminating evidence for "Component 1 is *the* fix" lives in the Tier 1 ablation matrix (shuffled-labels row + higher-capacity baseline). W1 numbers alone are suggestive, not causal.

**On dataset choice (D-NeRF "too simple"):**
- D-NeRF's uniqueness is the **clean evaluation oracle**, not easy supervision. Under SV4D supervision (Tier 2), D-NeRF becomes genuinely ill-posed (5 azimuths vs 100 cameras, drifty content) while keeping the held-out test cams as a fair oracle. Best of both worlds; DyCheck cannot match that.
- DyCheck (Tier 3) is the real-world existence proof, not the controlled comparison.

## 8. "Reset and re-run everything from disk if something breaks"

```bash
cd /home/cthsu/EV_Final_Project

# 1. Per-scene log parse + curves (no SC-GS render needed)
for s in jumpingjacks bouncingballs hellwarrior standup; do
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/inspect_scgs_failure.py \
        --run_dir outputs/${s}_scgs_default_node \
        --source_path data/dnerf/${s} \
        --skip_render
done

# 2. Qualitative (contact sheets / zooms / strobes)
/home/cthsu/miniconda3/envs/scgs/bin/python scripts/qualitative_inspect_scgs.py --all_scenes

# 3. Spatial error analysis (the headline number)
/home/cthsu/miniconda3/envs/scgs/bin/python scripts/spatial_error_analysis.py --all_scenes

# 4. Headline composite
/home/cthsu/miniconda3/envs/scgs/bin/python scripts/build_headline_figure.py

# Total wall time: ~3 min for steps 2-4 (step 1 is instant if logs exist).
```
