# SV4D 2.0 → SC-GS pipeline runbook

> Companion to `docs/design/sv4d2_api.md` (the API reference). This document is the
> step-by-step **operational** runbook for actually running the pipeline.

## What this pipeline does

Single command turns one D-NeRF scene into an SV4D-2.0-supervised SC-GS training
scene, end-to-end. Four stages, each idempotent and individually skippable:

```
PREPARE → SV4D inference → CONVERT → SC-GS train
   |          (GPU)            |            |
   |                           |            |
   v                           v            v
RGB frames (576²)     5 mp4 videos     105 train PNGs +
                                       transforms_train.json
                                       + original D-NeRF test/ split
                                         (eval oracle preserved)
```

Eval is against the **original D-NeRF test split** so SV4D-supervised numbers are
directly comparable to the multi-view-supervised W1 baselines.

## 0. CPU smoke test (no GPU required, ~10 s)

Run this first on any machine to verify the pipeline plumbing. Uses fabricated
fake SV4D frames; ends with SC-GS's own dataset reader parsing the output.

```bash
cd /home/cthsu/EV_Final_Project
/home/cthsu/miniconda3/envs/scgs/bin/python -m pytest tests/test_sv4d_pipeline.py -x
# 11 passed in <2s.

/home/cthsu/miniconda3/envs/scgs/bin/python scripts/run_sv4d_supervised_pipeline.py \
    --scene jumpingjacks \
    --scratch_root /tmp/sv4d_smoke \
    --out_data_root /tmp/dnerf_sv4d_smoke \
    --skip_sv4d --fake_sv4d --skip_train --overwrite
# emits SC-GS command, no training launched.
```

## 1. Lab A4500 (20 GB) attempt — low-VRAM SV4D

The README claims `encoding_t=1, decoding_t=1, img_size=512` can run on
"low-VRAM" GPUs. Worth trying on the lab box before giving up to RunPod.

```bash
# One-time: SV4D deps in the motionprior env (or a separate sv4d env)
conda activate motionprior
pip install fire einops omegaconf opencv-python rembg \
    transformers diffusers safetensors imageio[ffmpeg]
pip install -e third_party/generative-models  # editable install for sgm

# Run the pipeline. Lab A4500.
/home/cthsu/miniconda3/envs/scgs/bin/python scripts/run_sv4d_supervised_pipeline.py \
    --scene jumpingjacks \
    --sv4d_python /home/cthsu/miniconda3/envs/motionprior/bin/python \
    --encoding_t 1 --decoding_t 1 --img_size 512 \
    --overwrite
```

**Expected failure mode if VRAM is insufficient:**
`CUDA out of memory. Tried to allocate ... GB`. Look in
`outputs/sv4d_supervised/jumpingjacks/sv4d_output/sv4d_run.log`. If OOM, move to
the H100 path below.

## 2. RunPod H100 path (recommended)

SV4D 2.0 at default settings needs ~40 GB VRAM. H100 80GB is the safe pick.

```bash
# On the pod, after rclone-ing the repo + checkpoints + data/dnerf:
cd /workspace/EV_Final_Project
conda env create -f environment.yml   # or pip path
pip install -e third_party/generative-models

# Run SV4D for all 4 D-NeRF articulated scenes
for s in jumpingjacks hellwarrior bouncingballs standup; do
    python scripts/run_sv4d_supervised_pipeline.py \
        --scene $s --encoding_t 4 --decoding_t 2 --img_size 576 \
        --overwrite
done
# Wall time per scene: ~10-20 min on H100 (autoregressive sampling).
# Total: ~1 hour.

# Then sync the converted scenes back to the lab box for SC-GS training:
rclone copy outputs/sv4d_supervised/ lab:EV_Final_Project/outputs/sv4d_supervised/
rclone copy data/dnerf_sv4d/ lab:EV_Final_Project/data/dnerf_sv4d/
```

## 3. SC-GS training on the SV4D-supervised scenes (lab A4500)

```bash
cd /home/cthsu/EV_Final_Project
for s in jumpingjacks hellwarrior bouncingballs standup; do
    gpu=$((counter % 3)); counter=$((counter + 1))
    CUDA_VISIBLE_DEVICES=$gpu python third_party/SC-GS/train_gui.py \
        --source_path data/dnerf_sv4d/$s \
        --model_path outputs/${s}_scgs_sv4d_default \
        --deform_type node --node_num 512 --hyper_dim 8 \
        --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
        --resolution 1 --W 576 --H 576 \
        --iterations 30000 &
done
wait
# Same SC-GS config as the multi-view baselines except:
#   --resolution 1 (no downsample; the SV4D output is already 576)
#   --W/H 576 (was 800)
# Wall time per scene: ~15 min on A4500 (similar to baseline; 105 train cams vs 100).
```

## 4. Evaluation against D-NeRF GT

The converter copies the original D-NeRF test split into the synthesized scene
dir, so SC-GS's standard evaluation hooks already point at the right GT:

```bash
# Run the existing inspection scripts on the new outputs:
/home/cthsu/miniconda3/envs/scgs/bin/python scripts/inspect_scgs_failure.py \
    --run_dir outputs/jumpingjacks_scgs_sv4d_default \
    --source_path data/dnerf_sv4d/jumpingjacks \
    --skip_render   # if you only need the metrics curves
/home/cthsu/miniconda3/envs/scgs/bin/python scripts/qualitative_inspect_scgs.py \
    --run_dir outputs/jumpingjacks_scgs_sv4d_default
/home/cthsu/miniconda3/envs/scgs/bin/python scripts/spatial_error_analysis.py \
    --run_dir outputs/jumpingjacks_scgs_sv4d_default
```

The radial-profile and periphery/core ratio numbers are then directly
comparable to the W1 multi-view baseline.

## 5. Headline experiment (the comparison that lets us write the paper)

Two SC-GS runs per scene, both at 30K iters, identical config except for the
training data source:

| Run name | Source path | Supervision regime |
|---|---|---|
| `<scene>_scgs_default_node` | `data/dnerf/<scene>` | Full multi-view D-NeRF (100 train cameras) — **W1 baseline** |
| `<scene>_scgs_sv4d_default` | `data/dnerf_sv4d/<scene>` | SV4D-generated (5 views × 21 frames = 105 cameras) — **W3 deployment regime** |

The discriminating number for the paper is `Δ periphery/core ratio` between the
two regimes. Predicted:
- Multi-view supervision: ratios ~2.3 (jumpingjacks), ~1.3 (hellwarrior), ~1.1 (standup), ~1.0 (bouncingballs) — already measured.
- SV4D-supervised: ratios should **increase** (under-determined supervision → more articulation drift).
- The W2 hook (articulation-aware ARAP) should then **close the gap** more on SV4D-supervised than on multi-view-supervised.

## 6. Failure-mode escape hatches

| Symptom | Where to look | Likely fix |
|---|---|---|
| `CUDA OOM` in stage 2 | `<scratch>/<scene>/sv4d_output/sv4d_run.log` | `--encoding_t 1 --decoding_t 1` first; then `--img_size 512`; then RunPod |
| `_v001.mp4 not found` | same log | SV4D crashed mid-run; rerun without `--skip_sv4d` |
| SC-GS reader: `invalid literal for int()` | this means the file naming convention drifted — should not happen with the current converter | Re-verify `tests/test_sv4d_pipeline.py::test_convert_writes_flat_indexed_filenames` |
| SC-GS train fails: `points3d.ply` missing | the converter copies it from the original scene; check `data/dnerf/<scene>/points3d.ply` exists | Re-run convert stage |
| SV4D output has wrong number of views | `<scratch>/<scene>/sv4d_output/sv4d2/*.mp4` count | Check `--variant`; 4-view emits 5 mp4s (1 input + 4 novel) |

## 7. License posture (settled)

SV4D 2.0 Community License is free for research and for commercial use by
orgs/individuals under $1M annual revenue. We are well within scope; no
enterprise license needed. See `checkpoints/sv4d2.0/LICENSE.md`.
