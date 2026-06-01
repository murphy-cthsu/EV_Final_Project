#!/bin/bash
# Gated pipeline: wait for a genuinely idle GPU, then train the hellwarrior
# canonical (SC-GS static on 57-view frame-0 d-3dgs) and run the static floor eval.
set -e
cd /home/cthsu/EV_Final_Project
PY=/home/cthsu/miniconda3/envs/scgs/bin/python
[ -x "$PY" ] || PY=python

echo "[gate] waiting for a free-enough GPU (mem<3000MiB and util<45%)..."
GPU=""
for i in $(seq 1 480); do   # up to ~4h (480 * 30s)
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F', ' '$2 < 3000 && $3 < 45 {print $1; exit}')
  if [ -n "$GPU" ]; then echo "[gate] GPU $GPU is free (iter $i). proceeding."; break; fi
  sleep 30
done
if [ -z "$GPU" ]; then echo "[gate] no idle GPU after timeout. aborting."; exit 2; fi

export CUDA_VISIBLE_DEVICES=$GPU
echo "[run] CUDA_VISIBLE_DEVICES=$GPU"

echo "=== STEP 1: train static canonical (SC-GS) on hellwarrior_frame0 ==="
$PY third_party/SC-GS/train_gui.py \
    --source_path data/custom/hellwarrior_frame0 \
    --model_path outputs/custom/hellwarrior_canonical \
    --deform_type node --node_num 512 --hyper_dim 8 \
    --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
    --resolution 1 --W 576 --H 576 --iterations 5000

echo "=== STEP 2: static canonical floor eval (vs d-3dgs clean GT) ==="
$PY scripts/eval_static_canonical.py --scene hellwarrior

echo "=== DONE ==="
