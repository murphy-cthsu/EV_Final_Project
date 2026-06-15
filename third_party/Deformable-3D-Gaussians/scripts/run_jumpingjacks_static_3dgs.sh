#!/usr/bin/env bash
set -euo pipefail

# Generate static multi-view dataset from D-3DGS (t=0) and train standard 3DGS.
ROOT="/root/Deformable-3D-Gaussians"
DATA="/root/data_fixed/jumpingjacks_static_t0"
OUT="/root/Deformable-3D-Gaussians/output/jumpingjacks_static_3dgs"

echo "=== Step 1: Render static multi-view dataset (t=0, 200 cameras) ==="
cd "$ROOT"
conda run -n d-3dgs python utils/render_static_multiview_dataset.py \
  --model_path output/jumpingjacks \
  --iteration 40000 \
  --poses /root/data_fixed/jumpingjacks/transforms_train.json \
  --output_dir "$DATA" \
  --render_size 800 \
  --test_every 8 \
  --white_background

echo "=== Step 2: Train static 3DGS on generated dataset ==="
cd /root/SegAnyGAussians
conda run -n saga-3dgs python train_scene.py \
  -s "$DATA" \
  -m "$OUT" \
  --eval \
  --white_background \
  --iterations 30000 \
  --test_iterations 7000 15000 30000 \
  --save_iterations 7000 15000 30000

echo "Done. Dataset: $DATA"
echo "3DGS model: $OUT"
