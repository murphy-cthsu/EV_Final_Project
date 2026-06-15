#!/usr/bin/env bash
set -euo pipefail

# Run SV4D2 on lego_r7_train.gif with 4 elevations x 2 azimuth sets = 8 inferences.
# Usage (from repo root):
#   bash scripts/sampling/run_sv4d2_lego_views.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

INPUT_PATH="/root/generative-models/assets/wen_gen_videos/jumpingjacks_spliting_21frames.gif"
MODEL_PATH="/root/checkpoints/sv4d2.safetensors"
OUTPUT_ROOT="outputs/jumpingjacks_splitting_r10_train"

ELEVATIONS=(0)  # input view stays at 0°; novel views use +elev
# Input view (0°) + 4 novel views per run
AZIMUTH_SETS=(
  # "0,60,120,180,240"
  # "0,150,210,270,330"
  "0,20,40,60,80"
  "0,120,140,160,180"
  "0,200,220,240,260"
  "0,280,300,320,340"
)
AZIMUTH_LABELS=(
  # "60-240"
  # "150-330"
  "20-80"
  "120-180"
  "200-260"
  "280-340"
)

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "Error: input video not found: ${INPUT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Error: model checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi

run_idx=0
total_runs=$((${#ELEVATIONS[@]} * ${#AZIMUTH_SETS[@]}))

for elev in "${ELEVATIONS[@]}"; do
  for i in "${!AZIMUTH_SETS[@]}"; do
    run_idx=$((run_idx + 1))
    azimuths="${AZIMUTH_SETS[$i]}"
    az_label="${AZIMUTH_LABELS[$i]}"
    output_folder="${OUTPUT_ROOT}/elev${elev}_az${az_label}"

    echo "============================================================"
    echo "Run ${run_idx}/${total_runs}: elevation=${elev}°, azimuths=[${azimuths}]"
    echo "Output: ${output_folder}"
    echo "============================================================"

    python scripts/sampling/simple_video_sample_4d2.py \
      --input_path="${INPUT_PATH}" \
      --model_path="${MODEL_PATH}" \
      --output_folder="${output_folder}" \
      --elevations_deg="[0,${elev},${elev},${elev},${elev}]" \
      --azimuths_deg="[${azimuths}]" \
      --object_crop=False
  done
done

echo "Done. ${total_runs} runs saved under ${OUTPUT_ROOT}/"
