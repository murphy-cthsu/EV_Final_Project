#!/usr/bin/env bash
# Run the full ablation matrix from docs/experiments.md.
# Idempotent -- skips runs that already have eval.json under outputs/.
#
# Designed to run on either the lab A4500 box or a RunPod H100.
# On the lab box, supports 3-way parallelism via CUDA_VISIBLE_DEVICES rotation.
#
# Usage:
#   bash scripts/run_ablations.sh dnerf            # D-NeRF scenes only (W1/W2)
#   bash scripts/run_ablations.sh dnerf 3          # 3-way parallel on lab box
#   bash scripts/run_ablations.sh all 1            # everything, serial (RunPod)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GROUP="${1:-dnerf}"
PARALLEL="${2:-1}"

DNERF_SCENES=(jumpingjacks hellwarrior bouncingballs standup hook lego mutant trex)
DYCHECK_SCENES=(apple block paper-windmill space-out spin teddy wheel)
HYPERNERF_SCENES=(vrig-3dprinter vrig-broom vrig-chicken vrig-peel-banana)

case "${GROUP}" in
    dnerf)
        SCENES=("${DNERF_SCENES[@]}")
        CONFIG_PREFIX="scenes/dnerf"
        DATASET_DIR="data/dnerf"
        ;;
    dnerf_articulated)
        SCENES=(jumpingjacks hellwarrior bouncingballs standup)
        CONFIG_PREFIX="scenes/dnerf"
        DATASET_DIR="data/dnerf"
        ;;
    dycheck)
        SCENES=("${DYCHECK_SCENES[@]}")
        CONFIG_PREFIX="scenes/dycheck"
        DATASET_DIR="data/dycheck"
        ;;
    hypernerf)
        SCENES=("${HYPERNERF_SCENES[@]}")
        CONFIG_PREFIX="scenes/hypernerf"
        DATASET_DIR="data/hypernerf"
        ;;
    *)
        echo "Unknown group: ${GROUP}" >&2
        echo "Use: dnerf | dnerf_articulated | dycheck | hypernerf" >&2
        exit 1
        ;;
esac

# Ablation rows (matches docs/experiments.md §3)
ABLATIONS=(
    scgs_default
    articulation_only
    gating_only
    curriculum_only
    gating_curriculum
    full
    ours_minus_articulation
)

OUTPUT_ROOT="outputs"
mkdir -p "${OUTPUT_ROOT}"

# Build the (scene, ablation) job list, skipping already-done ones
JOBS=()
for scene in "${SCENES[@]}"; do
    for ablation in "${ABLATIONS[@]}"; do
        OUT_DIR="${OUTPUT_ROOT}/${scene}_${ablation}"
        EVAL_JSON="${OUT_DIR}/eval.json"
        if [ -f "${EVAL_JSON}" ]; then
            echo "[skip] ${scene}/${ablation} -- eval.json already present"
            continue
        fi
        JOBS+=("${scene}:${ablation}")
    done
done

echo ""
echo "Queued ${#JOBS[@]} runs (${PARALLEL}-way parallel) for group: ${GROUP}"
echo ""

run_one() {
    local scene="$1"
    local ablation="$2"
    local gpu_id="$3"
    local out_dir="${OUTPUT_ROOT}/${scene}_${ablation}"
    local config="${CONFIG_PREFIX}_${scene}"
    local log="${out_dir}.log"

    mkdir -p "${out_dir}"
    echo "[run] gpu=${gpu_id} scene=${scene} ablation=${ablation} -> ${out_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/train.py \
        --config "${config}" \
        --scene_root "${DATASET_DIR}/${scene}" \
        --output_dir "${out_dir}" \
        --ablation "${ablation}" \
        > "${log}" 2>&1 || {
        echo "[FAIL] train.py for ${scene}/${ablation}; see ${log}"
        return 1
    }
    echo "[eval] scene=${scene} ablation=${ablation}"
    python scripts/eval.py \
        --run_dir "${out_dir}" \
        --scene_root "${DATASET_DIR}/${scene}" \
        --gaussian_trajectory_path "${out_dir}/positions.pt" \
        --part_labels_path "${out_dir}/part_labels.pt" \
        --emit_urdf \
        >> "${log}" 2>&1 || echo "[WARN] eval.py for ${scene}/${ablation} failed; check log"
}

i=0
PIDS=()
for job in "${JOBS[@]}"; do
    scene="${job%%:*}"
    ablation="${job##*:}"
    gpu_id=$((i % PARALLEL))
    run_one "${scene}" "${ablation}" "${gpu_id}" &
    PIDS+=($!)
    i=$((i + 1))
    if [ "${i}" -ge "${PARALLEL}" ] && [ "$((i % PARALLEL))" -eq 0 ]; then
        wait "${PIDS[@]}"
        PIDS=()
    fi
done
if [ "${#PIDS[@]}" -gt 0 ]; then
    wait "${PIDS[@]}"
fi

echo ""
echo "All ${#JOBS[@]} runs complete. Aggregate with:"
echo "  python scripts/aggregate_results.py --runs_root ${OUTPUT_ROOT} --out_md results.md"
