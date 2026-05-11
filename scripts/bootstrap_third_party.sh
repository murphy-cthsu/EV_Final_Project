#!/usr/bin/env bash
# Clone upstream repositories at pinned SHAs into third_party/.
# third_party/ is gitignored -- this script is the only authoritative source
# for which upstream commits we depend on. Update SHAs here when bumping.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${REPO_ROOT}/third_party"
mkdir -p "${TP}"

# ---- SC-GS (deformable backbone, ARAP already implemented) ----
SCGS_REPO="https://github.com/yihua7/SC-GS.git"
SCGS_SHA="3a9d2ad4e4fc058b0763d446ae9e6b1be120b872"
SCGS_DIR="${TP}/SC-GS"

if [ ! -d "${SCGS_DIR}/.git" ]; then
    echo "Cloning SC-GS @ ${SCGS_SHA}..."
    git clone "${SCGS_REPO}" "${SCGS_DIR}"
    git -C "${SCGS_DIR}" checkout "${SCGS_SHA}"
    git -C "${SCGS_DIR}" submodule update --init --recursive
else
    echo "SC-GS already present; checking pin..."
    git -C "${SCGS_DIR}" fetch --quiet
    git -C "${SCGS_DIR}" checkout "${SCGS_SHA}"
    git -C "${SCGS_DIR}" submodule update --init --recursive
fi

# ---- AnySplat (feed-forward 3DGS + pose) ----
ANYSPLAT_REPO="https://github.com/OpenRobotLab/AnySplat.git"
ANYSPLAT_SHA="5f5e208a7dd57d52e43ea0d553a95eab526e8775"
ANYSPLAT_DIR="${TP}/AnySplat"

if [ ! -d "${ANYSPLAT_DIR}/.git" ]; then
    echo "Cloning AnySplat @ ${ANYSPLAT_SHA}..."
    git clone "${ANYSPLAT_REPO}" "${ANYSPLAT_DIR}"
    git -C "${ANYSPLAT_DIR}" checkout "${ANYSPLAT_SHA}"
    git -C "${ANYSPLAT_DIR}" submodule update --init --recursive
else
    echo "AnySplat already present; checking pin..."
    git -C "${ANYSPLAT_DIR}" fetch --quiet
    git -C "${ANYSPLAT_DIR}" checkout "${ANYSPLAT_SHA}"
    git -C "${ANYSPLAT_DIR}" submodule update --init --recursive
fi

echo ""
echo "third_party/ ready. Next steps:"
echo "  - SC-GS:    cd third_party/SC-GS    && follow its README to build CUDA extensions"
echo "  - AnySplat: cd third_party/AnySplat && pip install -r requirements.txt"
echo "  - Wan-2.2 weights: download from HuggingFace (Wan-AI/Wan2.2-I2V-A14B)"
