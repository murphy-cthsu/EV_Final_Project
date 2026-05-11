#!/usr/bin/env bash
# GPU environment setup. Works on:
#   * RunPod / containerized boxes (root, no sudo needed)
#   * Lab servers (non-root user, sudo may or may not be available)
#   * Personal workstations
#
# Idempotent. Detects privilege level and adapts:
#   * Root or sudo available  -> installs system packages via apt-get
#   * Non-root, no sudo       -> skips system packages, prints what's needed
# Miniconda path:
#   * Root: /opt/miniconda3   (system-wide)
#   * User: ~/miniconda3      (per-user)
#
# Usage:
#   bash scripts/setup_runpod.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Privilege detection ----
IS_ROOT=0
HAS_SUDO=0
if [ "$(id -u)" -eq 0 ]; then
    IS_ROOT=1
elif command -v sudo > /dev/null && sudo -n true 2>/dev/null; then
    # sudo without password prompt
    HAS_SUDO=1
fi

run_priv() {
    if [ "${IS_ROOT}" -eq 1 ]; then
        "$@"
    elif [ "${HAS_SUDO}" -eq 1 ]; then
        sudo "$@"
    else
        return 1
    fi
}

# ---- System packages ----
APT_PACKAGES=(
    git wget curl unzip build-essential cmake
    libgl1-mesa-glx libglib2.0-0 ffmpeg
    libegl1 libglu1-mesa
)

if [ "${IS_ROOT}" -eq 1 ] || [ "${HAS_SUDO}" -eq 1 ]; then
    echo "[setup] installing system packages via apt-get..."
    run_priv apt-get update -qq
    run_priv apt-get install -y -qq "${APT_PACKAGES[@]}" > /dev/null
else
    echo "[setup] non-root, no passwordless sudo -- skipping apt-get."
    echo "[setup] If anything below fails with missing-library errors, ask your"
    echo "[setup] sysadmin to install these packages system-wide:"
    for pkg in "${APT_PACKAGES[@]}"; do
        echo "          - ${pkg}"
    done
    # Quick sanity check on the most likely-to-be-missing libs
    if ! command -v cmake > /dev/null; then
        echo "[setup] WARN: cmake not on PATH; SC-GS rasterizer build will fail."
    fi
    if ! ldconfig -p 2>/dev/null | grep -q libGL.so; then
        echo "[setup] WARN: libGL not found in linker cache; AnySplat / SAM 2 may fail."
    fi
fi

# ---- Miniconda ----
# Pick install path based on privilege; reuse existing conda if already present
if command -v conda > /dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null || echo "")"
    if [ -n "${CONDA_BASE}" ]; then
        echo "[setup] found existing conda at ${CONDA_BASE}; using it."
    fi
else
    if [ "${IS_ROOT}" -eq 1 ]; then
        CONDA_BASE="/opt/miniconda3"
    else
        CONDA_BASE="${HOME}/miniconda3"
    fi
    if [ ! -d "${CONDA_BASE}" ]; then
        echo "[setup] installing Miniconda to ${CONDA_BASE}..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "${CONDA_BASE}"
        rm /tmp/miniconda.sh
    fi
    export PATH="${CONDA_BASE}/bin:${PATH}"
    # conda init writes to ~/.bashrc; user-level, safe without sudo
    conda init bash > /dev/null
fi

# Source conda for the rest of this script
# shellcheck disable=SC1091
. "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- Conda envs: motionprior, scgs, anysplat ----

# (1) motionprior: lightweight, for running our pytest + losses
if ! conda env list | grep -q '^motionprior '; then
    echo "[setup] creating env: motionprior (python 3.10)"
    conda create -y -n motionprior python=3.10 > /dev/null
fi
conda activate motionprior
echo "[setup] installing motionprior GPU requirements..."
pip install -q -r requirements-gpu.txt
pip install -q -e .
echo "[setup] running pytest in motionprior env..."
pytest -q
conda deactivate

# (2) scgs: SC-GS training env. SC-GS ships requirements.txt + two CUDA
# submodules (diff-gaussian-rasterization, simple-knn). Install in that order.
if [ -d third_party/SC-GS ] && [ -f third_party/SC-GS/requirements.txt ]; then
    if ! conda env list | grep -q '^scgs '; then
        echo "[setup] creating env: scgs (python 3.10)"
        conda create -y -n scgs python=3.10 > /dev/null
    fi
    conda activate scgs
    # Build toolchain inside the env so we don't need sudo on the lab box.
    # cmake + ninja for the CUDA extension builds; conda-forge gives us
    # a recent cmake without touching the system.
    echo "[setup] installing build toolchain (cmake, ninja) into scgs env..."
    conda install -y -n scgs -c conda-forge cmake ninja > /dev/null
    # Sanity-check that gcc/g++ are available (system-level; cannot install
    # without sudo). If missing, the CUDA extension build will fail.
    if ! command -v gcc > /dev/null; then
        echo "[setup] WARN: gcc not found on PATH. The scgs CUDA extensions"
        echo "        (diff-gaussian-rasterization, simple-knn) need a system C/C++"
        echo "        compiler. Ask sysadmin to install build-essential, OR try:"
        echo "        conda install -n scgs -c conda-forge gxx_linux-64=11"
    fi
    echo "[setup] installing PyTorch 2.1.2+cu121 in scgs env..."
    pip install -q torch==2.1.2 torchvision==0.16.2 \
        --extra-index-url https://download.pytorch.org/whl/cu121
    echo "[setup] installing SC-GS requirements..."
    pip install -q -r third_party/SC-GS/requirements.txt
    # Two CUDA extensions: diff-gaussian-rasterization + simple-knn.
    # Need nvcc + gcc. If the build fails (no CUDA toolkit, or older driver),
    # warn but continue so the user can still pytest motionprior.
    for ext in diff-gaussian-rasterization simple-knn; do
        ext_dir="third_party/SC-GS/submodules/${ext}"
        if [ -d "${ext_dir}" ]; then
            echo "[setup] building ${ext} (CUDA extension)..."
            if ! pip install -q "${ext_dir}"; then
                echo "[setup] WARN: ${ext} build failed. Check CUDA toolkit / nvcc."
                echo "        Continue setup; you'll need to fix this before training."
            fi
        else
            echo "[setup] WARN: ${ext_dir} not present. Did bootstrap_third_party.sh"
            echo "        recurse into SC-GS submodules? Try:"
            echo "        git -C third_party/SC-GS submodule update --init --recursive"
        fi
    done
    echo "[setup] installing motionprior into scgs env..."
    pip install -q -e .
    conda deactivate
else
    echo "[setup] WARN: third_party/SC-GS not bootstrapped."
    echo "         Run: bash scripts/bootstrap_third_party.sh"
fi

# (3) anysplat: AnySplat inference env
if [ -d third_party/AnySplat ]; then
    if ! conda env list | grep -q '^anysplat '; then
        echo "[setup] creating env: anysplat (python 3.10)"
        conda create -y -n anysplat python=3.10 > /dev/null
    fi
    conda activate anysplat
    echo "[setup] installing AnySplat deps..."
    pip install -q torch==2.1.2 torchvision==0.16.2 \
        --extra-index-url https://download.pytorch.org/whl/cu121
    if [ -f third_party/AnySplat/requirements.txt ]; then
        pip install -q -r third_party/AnySplat/requirements.txt
    fi
    conda deactivate
else
    echo "[setup] WARN: third_party/AnySplat missing."
    echo "         Run: bash scripts/bootstrap_third_party.sh"
fi

echo ""
echo "[setup] done. Conda base: ${CONDA_BASE}"
echo ""
echo "Available envs:"
echo "  conda activate motionprior   # local tests + our package only"
echo "  conda activate scgs          # SC-GS training (with motionprior installed)"
echo "  conda activate anysplat      # AnySplat inference"
echo ""
echo "Next steps:"
echo "  - SAM 2 weights:   huggingface-cli download facebook/sam2-hiera-large --local-dir checkpoints/sam2"
echo "  - SV4D 2.0 weights (primary VGM): huggingface-cli download stabilityai/sv4d2.0 --local-dir checkpoints/sv4d2"
echo "  - Wan-2.2 weights (fallback VGM): huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir checkpoints/wan22"
echo "  - D-NeRF dataset:  bash scripts/download_dnerf.sh"
echo ""
echo "If you're on the lab A4500x3 box, this is your dev environment."
echo "If you're on a rented RunPod H100, this is for W3-W4 scaling."
