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

# Note: we use `set -eo pipefail` but NOT `-u`. conda-forge's compiler
# activation hooks (binutils_linux-64, gcc_linux-64, gxx_linux-64) reference
# variables like ADDR2LINE / CONDA_BACKUP_CXX without defaulting them, so any
# `conda activate <env>` in an env with those packages aborts under nounset.
set -eo pipefail

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
    # IMPORTANT: install all conda packages BEFORE `conda activate`. The
    # gxx_linux-64 package registers a deactivate hook that references
    # CONDA_BACKUP_CXX, which is only set by its activate hook. If we activate
    # first and then install gxx, the activate hook never fires for the new
    # package, and any subsequent conda action that re-runs deactivate (e.g.
    # `conda install` on an active env) blows up under `set -u` with
    # "CONDA_BACKUP_CXX: unbound variable". Installing while inactive lets
    # the activate hooks fire cleanly on the next `conda activate`.
    echo "[setup] installing build toolchain (cmake, ninja) into scgs env..."
    conda install -y -n scgs -c conda-forge cmake ninja > /dev/null
    # The lab box has system CUDA 11.5, but torch 2.1.2+cu121 requires CUDA
    # 12.1 headers / nvcc. Install a 12.1 toolkit into the env from nvidia's
    # conda channel (cuda-nvcc + cuda-cudart-dev + cuda-libraries-dev is the
    # minimal set that satisfies torch's _check_cuda_version + headers).
    echo "[setup] installing CUDA 12.1 toolkit into scgs env (this may take a few minutes)..."
    conda install -y -n scgs -c "nvidia/label/cuda-12.1.1" \
        cuda-nvcc cuda-cudart-dev cuda-libraries-dev > /dev/null
    # The lab box's system has 'gcc' but no 'g++' / cc1plus, so nvcc's host
    # compile step fails. Install a self-contained gcc 11 toolchain (gcc 11
    # is within CUDA 12.1's supported host-compiler range).
    echo "[setup] installing gcc/g++ 11 toolchain into scgs env..."
    conda install -y -n scgs -c conda-forge \
        gcc_linux-64=11 gxx_linux-64=11 > /dev/null
    conda activate scgs
    echo "[setup] installing PyTorch 2.1.2+cu121 in scgs env..."
    pip install -q torch==2.1.2 torchvision==0.16.2 \
        --extra-index-url https://download.pytorch.org/whl/cu121
    # Pre-install Pillow + opencv via wheels: SC-GS pins Pillow==7.0.0 and
    # opencv-python==4.5.5.62 (yanked), both of which lack py3.10 manylinux
    # wheels and require system libjpeg-dev / etc. to build from source.
    # Newer versions ship binary deps inside the wheel -> no sudo needed.
    echo "[setup] pre-installing modern Pillow + opencv (avoids Pillow==7.0.0 source build)..."
    # opencv-python-headless 4.11+ requires numpy>=2 on py3.9+, but torch 2.1.2
    # is built against numpy 1.x ABI -- pin to 4.10.x so we stay on numpy 1.26.
    pip install -q "Pillow>=10,<12" "opencv-python-headless>=4.9,<4.11"
    # Build deps for --no-build-isolation paths below: SC-GS submodules and
    # pytorch3d import torch in their setup.py, so they need to see torch from
    # the env (not a fresh PEP 517 sandbox). That means we must also have
    # setuptools / wheel in the env.
    # Pin setuptools<70: torch 2.1.2's cpp_extension does `from pkg_resources
    # import packaging`, which setuptools 81+ removed. 69.x keeps pkg_resources.
    pip install -q "setuptools<70" wheel
    echo "[setup] installing SC-GS requirements (filtered: dropping pinned Pillow / opencv-python)..."
    scgs_req_filtered="$(mktemp --suffix=.txt)"
    grep -vE '^[[:space:]]*(Pillow|opencv-python)([[:space:]]*[=<>!~].*)?[[:space:]]*$' \
        third_party/SC-GS/requirements.txt > "${scgs_req_filtered}"
    pip install -q -r "${scgs_req_filtered}"
    rm -f "${scgs_req_filtered}"
    # Two CUDA extensions: diff-gaussian-rasterization + simple-knn.
    # Point CUDA_HOME at the env's CUDA 12.1 install so torch picks our nvcc
    # instead of the system's 11.5 toolkit. Also prepend its bin dir to PATH
    # so nvcc is callable. Export CC/CXX/CUDAHOSTCXX so distutils+nvcc both
    # use the conda gcc 11 wrappers instead of the system's gcc (which has
    # no cc1plus on this box).
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="${CXX}"
    for ext in diff-gaussian-rasterization simple-knn; do
        ext_dir="third_party/SC-GS/submodules/${ext}"
        if [ -d "${ext_dir}" ]; then
            echo "[setup] building ${ext} (CUDA extension)..."
            # --no-build-isolation: their setup.py imports torch at module
            # scope. PEP 517 builds otherwise spawn a clean sandbox without
            # torch, causing ModuleNotFoundError before nvcc even runs.
            if ! pip install -q --no-build-isolation "${ext_dir}"; then
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
    # Same conda-installs-before-activate rule as scgs (see comment above):
    # gxx_linux-64's deactivate hook breaks under `set -u` if its activate
    # hook never fired.
    # Same CUDA 12.1 toolkit + gcc 11 toolchain as scgs (pytorch3d +
    # torch_scatter both compile CUDA extensions from source).
    echo "[setup] installing CUDA 12.1 toolkit into anysplat env..."
    conda install -y -n anysplat -c "nvidia/label/cuda-12.1.1" \
        cuda-nvcc cuda-cudart-dev cuda-libraries-dev > /dev/null
    echo "[setup] installing gcc/g++ 11 toolchain into anysplat env..."
    conda install -y -n anysplat -c conda-forge \
        gcc_linux-64=11 gxx_linux-64=11 > /dev/null
    conda activate anysplat
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="${CXX}"
    echo "[setup] installing AnySplat deps..."
    # AnySplat's req file is consistent with torch 2.2.0 + cu121:
    #   - xformers==0.0.24 is built against torch 2.2.0
    #   - unpinned e3nn resolves to >=0.6.0 which requires torch>=2.2
    #   - the gsplat wheel at the bottom of their req is the pt22cu121 build
    # The scgs env stays on 2.1.2 (SC-GS upstream); only anysplat moves to 2.2.0.
    pip install -q torch==2.2.0 torchvision==0.17.0 \
        --extra-index-url https://download.pytorch.org/whl/cu121
    # AnySplat's req file includes pytorch3d (git) and torch_scatter, both of
    # which import torch in setup.py -> need --no-build-isolation.
    # Same setuptools<70 pin as scgs (torch 2.1.2's cpp_extension uses
    # pkg_resources, which 81+ removed).
    pip install -q "setuptools<70" wheel
    # Anchor numpy at <2 BEFORE resolving the req file. Without this anchor
    # pip's resolver has been observed pulling numpy 2.2.6 from a transitive
    # (e.g. gradio / scikit-image latest), which then breaks torch's numpy
    # 1.x ABI at the pytorch3d build step. AnySplat's req pins numpy==1.25.0
    # but that pin is processed too late in the resolution order.
    pip install -q "numpy<2"
    if [ -f third_party/AnySplat/requirements.txt ]; then
        pip install -q --no-build-isolation -r third_party/AnySplat/requirements.txt
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
echo "  - HF auth (gated repos):  hf auth login   # needed for SV4D 2.0; accept license at"
echo "                             https://huggingface.co/stabilityai/sv4d2.0 first"
echo "  - SAM 2 weights (.pt format -- HF mirror is transformers-converted and incompatible"
echo "                  with sam2.build_sam.build_sam2; use Meta's direct URL):"
echo "       mkdir -p checkpoints && wget -c -O checkpoints/sam2_hiera_large.pt \\"
echo "           https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
echo "  - SV4D 2.0 weights (primary VGM; matches vgm.py:SV4D2Adapter default checkpoint path):"
echo "       hf download stabilityai/sv4d2.0 --local-dir checkpoints/sv4d2.0"
echo "  - Wan-2.2 weights (fallback VGM, only if SV4D 2.0 license blocks):"
echo "       hf download Wan-AI/Wan2.2-I2V-A14B --local-dir checkpoints/wan22"
echo "  - D-NeRF dataset:  bash scripts/download_dnerf.sh"
echo ""
echo "If you're on the lab A4500x3 box, this is your dev environment."
echo "If you're on a rented RunPod H100, this is for W3-W4 scaling."
