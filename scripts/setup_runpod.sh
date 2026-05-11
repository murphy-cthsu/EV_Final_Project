#!/usr/bin/env bash
# One-shot RunPod (H100) environment setup.
# Run from the repo root on a fresh pod. Idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---- System packages ----
apt-get update -qq
apt-get install -y -qq \
    git wget curl unzip build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    libegl1 libglu1-mesa \
    > /dev/null

# ---- Miniconda (if not present) ----
if [ ! -d /opt/miniconda3 ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda3
    rm /tmp/miniconda.sh
fi
export PATH="/opt/miniconda3/bin:${PATH}"
conda init bash > /dev/null
. /opt/miniconda3/etc/profile.d/conda.sh

# ---- Three envs: motionprior dev, sc-gs, anysplat ----

# (1) motionprior-dev: lightweight, for running our pytest + losses on the pod
if ! conda env list | grep -q '^motionprior '; then
    conda create -y -n motionprior python=3.10
fi
conda activate motionprior
pip install -q -r requirements-gpu.txt
pip install -q -e .
pytest -q
conda deactivate

# (2) scgs: SC-GS training env. Defer to SC-GS's own environment.yml.
# We install motionprior into this env too so the training loop can import it.
if [ -f third_party/SC-GS/environment.yml ]; then
    if ! conda env list | grep -q '^scgs '; then
        conda env create -n scgs -f third_party/SC-GS/environment.yml
    fi
    conda activate scgs
    pip install -e .
    conda deactivate
else
    echo "WARN: third_party/SC-GS not bootstrapped yet. Run bootstrap_third_party.sh first."
fi

# (3) anysplat: AnySplat inference env. Used to produce the static 3DGS + pose.
if [ -d third_party/AnySplat ]; then
    if ! conda env list | grep -q '^anysplat '; then
        conda create -y -n anysplat python=3.10
    fi
    conda activate anysplat
    pip install -q torch==2.1.2 torchvision==0.16.2 \
        --extra-index-url https://download.pytorch.org/whl/cu121
    if [ -f third_party/AnySplat/requirements.txt ]; then
        pip install -q -r third_party/AnySplat/requirements.txt
    fi
    conda deactivate
else
    echo "WARN: third_party/AnySplat not bootstrapped yet. Run bootstrap_third_party.sh first."
fi

echo ""
echo "RunPod setup complete. Available envs:"
echo "  conda activate motionprior   # local tests + our package only"
echo "  conda activate scgs          # SC-GS training (with motionprior installed)"
echo "  conda activate anysplat      # AnySplat inference"
echo ""
echo "Wan-2.2 weights:"
echo "  huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir checkpoints/wan22"
