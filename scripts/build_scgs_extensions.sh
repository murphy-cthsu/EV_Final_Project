#!/usr/bin/env bash
# Build SC-GS CUDA extensions (diff-gaussian-rasterization, simple-knn) in sc-gs env.
# PyTorch 2.x+cu130 expects CUDA 13.x nvcc; system /usr/local/cuda is often 12.x.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCGS="${REPO_ROOT}/third_party/SC-GS"
PY="${PY:-python}"

if ! command -v "${PY}" >/dev/null 2>&1; then
  echo "Set PY to your sc-gs python, e.g. PY=/path/to/miniconda3/envs/sc-gs/bin/python"
  exit 1
fi

# Wrapper nvcc reports 13.0 for PyTorch's version check; delegates to real nvcc.
CUDA13="${REPO_ROOT}/.cuda13_toolkit"
mkdir -p "${CUDA13}/bin"
cat > "${CUDA13}/bin/nvcc" << 'EOF'
#!/bin/bash
if [[ "$1" == "--version" ]]; then
  echo "Cuda compilation tools, release 13.0, V13.0.0"
  exit 0
fi
exec /usr/local/cuda/bin/nvcc "$@"
EOF
chmod +x "${CUDA13}/bin/nvcc"

export CUDA_HOME="${CUDA13}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

cd "${SCGS}"
"${PY}" -m pip install -r requirements.txt
"${PY}" -m pip install --no-build-isolation --force-reinstall \
  ./submodules/diff-gaussian-rasterization \
  ./submodules/simple-knn

# requirements.txt pins old opencv; restore a working build for SC-GS imports
"${PY}" -m pip install -q "opencv-python>=4.8" "Pillow>=9.4"

TORCH_LIB="$("${PY}" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
"${PY}" -c "
from simple_knn._C import distCUDA2
import diff_gaussian_rasterization
print('SC-GS CUDA extensions OK')
"

# train_partrigid_hier only needs gaussian_model + render (not full Scene / pytorch3d)
"${PY}" -c "
import importlib.util, sys, types
from pathlib import Path
root = Path('${SCGS}').resolve()
sys.path.insert(0, str(root))

def load_sub(pkg, name, rel):
    spec = importlib.util.spec_from_file_location(f'{pkg}.{name}', root / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'{pkg}.{name}'] = mod
    spec.loader.exec_module(mod)

scene = types.ModuleType('scene')
scene.__path__ = [str(root / 'scene')]
sys.modules['scene'] = scene
load_sub('scene', 'gaussian_model', 'scene/gaussian_model.py')
import gaussian_renderer
print('train_partrigid import chain OK')
" || echo "WARN: import-chain smoke failed (pytorch3d not required for partrigid train)"

echo "Done. Before training:"
echo "  export LD_LIBRARY_PATH=${TORCH_LIB}:\$LD_LIBRARY_PATH"
echo "  (full SC-GS train_gui.py also needs a pytorch3d build matching your torch)"
