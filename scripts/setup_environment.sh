#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-looserope}"
PYTHON_VERSION="3.10"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
PYPI_INDEX="https://pypi.org/simple"
DETECTRON2_COMMIT="a1ce2f956a1d2212ad672e3c47d53405c2fe4312"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not available. Install Miniconda or Anaconda first." >&2
    exit 1
fi

if conda env list | awk 'NF > 1 && $1 !~ /^#/ {print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Error: conda environment '${ENV_NAME}' already exists." >&2
    echo "Choose another name: bash scripts/setup_environment.sh <name>" >&2
    exit 1
fi

echo "Creating conda environment '${ENV_NAME}'..."
conda create \
    --name "${ENV_NAME}" \
    "python=${PYTHON_VERSION}" \
    pip \
    --channel conda-forge \
    --override-channels \
    --yes

echo "Installing PyTorch 2.4.0 with CUDA 12.4..."
conda run --no-capture-output --name "${ENV_NAME}" \
    python -m pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url "${TORCH_INDEX}" \
    --extra-index-url "${PYPI_INDEX}"

echo "Installing LooseRoPE dependencies..."
conda run --no-capture-output --name "${ENV_NAME}" \
    python -m pip install \
    --index-url "${PYPI_INDEX}" \
    --requirement "${REPO_ROOT}/requirements.txt"

echo "Building Detectron2 against the installed PyTorch..."
conda run --no-capture-output --name "${ENV_NAME}" \
    python -m pip install \
    --index-url "${PYPI_INDEX}" \
    --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_COMMIT}"

echo "Checking the installation..."
conda run --no-capture-output --name "${ENV_NAME}" python - <<'PY'
import torch
import torchvision
import diffusers
import transformers
import detectron2
from qwen_vl_utils import process_vision_info  # noqa: F401

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch installed, but CUDA is not available.")

print(f"torch={torch.__version__}, torchvision={torchvision.__version__}")
print(f"diffusers={diffusers.__version__}, transformers={transformers.__version__}")
print(f"detectron2={detectron2.__version__}, CUDA={torch.version.cuda}")
print("LooseRoPE environment is ready.")
PY

echo "Activate it with: conda activate ${ENV_NAME}"
