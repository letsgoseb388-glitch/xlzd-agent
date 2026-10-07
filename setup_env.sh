#!/usr/bin/env bash
#
# One-shot environment setup for the xlzd-agent pipeline.
#
# Creates a Python 3.12 virtual environment in .venv and installs the pinned
# dependencies. Detects whether an NVIDIA GPU is present and installs the
# matching torch build:
#   - GPU present: a CUDA build new enough for recent cards (including
#     Blackwell / RTX 50-series, which need torch >= 2.7 on the cu128 index).
#   - No GPU: the CPU build, which runs the whole pipeline (slower but complete).
#
# Requires `uv` (https://github.com/astral-sh/uv). Install it with:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:
#   bash setup_env.sh
#   source .venv/bin/activate

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is not installed."
    echo "Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "==> Creating Python 3.12 venv in .venv"
uv venv .venv --python 3.12

# shellcheck disable=SC1091
source .venv/bin/activate

# Detect a usable NVIDIA GPU.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "==> NVIDIA GPU detected — installing CUDA torch (cu128, works on Blackwell/RTX 50-series)"
    uv pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
else
    echo "==> No GPU detected — installing CPU torch (pipeline will run on CPU)"
    uv pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Installing the rest of the dependencies"
uv pip install -r requirements.txt

echo
echo "==> Verifying the install"
python3 -c "
import torch, numpy, pandas, h5py, matplotlib
print('deps ok')
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
"

echo
echo "Done. Activate the environment with:  source .venv/bin/activate"
