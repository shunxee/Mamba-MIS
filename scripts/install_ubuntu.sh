#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==25.2' 'setuptools==78.1.0' 'wheel==0.45.1' 'packaging==25.0' 'ninja==1.11.1.4'
python -m pip install 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'
# Source build uses the installed CUDA PyTorch. nvcc from CUDA Toolkit 12.8 must be on PATH.
MAMBA_FORCE_BUILD=TRUE MAMBA_KEEP_CUDA_BUILD=TRUE python -m pip install \
  --no-build-isolation 'causal-conv1d==1.5.2' 'mamba-ssm==2.2.5'
python -m pip freeze > installed-ubuntu.lock.txt
printf '%s\n' 'Installation complete. Run: python -m pytest -m "not slow"'
