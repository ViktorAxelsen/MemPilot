#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="mempilot"
RECREATE_ENV=false

usage() {
    cat <<'EOF'
Usage: setup_environment.sh [--env-name NAME] [--recreate]

Create the pinned Linux Conda environment used by MemPilot.

Options:
  --env-name NAME  Conda environment name (default: mempilot)
  --recreate       Remove an existing environment with the same name first
  -h, --help       Show this help message
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --env-name)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "ERROR: --env-name requires a non-empty value." >&2
                exit 2
            fi
            ENV_NAME="$2"
            shift 2
            ;;
        --recreate)
            RECREATE_ENV=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: the pinned FlashAttention wheel requires Linux x86_64." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available. Install Miniconda or Anaconda first." >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
FLASH_ATTN_REQUIREMENTS_FILE="$PROJECT_ROOT/requirements-flash-attn.txt"

for required_file in "$REQUIREMENTS_FILE" "$FLASH_ATTN_REQUIREMENTS_FILE"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: required file not found: $required_file" >&2
        exit 1
    fi
done

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda_environment_exists() {
    conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"
}

if [ "$RECREATE_ENV" = true ] && conda_environment_exists; then
    if [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ]; then
        conda deactivate
    fi
    echo "Removing existing Conda environment: $ENV_NAME"
    conda env remove --name "$ENV_NAME" --yes
fi

if conda_environment_exists; then
    echo "Reusing existing Conda environment: $ENV_NAME"
else
    echo "Creating Conda environment: $ENV_NAME (Python 3.12)"
    conda create --name "$ENV_NAME" python=3.12 --yes
fi

conda activate "$ENV_NAME"

PYTHON_MINOR_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_MINOR_VERSION" != "3.12" ]; then
    echo "ERROR: environment $ENV_NAME uses Python $PYTHON_MINOR_VERSION, expected 3.12." >&2
    echo "Rerun with --recreate to rebuild it." >&2
    exit 1
fi

echo "Installing packaging tools"
python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch 2.8.0 with the CUDA 12.8 runtime"
python -m pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "Installing MemPilot dependencies"
python -m pip install --requirement "$REQUIREMENTS_FILE"

echo "Installing the prebuilt FlashAttention wheel"
python -m pip install --requirement "$FLASH_ATTN_REQUIREMENTS_FILE"

echo "Checking installed dependency constraints"
python -m pip check

echo "Verifying the training stack"
python - <<'PY'
from importlib.metadata import version
import platform

import flash_attn
import torch
import vllm


expected_prefixes = {
    "torch": "2.8.0",
    "verl": "0.8.0",
    "vllm": "0.11.0",
    "TransferQueue": "0.1.7",
    "flash-attn": "2.8.3",
    "llmlingua": "0.2.2",
    "nltk": "3.9.2",
    "transformers": "4.57.6",
}

print(f"Python: {platform.python_version()}")
for distribution, expected_prefix in expected_prefixes.items():
    installed = version(distribution)
    print(f"{distribution}: {installed}")
    if not installed.startswith(expected_prefix):
        raise RuntimeError(
            f"Unexpected {distribution} version: {installed}; expected {expected_prefix}"
        )

print(f"PyTorch CUDA runtime: {torch.version.cuda}")
if torch.version.cuda != "12.8":
    raise RuntimeError(f"Expected PyTorch CUDA runtime 12.8, got {torch.version.cuda}")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"Basic CUDA test passed: {tuple(y.shape)} on {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no CUDA device is visible in this shell; GPU execution was not tested.")

print(f"vLLM import passed: {vllm.__version__}")
print(f"FlashAttention import passed: {flash_attn.__version__}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver and visible GPUs:"
    if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; then
        echo "WARNING: nvidia-smi could not query a visible GPU in this shell." >&2
    fi
else
    echo "WARNING: nvidia-smi is unavailable; verify the NVIDIA driver before training." >&2
fi

echo "Environment setup completed."
echo "Activate it later with: conda activate $ENV_NAME"
