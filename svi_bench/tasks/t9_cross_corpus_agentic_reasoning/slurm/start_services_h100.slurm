#!/bin/bash
#SBATCH --job-name=t9_services_h100
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/services_h100_%j.out
#SBATCH --error=logs/services_h100_%j.err

set -e

PIPELINE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-svi-bench-t9}"
ARCH="${ARCH:-gpt5}"

cd "$PIPELINE_DIR"
mkdir -p "$PIPELINE_DIR/logs"

echo "============================================"
echo "T9 GPU Services (H100)"
echo "============================================"
echo "Node: $(hostname)"
echo "Architecture: $ARCH"
nvidia-smi -L
echo "============================================"

if [ -z "${CONDA_PROFILE:-}" ]; then
    echo "ERROR: CONDA_PROFILE env var is not set. SLURM compute nodes do not have a usable \$HOME." >&2
    echo "       export it BEFORE sbatch, e.g.:" >&2
    echo "         export CONDA_PROFILE=/path/to/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi
if [ ! -f "$CONDA_PROFILE" ]; then
    echo "ERROR: CONDA_PROFILE points at non-existent file: $CONDA_PROFILE" >&2
    exit 1
fi
source "$CONDA_PROFILE"
conda activate "$CONDA_ENV"

EXTRA_ARGS=""
[[ -n "$NO_FLASH_ATTN" ]] && EXTRA_ARGS="$EXTRA_ARGS --no-flash-attn"

python scripts/start_services.py --arch "$ARCH" $EXTRA_ARGS
