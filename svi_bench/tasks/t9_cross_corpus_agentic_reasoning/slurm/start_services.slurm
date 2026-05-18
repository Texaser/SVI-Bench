#!/bin/bash
#SBATCH --job-name=t9_services
#SBATCH --partition=a6000
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/arc/sha17/video_deep_research/data/public_release/SVI-Bench/svi_bench/tasks/t9_cross_corpus_agentic_reasoning/logs/services_%j.out
#SBATCH --error=/mnt/arc/sha17/video_deep_research/data/public_release/SVI-Bench/svi_bench/tasks/t9_cross_corpus_agentic_reasoning/logs/services_%j.err

set -e

# ============================================
# Configuration
# ============================================
PIPELINE_DIR=/mnt/arc/sha17/video_deep_research/data/public_release/SVI-Bench/svi_bench/tasks/t9_cross_corpus_agentic_reasoning
CONDA_ENV="${CONDA_ENV:-svi-bench}"

# Architecture to use (set via environment or default)
ARCH="${ARCH:-gpt5}"

# ============================================
# Setup
# ============================================
cd "$PIPELINE_DIR"
mkdir -p "$PIPELINE_DIR/logs"

echo "============================================"
echo "T9 GPU Services"
echo "============================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Architecture: $ARCH"
echo "GPUs:"
nvidia-smi -L
echo "============================================"

# Activate conda environment
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

# ============================================
# Start services using Python helper
# Reads config from archs/$ARCH.yaml + configs/models.yaml
# ============================================
EXTRA_ARGS=""
[[ -n "$NO_FLASH_ATTN" ]] && EXTRA_ARGS="$EXTRA_ARGS --no-flash-attn"

python scripts/start_services.py --arch "$ARCH" $EXTRA_ARGS
