#!/bin/bash
#SBATCH --job-name=t9_benchmark
#SBATCH --partition=cpu           # CPU partition - no GPUs needed (uses services)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-9               # 10 parallel workers (adjust as needed)
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err

set -e

# ============================================
# Configuration (set via environment or defaults)
# ============================================
PIPELINE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-svi-bench}"

# These should be set when submitting the job:
#   QUESTIONS_FILE=$T9_ROOT/questions/hockey.json
#   EXPERIMENT_NAME=my_experiment  (optional, auto-generated if not set)
#   TOTAL_QUESTIONS=1000
#   ARCH=gpt5

QUESTIONS_FILE="${QUESTIONS_FILE:-${T9_ROOT:-$PIPELINE_DIR/../../data/t9}/questions/hockey.json}"
TOTAL_QUESTIONS="${TOTAL_QUESTIONS:-1000}"
ARCH="${ARCH:-gpt5}"
SPORT="${SPORT:?ERROR: SPORT env var required (basketball/hockey/soccer)}"

# EXPERIMENT_NAME must be set before job submission to ensure all workers use the same directory.
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERROR: EXPERIMENT_NAME not set. All workers would create separate folders."
    echo "Usage: EXPERIMENT_NAME=batch_gpt5_$(date +%Y%m%d_%H%M%S) SPLIT=test sbatch slurm/run_benchmark.slurm"
    exit 1
fi

# Experiment directory
EXPERIMENT_DIR="$PIPELINE_DIR/experiments/$EXPERIMENT_NAME"

# ============================================
# Calculate question range for this worker
# ============================================
NUM_WORKERS=${SLURM_ARRAY_TASK_COUNT:-10}
WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}

# Ceiling division for even distribution
QUESTIONS_PER_WORKER=$(( (TOTAL_QUESTIONS + NUM_WORKERS - 1) / NUM_WORKERS ))
START_IDX=$((WORKER_ID * QUESTIONS_PER_WORKER))
END_IDX=$((START_IDX + QUESTIONS_PER_WORKER))

# Cap at total questions
if [ $END_IDX -gt $TOTAL_QUESTIONS ]; then
    END_IDX=$TOTAL_QUESTIONS
fi

# Skip if nothing to process
if [ $START_IDX -ge $TOTAL_QUESTIONS ]; then
    echo "Worker $WORKER_ID: No questions to process (start=$START_IDX >= total=$TOTAL_QUESTIONS)"
    exit 0
fi

# ============================================
# Setup
# ============================================
cd "$PIPELINE_DIR"
mkdir -p "$PIPELINE_DIR/logs"

echo "============================================"
echo "T9 Benchmark Worker $WORKER_ID"
echo "============================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Experiment Dir: $EXPERIMENT_DIR"
echo "Questions file: $QUESTIONS_FILE"
echo "Range: $START_IDX to $END_IDX"
echo "Architecture: $ARCH"
echo "Sport: $SPORT"
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
# Run batch processor
# ============================================
python run_batch.py \
    --questions "$QUESTIONS_FILE" \
    --experiment-name "$EXPERIMENT_NAME" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --arch "$ARCH" \
    --sport "$SPORT" \
    --worker-id $WORKER_ID \
    --resume

echo ""
echo "============================================"
echo "Worker $WORKER_ID Complete"
echo "Results: $EXPERIMENT_DIR/results/"
echo "$(date)"
echo "============================================"
