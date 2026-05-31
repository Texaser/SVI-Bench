#!/bin/bash
#SBATCH --job-name=t9_bench_8gpu
#SBATCH --partition=a6000         # A6000 partition
#SBATCH --nodes=1
#SBATCH --gpus=0                  # No GPUs - uses services
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/bench_8gpu_%A_%a.out
#SBATCH --error=logs/bench_8gpu_%A_%a.err

set -e

PIPELINE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-svi-bench-t9}"

QUESTIONS_FILE="${QUESTIONS_FILE:-${T9_ROOT:-$PIPELINE_DIR/../../data/T9}/questions/hockey.json}"
TOTAL_QUESTIONS="${TOTAL_QUESTIONS:-1000}"
ARCH="${ARCH:-gpt5}"
SPORT="${SPORT:?ERROR: SPORT env var required (basketball/hockey/soccer)}"

# EXPERIMENT_NAME must be set before job submission to ensure all workers use the same directory.
# Generate with: export EXPERIMENT_NAME="batch_${ARCH}_8gpu_$(date +%Y%m%d_%H%M%S)"
# Then submit with: sbatch --export=ALL slurm/run_benchmark_8gpu.slurm
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERROR: EXPERIMENT_NAME not set. All workers would create separate folders."
    echo "Usage: EXPERIMENT_NAME=batch_gpt5_8gpu_$(date +%Y%m%d_%H%M%S) SPLIT=test sbatch slurm/run_benchmark_8gpu.slurm"
    exit 1
fi

EXPERIMENT_DIR="$PIPELINE_DIR/experiments/$EXPERIMENT_NAME"

NUM_WORKERS=${SLURM_ARRAY_TASK_COUNT:-10}
WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}

QUESTIONS_PER_WORKER=$(( (TOTAL_QUESTIONS + NUM_WORKERS - 1) / NUM_WORKERS ))
START_IDX=$((WORKER_ID * QUESTIONS_PER_WORKER))
END_IDX=$((START_IDX + QUESTIONS_PER_WORKER))

if [ $END_IDX -gt $TOTAL_QUESTIONS ]; then
    END_IDX=$TOTAL_QUESTIONS
fi

if [ $START_IDX -ge $TOTAL_QUESTIONS ]; then
    echo "Worker $WORKER_ID: No questions to process"
    exit 0
fi

cd "$PIPELINE_DIR"
mkdir -p "$PIPELINE_DIR/logs"

echo "============================================"
echo "T9 Benchmark Worker $WORKER_ID (8x A6000)"
echo "============================================"
echo "Node: $(hostname)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Range: $START_IDX to $END_IDX"
echo "Sport: $SPORT"
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

python run_batch.py \
    --questions "$QUESTIONS_FILE" \
    --experiment-name "$EXPERIMENT_NAME" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --arch "$ARCH" \
    --sport "$SPORT" \
    --worker-id $WORKER_ID \
    --resume

echo "Worker $WORKER_ID Complete - Results: $EXPERIMENT_DIR/results/"
