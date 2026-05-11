#!/usr/bin/env bash
# T7 — soccer inference.
# Same pipeline as basketball.sh but on the soccer test set with the
# soccer-trained LoRA checkpoint.
#
# Usage:
#   bash inference/soccer.sh [output_path]
#
# Edit TEST_SUBSET / VALIDATION_VIDEO_BASE / VALIDATION_BACKGROUND_VIDEO_BASE
# below to point at your local soccer data.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$HERE/.." && pwd)"

# `from diffsynth import ...` in soccer.py resolves to TASK_DIR/diffsynth.
export PYTHONPATH="$TASK_DIR:${PYTHONPATH:-}"

DEFAULT_OUTPUT_PATH="./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_soccer_full_scale"
OUTPUT_PATH="${1:-$DEFAULT_OUTPUT_PATH}"

VALIDATION_SCRIPT="$HERE/soccer.py"
TEST_SUBSET="/mnt/bum/hanyi/repo/sports_detection/segment-anything-2-real-time/soccer_set/test_5.txt"
NUM_GPUS=4
SPLIT_DIR="./validation_splits"

echo "============================================================"
echo "T7 Soccer Multi-GPU Inference"
echo "============================================================"
echo "Output path: $OUTPUT_PATH"
echo "Number of GPUs: $NUM_GPUS"
echo ""

if [ ! -d "$OUTPUT_PATH" ]; then
    echo "Error: Output directory not found: $OUTPUT_PATH"
    exit 1
fi

CHECKPOINTS=$(find "$OUTPUT_PATH" -name "step-*.safetensors" -type f 2>/dev/null)
if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No checkpoint files found in $OUTPUT_PATH"
    exit 1
fi

LATEST_CHECKPOINT=$(echo "$CHECKPOINTS" | while read -r ckpt; do
    step_num=$(basename "$ckpt" | sed -n 's/step-\([0-9]*\)\.safetensors/\1/p')
    if [ -n "$step_num" ]; then
        printf "%06d %s\n" "$step_num" "$ckpt"
    fi
done | sort -rn | head -1 | awk '{print $2}')

STEP_NUM=$(basename "$LATEST_CHECKPOINT" | sed -n 's/step-\([0-9]*\)\.safetensors/\1/p')
echo "Latest checkpoint: $LATEST_CHECKPOINT (step $STEP_NUM)"

if [ ! -f "$TEST_SUBSET" ]; then
    echo "Error: Test subset not found: $TEST_SUBSET"
    exit 1
fi

echo ""
echo "Splitting test set..."
python "$HERE/split_validation_set.py" \
    --input "$TEST_SUBSET" \
    --output-dir "$SPLIT_DIR" \
    --num-splits $NUM_GPUS

export VALIDATION_NUM_FRAMES=81
export VALIDATION_TIME_DIVISION_FACTOR=1
export VALIDATION_VIDEO_BASE="/mnt/bum/hanyi/data/soccer_video_fps_15"
export VALIDATION_BACKGROUND_VIDEO_BASE="/mnt/bum/hanyi/data/soccer_inpainting_video"

START_TIME=$(date +%s)
PIDS=()
LOG_DIR="./validation_logs"
mkdir -p "$LOG_DIR"

TEST_BASENAME=$(basename "$TEST_SUBSET" .txt)
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    SPLIT_FILE="${SPLIT_DIR}/${TEST_BASENAME}_split_${gpu_id}.txt"
    [ -f "$SPLIT_FILE" ] || { echo "Warning: missing $SPLIT_FILE"; continue; }
    NUM_SAMPLES=$(wc -l < "$SPLIT_FILE")
    LOG_FILE="${LOG_DIR}/gpu_${gpu_id}_step_${STEP_NUM}.log"
    echo "GPU $gpu_id -> $NUM_SAMPLES samples, log $LOG_FILE"
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        export NUM_VALIDATION_SAMPLES=$NUM_SAMPLES
        export VALIDATION_BBOX_FOLDER="$SPLIT_FILE"
        python "$VALIDATION_SCRIPT" "$LATEST_CHECKPOINT" > "$LOG_FILE" 2>&1
    ) &
    PIDS+=($!)
    sleep 2
done

for pid in "${PIDS[@]}"; do
    wait $pid || echo "Warning: pid $pid exited non-zero"
done

ELAPSED_TIME=$(($(date +%s) - START_TIME))
printf "Done in %02d:%02d:%02d\n" $((ELAPSED_TIME/3600)) $(((ELAPSED_TIME%3600)/60)) $((ELAPSED_TIME%60))
echo "Results: $OUTPUT_PATH/validation/step-$STEP_NUM/"
