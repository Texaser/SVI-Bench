#!/usr/bin/env bash
# T8 — basketball (task2) SigLIP2 last-frame feature similarity.
#
# Inputs needed (provided by the mIoU pipeline + polished captions):
#   $VIDEO_DIR             flat dir of generated <clip>.mp4 (or <clip>/generated.mp4)
#   $GT_LIST               test_task2_final_*.txt with mixsort bbox paths
#   $POLISHED_CAPTIONS     polished_captions_final.json
#   $EVAL_RESULTS_DIR      MixSort tracking output (gpu*/{clip}/...)
#
# Usage:
#   bash eval/run_basketball_featsim.sh [VIDEO_DIR] [GT_LIST] [POLISHED_CAPTIONS] [EVAL_RESULTS_DIR]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
cd "$HERE"

VIDEO_DIR="${1:-${VIDEO_DIR:-}}"
GT_LIST="${2:-/mnt/bum/hanyi/repo/sports_detection/segment-anything-2-real-time/basketball_set/test_task2_final_1000.txt}"
POLISHED_CAPTIONS="${3:-/mnt/bum/hanyi/repo/sports_detection/segment-anything-2-real-time/polished_captions_final.json}"
EVAL_RESULTS_DIR="${4:-${VIDEO_DIR}/eval_results}"
OUTPUT_DIR="${OUTPUT_DIR:-${VIDEO_DIR}/feature_sim_task2}"
NUM_GPUS="${NUM_GPUS:-8}"

if [ -z "$VIDEO_DIR" ]; then
    echo "Error: VIDEO_DIR (1st arg) required." >&2
    exit 1
fi

echo "============================================================"
echo "T8 Basketball Last-Frame Feature Similarity (SigLIP2)"
echo "============================================================"
echo "VIDEO_DIR:         $VIDEO_DIR"
echo "GT_LIST:           $GT_LIST"
echo "POLISHED_CAPTIONS: $POLISHED_CAPTIONS"
echo "EVAL_RESULTS_DIR:  $EVAL_RESULTS_DIR"
echo "OUTPUT_DIR:        $OUTPUT_DIR"
echo "NUM_GPUS:          $NUM_GPUS"
echo ""

mkdir -p "$OUTPUT_DIR"
SPLIT_DIR="$(mktemp -d /tmp/svi_t8_featsim_split.XXXXXX)"
split -n l/${NUM_GPUS} -d -a 1 "$GT_LIST" "$SPLIT_DIR/part_"

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SPLIT_FILE="$SPLIT_DIR/part_${GPU_ID}"
    [ -s "$SPLIT_FILE" ] || continue
    echo "  GPU $GPU_ID: $(wc -l < "$SPLIT_FILE") entries"
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$HERE/feature_sim.py" \
        --video_dir "$VIDEO_DIR" \
        --gt_list "$SPLIT_FILE" \
        --captions_json "$POLISHED_CAPTIONS" \
        --eval_results_dir "$EVAL_RESULTS_DIR" \
        --sport basketball \
        --output_dir "$OUTPUT_DIR" \
        --skip_existing &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid" || echo "Warning: $pid exited non-zero"; done
rm -rf "$SPLIT_DIR"

echo ""
echo ">>> Aggregating per-clip JSONs..."
python "$HERE/feature_sim.py" \
    --output_dir "$OUTPUT_DIR" \
    --aggregate_only

echo ""
echo ">>> Summary: $OUTPUT_DIR/summary.json"
