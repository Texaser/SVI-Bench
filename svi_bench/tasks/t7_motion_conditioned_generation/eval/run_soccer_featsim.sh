#!/usr/bin/env bash
# T7 — soccer SigLIP2 IoU-gated feature similarity.
# Same as run_basketball_featsim.sh but --sport soccer + soccer test set.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
DATA_ROOT="${SVI_BENCH_DATA:-$REPO_ROOT/data}"
SPORT_DIR="$DATA_ROOT/T7/soccer"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
cd "$HERE"

STEP_DIR="${1:-}"
GT_LIST="${2:-$SPORT_DIR/splits/test_subset_100.bbox_paths.txt}"
RESULTS_DIR="${RESULTS_DIR:-${STEP_DIR}/miou_results_all}"
OUTPUT_DIR="${OUTPUT_DIR:-${STEP_DIR}/feature_sim}"
MODE="${MODE:-iou_gated}"
NUM_GPUS="${NUM_GPUS:-8}"

if [ -z "$STEP_DIR" ]; then
    echo "Error: STEP_DIR (1st arg) required." >&2
    exit 1
fi

echo "============================================================"
echo "T7 Soccer Feature Similarity (SigLIP2, $MODE)"
echo "============================================================"
echo "STEP_DIR:    $STEP_DIR"
echo "GT_LIST:     $GT_LIST"
echo "RESULTS_DIR: $RESULTS_DIR"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "MODE:        $MODE"
echo "NUM_GPUS:    $NUM_GPUS"
echo ""

mkdir -p "$OUTPUT_DIR"
SPLIT_DIR="$(mktemp -d /tmp/svi_t7_soccer_featsim_split.XXXXXX)"
split -n l/${NUM_GPUS} -d -a 1 "$GT_LIST" "$SPLIT_DIR/part_"

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SPLIT_FILE="$SPLIT_DIR/part_${GPU_ID}"
    [ -s "$SPLIT_FILE" ] || continue
    echo "  GPU $GPU_ID: $(wc -l < "$SPLIT_FILE") entries"
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$HERE/feature_sim.py" \
        --results_dir "$RESULTS_DIR" \
        --video_dir "$STEP_DIR" \
        --gt_list "$SPLIT_FILE" \
        --sport soccer \
        --mode "$MODE" \
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
    --mode "$MODE" \
    --aggregate_only

echo ""
echo ">>> Summary: $OUTPUT_DIR/summary.json"
