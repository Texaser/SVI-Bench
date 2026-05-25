#!/usr/bin/env bash
# T8 — basketball (task2) post-generation mIoU evaluation.
#
# Pipeline:
#   0. (Optional) If $VALIDATION_DIR is passed instead of $VIDEO_DIR, first
#      flatten the per-clip subdirs (`<clip>/generated.mp4`) into a flat
#      symlink directory matching `<clip>.mp4`.
#   1. Shard the GT bbox listing across NUM_GPUS GPUs.
#   2. Each GPU runs eval_generated_videos.py — YOLOX sports-mot detector
#      + MixSort tracker, on every clip in $VIDEO_DIR.
#   3. After all GPUs finish, run video_miou_task2.py over the combined
#      tracker output — computes last-frame mIoU against the end_bbox in
#      captions.json.
#
# Usage:
#   bash eval/run_basketball.sh [VIDEO_DIR] [GT_LIST] [CAPTIONS] [CKPT]
#
# Set VALIDATION_DIR env var to auto-flatten subdir structure (typical for
# DiffSynth output: validation/step-N/<clip>/generated.mp4).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
DATA_ROOT="${SVI_BENCH_DATA:-$REPO_ROOT/data}"
SPORT_DIR="$DATA_ROOT/T8/basketball"
export PYTHONPATH="$HERE:$HERE/MixViT:${PYTHONPATH:-}"

# CD into the eval/ dir so `track.yaml`'s relative
# `pretrained/MixFormer_sports_train.pth.tar` (loaded by mixformer_deit at
# tracker init) resolves to $HERE/pretrained/.
cd "$HERE"

VIDEO_DIR="${1:-${VIDEO_DIR:-}}"
GT_LIST="${2:-$SPORT_DIR/splits/test_1000.bbox_paths.txt}"
CAPTIONS="${3:-$SPORT_DIR/captions.json}"
CKPT="${4:-$HERE/pretrained/yolox_x_sports_train.pth.tar}"

# Optional preprocess: flatten <validation>/<clip>/generated.mp4 -> <video_dir>/<clip>.mp4
if [ -n "${VALIDATION_DIR:-}" ]; then
    VIDEO_DIR="${VIDEO_DIR:-${VALIDATION_DIR}/generated_flat}"
    mkdir -p "$VIDEO_DIR"
    echo ">>> Flattening $VALIDATION_DIR -> $VIDEO_DIR ..."
    for clip_dir in "$VALIDATION_DIR"/*/; do
        clip_name=$(basename "$clip_dir")
        gen="$clip_dir/generated.mp4"
        [ -f "$gen" ] && ln -sf "$gen" "$VIDEO_DIR/${clip_name}.mp4"
    done
    echo ">>> Linked $(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l) videos"
fi

if [ -z "$VIDEO_DIR" ]; then
    echo "Error: provide VIDEO_DIR as 1st arg, or set VALIDATION_DIR env var." >&2
    exit 1
fi

EXP_FILE="$HERE/exps/example/mot/yolox_x_sportsmot.py"
OUTPUT_DIR="${VIDEO_DIR}/eval_results"
NUM_GPUS="${NUM_GPUS:-8}"

echo "============================================================"
echo "T8 Basketball (task2) Tracker + mIoU"
echo "============================================================"
echo "VIDEO_DIR:          $VIDEO_DIR"
echo "GT_LIST:            $GT_LIST"
echo "CAPTIONS:           $CAPTIONS"
echo "CKPT:               $CKPT"
echo "OUTPUT_DIR:         $OUTPUT_DIR"
echo "NUM_GPUS:           $NUM_GPUS"
echo ""

SPLIT_DIR="$(mktemp -d /tmp/svi_t8_eval_split.XXXXXX)"
split -n l/${NUM_GPUS} -d -a 1 "$GT_LIST" "$SPLIT_DIR/part_"

echo ">>> Launching $NUM_GPUS GPU workers..."
PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SPLIT_FILE="$SPLIT_DIR/part_${GPU_ID}"
    [ -s "$SPLIT_FILE" ] || continue
    echo "  GPU $GPU_ID: $(wc -l < "$SPLIT_FILE") entries"
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$HERE/eval_generated_videos.py" \
        --video_dir "$VIDEO_DIR" \
        --gt_list "$SPLIT_FILE" \
        --exp_file "$EXP_FILE" \
        --ckpt "$CKPT" \
        --output_dir "${OUTPUT_DIR}/gpu${GPU_ID}" \
        --skip_existing &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid" || echo "Warning: $pid exited non-zero"; done
rm -rf "$SPLIT_DIR"

echo ""
echo ">>> Tracker done. Running video_miou_task2.py for last-frame mIoU..."
python "$HERE/video_miou_task2.py" \
    --video_dir "$VIDEO_DIR" \
    --gt_list "$GT_LIST" \
    --captions_json "$CAPTIONS" \
    --eval_results_dir "$OUTPUT_DIR"

echo ""
echo ">>> Last-frame mIoU summary: $VIDEO_DIR/video_miou_task2_results/summary.json"
