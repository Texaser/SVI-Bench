#!/usr/bin/env bash
# T7 — basketball post-generation mIoU evaluation.
#
# Pipeline:
#   1. Shard the GT bbox listing across NUM_GPUS GPUs.
#   2. Each GPU runs eval_generated_videos.py — which loads the YOLOX
#      sports-mot detector + MixSort tracker and tracks every clip in the
#      generated-videos directory.
#   3. After all GPUs finish, run video_miou.py over the combined tracker
#      output to produce the final holistic Video mIoU score.
#
# Usage:
#   bash eval/run_basketball.sh [VIDEO_DIR] [GT_LIST] [CKPT]
#
# All three args are optional; sensible defaults below.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# `from yolox..` / `from miou_metric..` / `from MixViT lib..` all resolve here.
export PYTHONPATH="$HERE:$HERE/MixViT:${PYTHONPATH:-}"

VIDEO_DIR="${1:-/mnt/bum/hanyi/repo/MagicMotion/magicmotion_gen/final_output}"
GT_LIST="${2:-/mnt/bum/hanyi/repo/ATI/test_subset_100.txt}"
CKPT="${3:-pretrained/yolox_x_sports_train.pth.tar}"

EXP_FILE="$HERE/exps/example/mot/yolox_x_sportsmot.py"
OUTPUT_DIR="${VIDEO_DIR}/eval_results"
NUM_GPUS="${NUM_GPUS:-8}"

echo "============================================================"
echo "T7 Basketball Tracker + mIoU"
echo "============================================================"
echo "VIDEO_DIR: $VIDEO_DIR"
echo "GT_LIST:   $GT_LIST"
echo "CKPT:      $CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NUM_GPUS:  $NUM_GPUS"
echo ""

SPLIT_DIR="$(mktemp -d /tmp/svi_t7_eval_split.XXXXXX)"
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
echo ">>> Tracker done. Running video_miou.py to compute final score..."
python "$HERE/video_miou.py" \
    --video_dir "$VIDEO_DIR" \
    --gt_list "$GT_LIST" \
    --eval_results_dir "$OUTPUT_DIR"

echo ""
echo ">>> mIoU summary: $VIDEO_DIR/video_miou_results/summary.json"
