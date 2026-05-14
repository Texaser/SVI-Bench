#!/usr/bin/env bash
# T7 — soccer post-generation mIoU evaluation.
# Same pipeline as run_basketball.sh but loads the soccernet YOLOX exp config
# and points at the soccer test set.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="$HERE:$HERE/MixViT:${PYTHONPATH:-}"

VIDEO_DIR="${1:-/mnt/bum/hanyi/repo/MagicMotion/magicmotion_gen_soccer/final_output}"
GT_LIST="${2:-/mnt/bum/hanyi/repo/ATI/test_subset_soccer_100.txt}"
CKPT="${3:-$REPO_ROOT/pretrained/yolox_x_sports_train.pth.tar}"

EXP_FILE="$HERE/exps/example/mot/yolox_x_soccernet.py"
OUTPUT_DIR="${VIDEO_DIR}/eval_results"
NUM_GPUS="${NUM_GPUS:-8}"

echo "============================================================"
echo "T7 Soccer Tracker + mIoU"
echo "============================================================"
echo "VIDEO_DIR: $VIDEO_DIR"
echo "GT_LIST:   $GT_LIST"
echo "CKPT:      $CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NUM_GPUS:  $NUM_GPUS"
echo ""

SPLIT_DIR="$(mktemp -d /tmp/svi_t7_soccer_eval_split.XXXXXX)"
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
echo ">>> Tracker done. Running video_miou.py..."
python "$HERE/video_miou.py" \
    --video_dir "$VIDEO_DIR" \
    --gt_list "$GT_LIST" \
    --eval_results_dir "$OUTPUT_DIR"

echo ""
echo ">>> mIoU summary: $VIDEO_DIR/video_miou_results/summary.json"
