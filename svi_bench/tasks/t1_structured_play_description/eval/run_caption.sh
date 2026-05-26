#!/usr/bin/env bash
# T1 caption generation: runs the fine-tuned LLaVA-Video model over the
# T1 val/test JSON splits and writes captions to <results_dir>/<sport>/<split>/.
#
# Usage:
#   bash eval/run_caption.sh [MODEL_PATH] [RESULTS_DIR]
#
# MODEL_PATH defaults to the HF-downloaded T1 checkpoint under $SVI_BENCH_DATA.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
DATA_ROOT="${SVI_BENCH_DATA:-$REPO_ROOT/data}"
export PYTHONPATH="$TASK_DIR:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-llava_qwen}"
MODEL_PATH="${1:-${MODEL_PATH:-$DATA_ROOT/T1/checkpoint}}"
RESULTS_DIR="${2:-${RESULTS_DIR:-./results/T1}}"
DATA_DIR="${DATA_DIR:-$DATA_ROOT/T1/captions}"
EVAL_FRAMES="${EVAL_FRAMES:-16}"
SPLIT="${SPLIT:-val}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH does not exist: $MODEL_PATH" >&2
    echo "Download via: huggingface-cli download MVP-Group/SVI-Bench --include 'T1/checkpoint/*' --local-dir \$SVI_BENCH_DATA" >&2
    exit 1
fi

for sport in basketball soccer hockey; do
    python "$TASK_DIR/llava/eval/eval_sports.py" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --test_json_path "$DATA_DIR/${sport}_caption_${SPLIT}_1k.json" \
        --results_dir "$RESULTS_DIR/${sport}/${SPLIT}" \
        --eval_type caption \
        --eval_frames "$EVAL_FRAMES" \
        --infer_only
done

echo "Captions written under: $RESULTS_DIR/{basketball,soccer,hockey}/${SPLIT}/"
