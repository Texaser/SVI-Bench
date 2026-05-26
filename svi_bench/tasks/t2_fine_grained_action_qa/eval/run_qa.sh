#!/usr/bin/env bash
# T2 multi-choice QA evaluation. Uses the T1+T2 jointly-trained LLaVA-Video
# checkpoint and the shared eval_sports.py worker that lives under T1.
#
# Usage:
#   bash eval/run_qa.sh [MODEL_PATH] [RESULTS_DIR]
#
# MODEL_PATH defaults to T1/checkpoint/ under $SVI_BENCH_DATA (T1+T2 share
# the same checkpoint; see T1 README for training).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
T1_DIR="$REPO_ROOT/svi_bench/tasks/t1_structured_play_description"
DATA_ROOT="${SVI_BENCH_DATA:-$REPO_ROOT/data}"
export PYTHONPATH="$T1_DIR:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-llava_qwen}"
MODEL_PATH="${1:-${MODEL_PATH:-$DATA_ROOT/T1/checkpoint}}"
RESULTS_DIR="${2:-${RESULTS_DIR:-./results/T2}}"
DATA_DIR="${DATA_DIR:-$DATA_ROOT/T2/data}"
EVAL_FRAMES="${EVAL_FRAMES:-16}"
SPLIT="${SPLIT:-val}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH does not exist: $MODEL_PATH" >&2
    echo "Download via: huggingface-cli download MVP-Group/SVI-Bench --include 'T1/checkpoint/*' --local-dir \$SVI_BENCH_DATA" >&2
    exit 1
fi

for sport in basketball soccer hockey; do
    python "$T1_DIR/llava/eval/eval_sports.py" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --test_json_path "$DATA_DIR/${sport}_qa_${SPLIT}_10k.json" \
        --results_dir "$RESULTS_DIR/${sport}/${SPLIT}" \
        --eval_type qa \
        --eval_frames "$EVAL_FRAMES"
done

echo "QA results written under: $RESULTS_DIR/{basketball,soccer,hockey}/${SPLIT}/"
