#!/usr/bin/env bash
# T1 LLM-as-a-judge scoring on already-generated captions.
# Run AFTER `run_caption.sh` — it consumes <results_dir>/<sport>/<split>/caption_eval_f16_outputs.json.
#
# Usage:
#   bash eval/run_llm_judge.sh [JUDGE_MODEL] [RESULTS_DIR]
#
# Set OPENAI_API_KEY / GOOGLE_API_KEY etc. matching the judge.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
export PYTHONPATH="$TASK_DIR:${PYTHONPATH:-}"

JUDGE_MODEL="${1:-${JUDGE_MODEL:-gpt-5.2-2025-12-11}}"
RESULTS_DIR="${2:-${RESULTS_DIR:-./results/T1}}"
EVAL_FRAMES="${EVAL_FRAMES:-16}"
SPLIT="${SPLIT:-val}"

for sport in basketball soccer hockey; do
    in_path="$RESULTS_DIR/${sport}/${SPLIT}/caption_eval_f${EVAL_FRAMES}_outputs.json"
    if [ ! -f "$in_path" ]; then
        echo "Skip $sport: $in_path not found (run run_caption.sh first)" >&2
        continue
    fi
    python "$TASK_DIR/llava/eval/eval_llm_judge.py" \
        --model_name "$JUDGE_MODEL" \
        --sport "$sport" \
        --test_json_path "$in_path"
done
