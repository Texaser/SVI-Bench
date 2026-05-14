#!/usr/bin/env bash
# T8 — basketball goal accuracy via fine-tuned LLaVA-Qwen video-language QA.
#
# Pipeline assumption: the QA source dir already has per-QA-type Q*.json
# files + the pre-rendered bbox-overlay videos that the JSONs reference.
# The dataset team produces those — see the per-task README for the HF
# dataset path.
#
# Usage:
#   bash eval/run_basketball_goalacc.sh \
#       <VIDEO_DIR>     -- flat dir of generated <clip>.mp4 (your method's outputs)
#       <QA_SOURCE>     -- dir containing Q*.json files (and a copy of the
#                          rendered bbox-overlay videos the JSONs point at)
#       <MODEL_PATH>    -- fine-tuned LLaVA-Qwen checkpoint dir
#                          (e.g. .../basketball_bbox_qa_f16_full_ft_h100/checkpoint-15500)
#
# Or set them as env vars (VIDEO_DIR / QA_SOURCE / MODEL_PATH) and call
# the script with no positional args.
#
# Output: $OUTPUT_DIR/<qa_type>/qa_eval_f${EVAL_FRAMES}_results.json
# Each file's `overall.accuracy` is the per-QA-type goal accuracy; we
# print a summary table at the end.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
cd "$HERE"

VIDEO_DIR="${1:-${VIDEO_DIR:-}}"
QA_SOURCE="${2:-${QA_SOURCE:-}}"
MODEL_PATH="${3:-${MODEL_PATH:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${VIDEO_DIR}/goal_accuracy_results}"

MODEL_NAME="${MODEL_NAME:-llava_qwen}"
EVAL_FRAMES="${EVAL_FRAMES:-16}"

if [ -z "$VIDEO_DIR" ] || [ -z "$QA_SOURCE" ] || [ -z "$MODEL_PATH" ]; then
    echo "Error: VIDEO_DIR / QA_SOURCE / MODEL_PATH all required (positional 1-3 or env vars)." >&2
    echo "Example:" >&2
    echo "  bash eval/run_basketball_goalacc.sh \\" >&2
    echo "       /path/to/generated_videos \\" >&2
    echo "       /path/to/QA/llava_format/test_final \\" >&2
    echo "       /path/to/basketball_bbox_qa_f16_full_ft/checkpoint-15500" >&2
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH does not exist: $MODEL_PATH" >&2
    echo "Fine-tuned LLaVA-Qwen checkpoint (~15 GB) is not bundled — see eval/pretrained/README.md." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "T8 Basketball Goal Accuracy (LLaVA-Qwen QA)"
echo "============================================================"
echo "VIDEO_DIR:    $VIDEO_DIR"
echo "QA_SOURCE:    $QA_SOURCE"
echo "MODEL_PATH:   $MODEL_PATH"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "MODEL_NAME:   $MODEL_NAME"
echo "EVAL_FRAMES:  $EVAL_FRAMES"
echo ""

# Iterate over QA-type JSONs (Q1*.json, Q2*.json, ...). test_llavaov.py
# internally fans out across visible GPUs via torch.multiprocessing, so
# we run one QA-type at a time sequentially.
SHOPT_COUNT=0
for json_file in "${QA_SOURCE}"/Q*.json; do
    [ -f "$json_file" ] || continue
    SHOPT_COUNT=$((SHOPT_COUNT + 1))
    json_name=$(basename "${json_file}" .json)
    echo ">>> [$SHOPT_COUNT] ${json_name}"

    python "$HERE/test_llavaov.py" \
        --model_name "${MODEL_NAME}" \
        --model_path "${MODEL_PATH}" \
        --test_json_path "${json_file}" \
        --results_dir "${OUTPUT_DIR}/${json_name}" \
        --eval_type qa \
        --eval_frames "${EVAL_FRAMES}" \
        --max_samples 0
done

if [ "$SHOPT_COUNT" -eq 0 ]; then
    echo "Error: no Q*.json files found in $QA_SOURCE" >&2
    exit 1
fi

# ============================================================
# Summary table
# ============================================================
echo ""
echo "============================================================"
echo "GOAL ACCURACY SUMMARY"
echo "============================================================"

for results_json in "${OUTPUT_DIR}"/*/qa_eval_f${EVAL_FRAMES}_results.json; do
    [ -f "$results_json" ] || continue
    qa_type=$(basename "$(dirname "$results_json")")
    overall=$(python3 -c "
import json, sys
d = json.load(open('$results_json'))
o = d['overall']
print(f\"{o['accuracy']:.4f} ({o['correct']}/{o['total']})\")
" 2>/dev/null) || overall="(parse error)"
    printf "  %-35s %s\n" "$qa_type" "$overall"
done

echo ""
echo "============================================================"
echo "Done. Per-QA-type JSONs in: $OUTPUT_DIR/"
echo "============================================================"
