#!/usr/bin/env bash
# T8 — basketball goal accuracy via fine-tuned LLaVA-Qwen video-language QA.
#
# Pipeline:
#   1. Filter the anonymized master QA at $QA_MASTER (downloaded from
#      MVP-Group/SVI-Bench:T8/basketball/qa_test/) to keep only entries
#      whose anonymized clip ID has a matching <anon_id>.mp4 under VIDEO_DIR.
#   2. Render a red bbox-overlay on frame 0 of each kept generated video
#      using the start_bbox supplied in the master QA. These overlay copies
#      land in $PREPARED_DIR/rendered_videos/.
#   3. Write per-question-type Q*.json under $PREPARED_DIR/qa_json/ with
#      `video` paths pointing at the rendered overlays.
#   4. Loop over those Q*.json files and run test_llavaov.py for each
#      (multi-GPU fan-out happens inside test_llavaov.py).
#
# Steps 1-3 are skipped if $PREPARED_DIR/qa_json/ already has Q*.json files.
#
# Usage:
#   bash eval/run_basketball_goalacc.sh <VIDEO_DIR> [QA_MASTER] [MODEL_PATH]
#
# QA_MASTER and MODEL_PATH default to $SVI_BENCH_DATA/T8/{basketball/qa_test,llava_qa_checkpoint},
# i.e. the layout produced by scripts/download_t7_t8.sh.
#
# Output: $OUTPUT_DIR/<qa_type>/qa_eval_f${EVAL_FRAMES}_results.json
# Each file's `overall.accuracy` is the per-QA-type goal accuracy; we
# print a summary table at the end.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
DATA_ROOT="${SVI_BENCH_DATA:-$REPO_ROOT/data}"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
cd "$HERE"

VIDEO_DIR="${1:-${VIDEO_DIR:-}}"
QA_MASTER="${2:-${QA_MASTER:-$DATA_ROOT/T8/basketball/qa_test}}"
MODEL_PATH="${3:-${MODEL_PATH:-$DATA_ROOT/T8/llava_qa_checkpoint}}"
PREPARED_DIR="${PREPARED_DIR:-${VIDEO_DIR}/qa_prepared}"
QA_SOURCE="${PREPARED_DIR}/qa_json"
OUTPUT_DIR="${OUTPUT_DIR:-${VIDEO_DIR}/goal_accuracy_results}"

MODEL_NAME="${MODEL_NAME:-llava_qwen}"
EVAL_FRAMES="${EVAL_FRAMES:-16}"

if [ -z "$VIDEO_DIR" ]; then
    echo "Error: VIDEO_DIR required (positional 1 or env var)." >&2
    echo "Example:" >&2
    echo "  bash eval/run_basketball_goalacc.sh /path/to/generated_videos" >&2
    echo "" >&2
    echo "QA_MASTER and MODEL_PATH default to \$SVI_BENCH_DATA/T8/..." >&2
    echo "(downloaded by scripts/download_t7_t8.sh). Override via positional" >&2
    echo "args 2-3 or env vars if your layout differs." >&2
    exit 1
fi

if [ ! -d "$QA_MASTER" ]; then
    echo "Error: QA_MASTER does not exist: $QA_MASTER" >&2
    echo "Run scripts/download_t7_t8.sh to fetch the T8 QA master set." >&2
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH does not exist: $MODEL_PATH" >&2
    echo "Run scripts/download_t7_t8.sh to fetch the fine-tuned LLaVA-Qwen checkpoint" >&2
    echo "(~15 GB, under T8/llava_qa_checkpoint/ on MVP-Group/SVI-Bench)." >&2
    exit 1
fi

# Prepare method-specific QA bundle (filter master + render bbox overlays)
# only if it hasn't been produced yet for this VIDEO_DIR.
if [ ! -d "$QA_SOURCE" ] || [ -z "$(ls -A "$QA_SOURCE" 2>/dev/null)" ]; then
    echo "Preparing method-specific QA bundle at $PREPARED_DIR ..."
    python "$HERE/prepare_qa_for_method.py" \
        --video_dir "$VIDEO_DIR" \
        --qa_dir    "$QA_MASTER" \
        --output_dir "$PREPARED_DIR" \
        --skip_existing
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
