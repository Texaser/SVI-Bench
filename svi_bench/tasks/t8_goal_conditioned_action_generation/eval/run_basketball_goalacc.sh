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

# Prepare method-specific QA bundle (filter master + render bbox overlays).
# --skip_existing makes this idempotent: completed renders + Q*.json files
# are reused on subsequent runs, so an interrupted prepare resumes cleanly.
echo "Preparing method-specific QA bundle at $PREPARED_DIR ..."
python "$HERE/prepare_qa_for_method.py" \
    --video_dir "$VIDEO_DIR" \
    --qa_dir    "$QA_MASTER" \
    --output_dir "$PREPARED_DIR" \
    --skip_existing

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

# Pass all Q*.json files in a single test_llavaov.py invocation so the
# fine-tuned LLaVA-Qwen checkpoint loads once per GPU instead of once per
# question type. Per-Q*.json subdirs (qa_eval_f<frames>_{outputs,results}.json)
# are written by the worker.
shopt -s nullglob
JSON_FILES=("${QA_SOURCE}"/Q*.json)
shopt -u nullglob
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "Error: no Q*.json files found in $QA_SOURCE" >&2
    exit 1
fi

echo ">>> Running ${#JSON_FILES[@]} QA files with one model load per GPU"
python "$HERE/test_llavaov.py" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --test_json_paths "${JSON_FILES[@]}" \
    --results_dir "${OUTPUT_DIR}" \
    --eval_type qa \
    --eval_frames "${EVAL_FRAMES}" \
    --max_samples 0

# ============================================================
# Summary + aggregation
#
# Writes $OUTPUT_DIR/summary.json with per-QA-type accuracies plus two
# aggregate metrics:
#   - micro_accuracy = sum(correct) / sum(total)         (entry-weighted)
#   - macro_accuracy = mean of the per-type accuracies   (type-weighted)
# Headline goal-accuracy reported by SVI-Bench is the micro average
# (each QA entry contributes equally regardless of question type).
# ============================================================
echo ""
echo "============================================================"
echo "GOAL ACCURACY SUMMARY"
echo "============================================================"

python3 - "$OUTPUT_DIR" "$EVAL_FRAMES" <<'PY'
import json, os, sys, glob

output_dir, eval_frames = sys.argv[1], sys.argv[2]
per_type = {}
total_correct = total_count = 0
for results_json in sorted(glob.glob(os.path.join(output_dir, "*", f"qa_eval_f{eval_frames}_results.json"))):
    qa_type = os.path.basename(os.path.dirname(results_json))
    try:
        o = json.load(open(results_json))["overall"]
    except Exception as e:
        print(f"  {qa_type:<35} (parse error: {e})")
        continue
    acc, correct, total = o["accuracy"], o["correct"], o["total"]
    per_type[qa_type] = {"accuracy": acc, "correct": correct, "total": total}
    total_correct += correct
    total_count   += total
    print(f"  {qa_type:<35} {acc:.4f} ({correct}/{total})")

if not per_type:
    print("  (no Q*.json results found)")
    sys.exit(1)

micro = total_correct / total_count if total_count else 0.0
macro = sum(v["accuracy"] for v in per_type.values()) / len(per_type)
print()
print(f"  {'OVERALL (micro, entry-weighted)':<35} {micro:.4f} ({total_correct}/{total_count})")
print(f"  {'OVERALL (macro, type-weighted)':<35} {macro:.4f} ({len(per_type)} question types)")

summary = {
    "per_type":       per_type,
    "micro_accuracy": micro,
    "macro_accuracy": macro,
    "total_correct":  total_correct,
    "total_count":    total_count,
    "num_question_types": len(per_type),
}
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Wrote {os.path.join(output_dir, 'summary.json')}")
PY

echo ""
echo "============================================================"
echo "Done. Per-QA-type JSONs in: $OUTPUT_DIR/"
echo "Aggregate report:           $OUTPUT_DIR/summary.json"
echo "============================================================"
