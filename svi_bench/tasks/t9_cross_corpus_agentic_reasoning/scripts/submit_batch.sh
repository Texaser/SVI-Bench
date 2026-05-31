#!/bin/bash
# T9 batch entry — non-SLURM wrapper around run_batch.main_as_function.
# Runs the worker locally (no scheduler).
#
# Env vars (override defaults):
#   ARCH                 — arch id from KNOWN_ARCHS (default: gpt5)
#   QUESTIONS            — path to questions JSON (REQUIRED; e.g., $T9_ROOT/questions/hockey.json)
#   SPORT                — basketball|hockey|soccer (default: unset = all sports)
#   EXPERIMENT_NAME      — experiment name (default: t9_batch_<arch>_<timestamp>)
#   T9_ROOT              — data root (auto-detected from <repo>/data/T9 if unset)
#   T9_ES_URL            — Elasticsearch URL (default: http://localhost:9200)
#   T9_TOOL_SERVER_HOST     — hostname of tool-services node (default: localhost)
#   T9_AGENT_SERVER_HOST    — hostname of agent vLLM node (default: localhost)
#   OPENAI_API_KEY       — required for the GPT-4o judge

set -euo pipefail

ARCH="${ARCH:-gpt5}"
SPORT="${SPORT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"

# Auto-detect T9_ROOT if not set
if [ -z "${T9_ROOT:-}" ]; then
    _dir="$TASK_DIR"
    while [ "$_dir" != "/" ] && [ ! -f "$_dir/pyproject.toml" ]; do
        _dir="$(dirname "$_dir")"
    done
    if [ -d "$_dir/data/T9" ]; then
        T9_ROOT="$_dir/data/T9"
    elif [ -d "$_dir/data/t9" ]; then
        T9_ROOT="$_dir/data/t9"
    else
        echo "ERROR: T9_ROOT not set and could not auto-detect <repo>/data/T9." >&2
        exit 1
    fi
fi
export T9_ROOT

# QUESTIONS is required — release ships split=all only, no implicit filename pattern
if [ -z "${QUESTIONS:-}" ]; then
    echo "ERROR: QUESTIONS env var is required." >&2
    echo "       e.g.  QUESTIONS=\$T9_ROOT/questions/hockey.json" >&2
    exit 1
fi
if [ ! -f "$QUESTIONS" ]; then
    echo "ERROR: Questions file not found: $QUESTIONS" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-t9_batch_${ARCH}_${TIMESTAMP}}"
OUTPUT_DIR="${T9_ROOT}/results/${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"

SPORT_ARG=""
[ -n "$SPORT" ] && SPORT_ARG="--sport $SPORT"

cd "$TASK_DIR"

python -m svi_bench.tasks.t9_cross_corpus_agentic_reasoning.run_batch \
    --questions "$QUESTIONS" \
    --arch "$ARCH" \
    --experiment-name "$EXPERIMENT_NAME" \
    --worker-id 0 \
    $SPORT_ARG \
    2>&1 | tee "${OUTPUT_DIR}/batch.log"
