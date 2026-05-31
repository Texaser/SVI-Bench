#!/bin/bash
# T9 environment setup. Source this before running any T9 entry point.
#
# Prerequisites (set in your shell, e.g. ~/.bashrc):
#   export CONDA_PROFILE=<path-to>/miniconda3/etc/profile.d/conda.sh
#   export CONDA_ENV=<name-of-your-env>   # optional, defaults to svi-bench
#   export OPENAI_API_KEY=<your-key>      # required for the LLM judge
#
# What this sets up:
#   - Activates the conda env
#   - Exports T9_REPO / T9_TASK / T9_ROOT
#   - Cds to T9_TASK
#   - Defines helper bash functions: run, wait_running, wait_done, job_id
#   - Creates /tmp/t9_sanity_logs/
#
# Usage:
#   source <task-dir>/scripts/setup_env.sh

if [ -z "${CONDA_PROFILE:-}" ]; then
    echo "ERROR: CONDA_PROFILE env var not set." >&2
    echo "       export it before sourcing this script:" >&2
    echo "         export CONDA_PROFILE=/path/to/miniconda3/etc/profile.d/conda.sh" >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -f "$CONDA_PROFILE" ]; then
    echo "ERROR: CONDA_PROFILE points at non-existent file: $CONDA_PROFILE" >&2
    return 1 2>/dev/null || exit 1
fi

source "$CONDA_PROFILE"
conda activate "${CONDA_ENV:-svi-bench-t9}"

# Resolve task / repo / data roots from this script's location
T9_TASK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T9_REPO="$(cd "$T9_TASK/../../.." && pwd)"
export T9_TASK T9_REPO
if [ -d "$T9_REPO/data/T9" ]; then
    export T9_ROOT="$T9_REPO/data/T9"
elif [ -d "$T9_REPO/data/t9" ]; then
    export T9_ROOT="$T9_REPO/data/t9"
else
    echo "WARNING: could not find data/T9 or data/t9 under $T9_REPO" >&2
fi
cd "$T9_TASK"

# Logging helper: `run <tag> <cmd...>` tees stdout+stderr to a per-tag logfile
mkdir -p /tmp/t9_sanity_logs
run() { local tag="$1"; shift; { echo "$ $*"; "$@"; } 2>&1 | tee /tmp/t9_sanity_logs/${tag}.log; }

# SLURM helpers
wait_running() {
    local jid=$1
    while :; do
        local s=$(squeue -h -j "$jid" -o "%T %N" 2>/dev/null)
        [ -z "$s" ] && { echo "job $jid not in queue (failed?)" >&2; return 1; }
        [ "${s%% *}" = "RUNNING" ] && { echo "${s##* }"; return 0; }
        sleep 5
    done
}
wait_done() { local jid=$1; while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 15; done; }
job_id()    { grep -oE '(services|batch) job: [0-9]+' "$1" | awk '{print $NF}'; }

echo "T9 env ready."
echo "  T9_REPO  : $T9_REPO"
echo "  T9_TASK  : $T9_TASK"
echo "  T9_ROOT  : $T9_ROOT"
echo "  conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "  helpers  : run, wait_running, wait_done, job_id"
