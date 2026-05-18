#!/bin/bash
# ============================================
# Experiment Submission Wrapper
# ============================================
# Creates experiment-specific log directories and submits SLURM jobs.
#
# Usage:
#   ./scripts/submit_experiment.sh <arch> [options]
#
# Examples:
#   ./scripts/submit_experiment.sh gpt5
#   ./scripts/submit_experiment.sh gpt5 --name my_ablation
#   ./scripts/submit_experiment.sh qwen3_omni_30b --service-node <node-A>
#   ./scripts/submit_experiment.sh qwen3_omni_30b --service-node <node-B> --port-offset 100
# ============================================

set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"

SCRIPT_DIR=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")
PIPELINE_DIR=$(readlink -f "$SCRIPT_DIR/..")

# Defaults
ARCH=""
CUSTOM_NAME=""
NUM_WORKERS=10
TOTAL_QUESTIONS=""  # auto-detect from --questions-file if not set explicitly
QUESTIONS_FILE=""   # required via --questions-file or QUESTIONS_FILE env var
SPORT=""            # required via --sport or SPORT env var
GPU_CONFIG="8gpu"
PORT_OFFSET="${PORT_OFFSET:-0}"

# Node placement is driven by env vars (not flags):
#   T9_TOOL_SERVER_HOST   — hostname of tools-services node; worker redirects tool
#                        calls there and is co-located on that node via --nodelist.
#   T9_AGENT_SERVER_HOST  — hostname of agent-vLLM node; worker redirects agent
#                        calls there. (Multi-node archs only: qwen3_235b, minimax_m2_5.)
# Both flow through to run_batch.py via `sbatch --export=ALL` below.

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            CUSTOM_NAME="$2"
            shift 2
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --questions)
            TOTAL_QUESTIONS="$2"
            shift 2
            ;;
        --questions-file)
            QUESTIONS_FILE="$2"
            shift 2
            ;;
        --sport)
            SPORT="$2"
            shift 2
            ;;
        --gpu-config)
            GPU_CONFIG="$2"
            shift 2
            ;;
        --port-offset)
            PORT_OFFSET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: [T9_TOOL_SERVER_HOST=<node>] [T9_AGENT_SERVER_HOST=<node>] $0 <arch> [options]"
            echo ""
            echo "Arguments:"
            echo "  arch                  Architecture id (e.g., gpt5, qwen3_32b, qwen3_235b, minimax_m2_5)"
            echo ""
            echo "Options:"
            echo "  --name NAME           Optional name prefix. Default: ${arch}_${sport}_${gpu}_${ts}"
            echo "  --workers N           Number of SLURM array workers (default: 10)"
            echo "  --questions N         Total questions to process (default: auto-detected from --questions-file)"
            echo "  --questions-file F    Path to questions JSON file (REQUIRED; e.g., \$T9_ROOT/questions/hockey.json)"
            echo "  --sport SPORT         Sport filter (REQUIRED; basketball|hockey|soccer)"
            echo "  --gpu-config CFG      Maps to slurm/run_benchmark_<CFG>.slurm (default: 8gpu; options: 8gpu, h100)"
            echo "  --port-offset N       Port offset for parallel experiments (default: 0)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Env vars (set BEFORE invoking):"
            echo "  T9_TOOL_SERVER_HOST=<hostname>   Hostname of tools-services node. Workers will"
            echo "                                redirect tool calls there AND be co-located on that"
            echo "                                node via --nodelist."
            echo "  T9_AGENT_SERVER_HOST=<hostname>  Hostname of agent-vLLM node. Workers will redirect"
            echo "                                agent calls there. Set only for multi-node archs"
            echo "                                (qwen3_235b, minimax_m2_5)."
            echo ""
            echo "Single-node usage (gpt5/qwen3_32b/qwen3_omni_30b): no env vars needed."
            echo "Multi-node usage (qwen3_235b/minimax_m2_5):"
            echo "  export T9_TOOL_SERVER_HOST=<tools-node>"
            echo "  export T9_AGENT_SERVER_HOST=<agent-node>"
            echo "  $0 qwen3_235b --questions-file ..."
            exit 0
            ;;
        *)
            if [[ -z "$ARCH" ]]; then
                ARCH="$1"
            else
                echo "Error: Unknown option $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arch
if [[ -z "$ARCH" ]]; then
    echo "Error: Architecture required"
    echo "Usage: $0 <arch> [options]"
    exit 1
fi

ARCH_FILE="$PIPELINE_DIR/archs/${ARCH}.yaml"
if [[ ! -f "$ARCH_FILE" ]]; then
    echo "Error: Architecture file not found: $ARCH_FILE"
    exit 1
fi

# Validate questions file
if [[ -z "$QUESTIONS_FILE" ]]; then
    echo "Error: --questions-file is required (or set QUESTIONS_FILE env var)."
    echo "       e.g.  --questions-file \$T9_ROOT/questions/hockey.json"
    exit 1
fi
if [[ ! -f "$QUESTIONS_FILE" ]]; then
    echo "Error: Questions file not found: $QUESTIONS_FILE"
    exit 1
fi

# Validate sport
if [[ -z "$SPORT" ]]; then
    echo "Error: --sport is required (basketball|hockey|soccer)."
    echo "       Without it, search calls would default to basketball indices regardless of question content."
    exit 1
fi
case "$SPORT" in
    basketball|hockey|soccer) ;;
    *) echo "Error: --sport must be basketball, hockey, or soccer (got: $SPORT)"; exit 1 ;;
esac

# Auto-detect TOTAL_QUESTIONS from the questions file if not passed explicitly.
# This way `--workers N` evenly splits the actual file and no worker idles on
# an out-of-range slice.
if [[ -z "$TOTAL_QUESTIONS" ]]; then
    TOTAL_QUESTIONS=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d) if isinstance(d, list) else len(d.get('questions', d)))" "$QUESTIONS_FILE")
fi

# Generate experiment name. Default: ${arch}_${sport}_${gpu}_${timestamp},
# which lets the judge step glob ${arch}_${sport}_* to find all runs for a
# given (arch, sport) pair regardless of timestamp.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$CUSTOM_NAME" ]]; then
    EXPERIMENT_NAME="${CUSTOM_NAME}_${ARCH}_${SPORT}_${GPU_CONFIG}_${TIMESTAMP}"
else
    EXPERIMENT_NAME="${ARCH}_${SPORT}_${GPU_CONFIG}_${TIMESTAMP}"
fi

# Create experiment directory structure under T9_ROOT/results/ — the canonical
# location used by run_batch.py:setup_experiment (matches paths.yaml:experiment_log_dir).
# Auto-detect T9_ROOT if not exported.
if [ -z "${T9_ROOT:-}" ]; then
    T9_ROOT="$(cd "$PIPELINE_DIR/../../.." && pwd)/data/t9"
fi
EXPERIMENT_DIR="$T9_ROOT/results/$EXPERIMENT_NAME"
LOGS_DIR="$EXPERIMENT_DIR/logs"

mkdir -p "$LOGS_DIR"
mkdir -p "$EXPERIMENT_DIR/results"

# Select SLURM script
SLURM_SCRIPT="$PIPELINE_DIR/slurm/run_benchmark_${GPU_CONFIG}.slurm"
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Error: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

# Export environment variables for sbatch --export=ALL passthrough.
# T9_TOOL_SERVER_HOST / T9_AGENT_SERVER_HOST are inherited from the caller's shell
# (not re-set here). run_batch.py reads them directly.
export EXPERIMENT_NAME
export ARCH
export QUESTIONS_FILE
export TOTAL_QUESTIONS
export PORT_OFFSET
export SPORT

echo "==========================================="
echo "Submitting Experiment: $EXPERIMENT_NAME"
echo "==========================================="
echo "Architecture:      $ARCH"
echo "GPU Config:        $GPU_CONFIG"
echo "Workers:           $NUM_WORKERS"
echo "Questions file:    $QUESTIONS_FILE"
echo "Questions total:   $TOTAL_QUESTIONS"
echo "T9_TOOL_SERVER_HOST:  ${T9_TOOL_SERVER_HOST:-<unset; tool servers expected on localhost>}"
echo "T9_AGENT_SERVER_HOST: ${T9_AGENT_SERVER_HOST:-<unset; agent server expected on localhost>}"
echo "Port Offset:       $PORT_OFFSET"
echo "Logs Directory:    $LOGS_DIR"
echo "==========================================="

# Worker co-locates on the tools node if T9_TOOL_SERVER_HOST is set (tool calls
# become LAN-local; agent calls cross-node to T9_AGENT_SERVER_HOST when set).
NODELIST_ARG=""
[[ -n "${T9_TOOL_SERVER_HOST:-}" ]] && NODELIST_ARG="--nodelist=$T9_TOOL_SERVER_HOST"

JOB_NAME="t9_worker_${ARCH}"

# Submit with experiment-specific log paths
JOB_ID=$(sbatch \
    --parsable \
    --job-name="$JOB_NAME" \
    $NODELIST_ARG \
    --array=0-$((NUM_WORKERS - 1)) \
    --output="$LOGS_DIR/slurm_%A_%a.out" \
    --error="$LOGS_DIR/slurm_%A_%a.err" \
    --export=ALL \
    "$SLURM_SCRIPT")

echo ""
echo "Submitted batch job: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f $LOGS_DIR/slurm_${JOB_ID}_*.out"
echo ""
echo "Results will be in:"
echo "  $EXPERIMENT_DIR/results/"
