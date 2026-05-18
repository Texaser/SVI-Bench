#!/bin/bash
# ============================================
# Services Submission Wrapper
# ============================================
# Starts GPU services with optional experiment-specific logging.
#
# Usage:
#   ./scripts/submit_services.sh <arch> [options]
#
# Examples:
#   ./scripts/submit_services.sh gpt5
#   ./scripts/submit_services.sh qwen3_omni_30b --node <node-A>
#   ./scripts/submit_services.sh qwen3_omni_30b --node <node-B> --port-offset 100
# ============================================

set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"

SCRIPT_DIR=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")
PIPELINE_DIR=$(readlink -f "$SCRIPT_DIR/..")

# Defaults
ARCH=""
GPU_CONFIG="8gpu"
NO_FLASH_ATTN=""
NODE=""
PORT_OFFSET="${PORT_OFFSET:-0}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in

        --gpu-config)
            GPU_CONFIG="$2"
            shift 2
            ;;
        --no-flash-attn)
            NO_FLASH_ATTN="1"
            shift
            ;;
        --node)
            NODE="$2"
            shift 2
            ;;
        --port-offset)
            PORT_OFFSET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <arch> [options]"
            echo ""
            echo "Arguments:"
            echo "  arch                   Architecture id (e.g., gpt5, qwen3_32b, qwen3_235b_tools, minimax_m2_5_tools)"
            echo ""
            echo "Options:"
            echo "  --gpu-config CFG       Maps to slurm/start_services_<CFG>.slurm (default: 8gpu; options: 8gpu, h100)"
            echo "  --node NODE            Target node (any reachable hostname). If not set, SLURM picks."
            echo "  --port-offset N        Port offset for parallel experiments on same node (default: 0)"
            echo "  --no-flash-attn        Disable FlashAttention for all services"
            echo "  -h, --help             Show this help message"
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

# Determine logs directory
# Always use shared logs directory for services
LOGS_DIR="$PIPELINE_DIR/logs"

mkdir -p "$LOGS_DIR"

# Select SLURM script
SLURM_SCRIPT="$PIPELINE_DIR/slurm/start_services_${GPU_CONFIG}.slurm"
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Error: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

# Export environment variables
export ARCH
export LOGS_DIR
export NO_FLASH_ATTN
export PORT_OFFSET

echo "==========================================="
echo "Starting GPU Services"
echo "==========================================="
echo "Architecture:    $ARCH"
echo "GPU Config:      $GPU_CONFIG"
echo "Node:            ${NODE:-<SLURM-assigned>}"
echo "Port Offset:     $PORT_OFFSET"
echo "FlashAttention:  ${NO_FLASH_ATTN:+disabled}${NO_FLASH_ATTN:-enabled}"
echo "Logs Directory:  $LOGS_DIR"
echo "==========================================="

# Build optional sbatch args
NODELIST_ARG=""
[[ -n "$NODE" ]] && NODELIST_ARG="--nodelist=$NODE"

# Job name reflects role (tool / agent) so squeue is easier to scan
case "$ARCH" in
    *_tools) JOB_ROLE="tool" ;;
    gpt5*)   JOB_ROLE="tool" ;;   # API-based agent — services job is tools-only
    *)       JOB_ROLE="agent" ;;
esac
JOB_NAME="t9_${JOB_ROLE}_${ARCH}"

# Submit with log paths
JOB_ID=$(sbatch \
    --parsable \
    --job-name="$JOB_NAME" \
    $NODELIST_ARG \
    --output="$LOGS_DIR/services_${GPU_CONFIG}_%j.out" \
    --error="$LOGS_DIR/services_${GPU_CONFIG}_%j.err" \
    --export=ALL \
    "$SLURM_SCRIPT")

echo ""
echo "Submitted services job: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f $LOGS_DIR/services_${GPU_CONFIG}_${JOB_ID}.out"
echo ""
echo "Service logs will be in:"
echo "  $LOGS_DIR/vllm_*.log"
echo ""
echo "When submitting experiment workers, pass the service node:"
echo "  scontrol show job $JOB_ID | grep ' NodeList'   # find the node"
echo "  ./scripts/submit_experiment.sh $ARCH --service-node <node> --port-offset $PORT_OFFSET"
