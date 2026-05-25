#!/usr/bin/env bash
# Download the T7 motion-conditioned-generation LoRA checkpoint
# from the MVP-Group/SVI-Bench HF dataset.
#
# Usage: bash download_checkpoint.sh {basketball|soccer}
#
# After download, run inference:
#   python inference/{basketball,soccer}.py checkpoints/T7/{basketball,soccer}/checkpoint.safetensors
set -euo pipefail

SPORT=${1:-}
if [[ "$SPORT" != "basketball" && "$SPORT" != "soccer" ]]; then
  echo "Usage: bash download_checkpoint.sh {basketball|soccer}" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

hf download MVP-Group/SVI-Bench \
  "T7/${SPORT}/checkpoint.safetensors" \
  --repo-type dataset \
  --local-dir checkpoints

echo "✓ Downloaded to $SCRIPT_DIR/checkpoints/T7/${SPORT}/checkpoint.safetensors"
