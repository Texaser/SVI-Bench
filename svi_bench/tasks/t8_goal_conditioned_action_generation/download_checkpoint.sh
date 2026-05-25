#!/usr/bin/env bash
# Download the T8 goal-conditioned-action-generation LoRA checkpoint
# (basketball only) from the MVP-Group/SVI-Bench HF dataset.
#
# Usage: bash download_checkpoint.sh
#
# After download, run inference:
#   python inference/basketball.py checkpoints/T8/basketball/checkpoint.safetensors
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

hf download MVP-Group/SVI-Bench \
  "T8/basketball/checkpoint.safetensors" \
  --repo-type dataset \
  --local-dir checkpoints

echo "✓ Downloaded to $SCRIPT_DIR/checkpoints/T8/basketball/checkpoint.safetensors"
