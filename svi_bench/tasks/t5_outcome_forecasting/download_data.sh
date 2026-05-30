#!/usr/bin/env bash
# Download T5 Outcome Forecasting data from HuggingFace
#
# Usage:
#   bash download_data.sh                    # downloads to ./data/
#   bash download_data.sh /path/to/dir       # downloads to /path/to/dir/

set -euo pipefail

LOCAL_DIR="${1:-data}"

echo "Downloading T5 data to ${LOCAL_DIR}/T5/ ..."

huggingface-cli download MVP-Group/SVI-Bench \
    --repo-type dataset \
    --include "T5/**" \
    --local-dir "$LOCAL_DIR"

echo "Done. Data saved to ${LOCAL_DIR}/T5/"
