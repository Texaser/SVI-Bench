#!/usr/bin/env bash
# Download T7 and T8 data for SVI-Bench from HuggingFace
# (MVP-Group/SVI-Bench) and extract the tar bundles.
#
# Final layout (under $SVI_BENCH_DATA, default: ./data/):
#   data/T7/{soccer,basketball}/{clips,bboxes,backgrounds}/{00..99}/{ID}.{mp4,txt}
#   data/T7/{soccer,basketball}/splits/{train,val,test}_final.txt   (IDs only)
#   data/T8/basketball/{clips,bboxes,backgrounds}/{00..99}/{ID}.{mp4,txt}
#   data/T8/basketball/splits/{train,val,test}_task2_final.txt      (IDs only)
#   data/T8/basketball/captions.json                                (id -> refined_instruction + player_specifications)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
DATA_ROOT="${SVI_BENCH_DATA:-$REPO_ROOT/data}"
mkdir -p "$DATA_ROOT"

REPO_ID="MVP-Group/SVI-Bench"

echo "Downloading T7/T8 from ${REPO_ID} to ${DATA_ROOT} ..."
python3 - "$DATA_ROOT" "$REPO_ID" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
DATA_ROOT, REPO_ID = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=DATA_ROOT,
    allow_patterns=["T7/**", "T8/**"],
    max_workers=8,
)
print("Snapshot download complete.")
PY

echo ""
echo "Extracting tar bundles (deletes tars after success) ..."
python3 "$HERE/extract_tars.py" --root "$DATA_ROOT/T7" --delete-after
python3 "$HERE/extract_tars.py" --root "$DATA_ROOT/T8" --delete-after

echo ""
echo "Done. Data is at: $DATA_ROOT"
echo "Set 'export SVI_BENCH_DATA=$DATA_ROOT' before running task train/eval scripts."
