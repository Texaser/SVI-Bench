#!/usr/bin/env bash
# Download T7 and T8 data for SVI-Bench from HuggingFace
# (MVP-Group/SVI-Bench) and extract the tar bundles.
#
# Final layout:
#   $SVI_BENCH_DATA/T7/{soccer,basketball}/{clips,bboxes,backgrounds}/{00..99}/{ID}.{mp4,txt}
#   $SVI_BENCH_DATA/T7/{soccer,basketball}/splits/{train,val,test,test_100}.txt
#   $SVI_BENCH_DATA/T7/tracker_weights/{yolox,MixFormer}_sports_train.pth.tar
#   $SVI_BENCH_DATA/T8/basketball/{clips,bboxes,backgrounds}/{00..99}/{ID}.{mp4,txt}
#   $SVI_BENCH_DATA/T8/basketball/splits/{train,val,test,test_100,test_1000}.txt
#   $SVI_BENCH_DATA/T8/basketball/captions.json                                (id -> refined_instruction + player_specifications)
#   $SVI_BENCH_DATA/T8/basketball/qa_test/Q*.json                              (goal-accuracy QA bank)
#   $SVI_BENCH_DATA/T8/tracker_weights/{yolox,MixFormer}_sports_train.pth.tar
#   $SVI_BENCH_DATA/T8/llava_qa_checkpoint/                                    (fine-tuned LLaVA-Qwen QA model, ~15 GB)
#
# Tracker pretrained weights (yolox + MixFormer-ViT, ~1.2 GB per task) are
# symlinked into each task's eval/pretrained/ so the relative loaders in
# track.yaml resolve without further configuration.

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
python3 "$REPO_ROOT/scripts/extract_tars.py" --root "$DATA_ROOT/T7" --delete-after
python3 "$REPO_ROOT/scripts/extract_tars.py" --root "$DATA_ROOT/T8" --delete-after

# Tracker weights came in with the T7/T8 snapshot above. They live next
# to the eval scripts (track.yaml uses a relative `pretrained/...` path),
# so symlink them into each task's eval/pretrained/.
T7_PRETRAINED="$REPO_ROOT/svi_bench/tasks/t7_motion_conditioned_generation/eval/pretrained"
T8_PRETRAINED="$REPO_ROOT/svi_bench/tasks/t8_goal_conditioned_action_generation/eval/pretrained"
mkdir -p "$T7_PRETRAINED" "$T8_PRETRAINED"
for f in yolox_x_sports_train.pth.tar MixFormer_sports_train.pth.tar; do
    t7_src="$DATA_ROOT/T7/tracker_weights/$f"
    t8_src="$DATA_ROOT/T8/tracker_weights/$f"
    [ -f "$t7_src" ] && [ ! -e "$T7_PRETRAINED/$f" ] && ln -sf "$t7_src" "$T7_PRETRAINED/$f"
    [ -f "$t8_src" ] && [ ! -e "$T8_PRETRAINED/$f" ] && ln -sf "$t8_src" "$T8_PRETRAINED/$f"
done
echo "Tracker weights symlinked into T7 and T8 eval/pretrained/."

echo ""
echo "Done. Data is at: $DATA_ROOT"
echo "Set 'export SVI_BENCH_DATA=$DATA_ROOT' before running task train/eval scripts."
