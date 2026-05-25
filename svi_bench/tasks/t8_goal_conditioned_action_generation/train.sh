#!/usr/bin/env bash
# T8 — Goal-Conditioned Action Generation
# LoRA fine-tunes Wan2.1-Fun-V1.1-1.3B-Control with per-video polished
# captions and first/last-frame bbox conditioning. Typically chained off a
# T7 checkpoint via --lora_checkpoint. Data is pulled from HuggingFace
# (MVP-Group/SVI-Bench) via `scripts/download_t7_t8.sh` and lives under
# $SVI_BENCH_DATA (default: ./data at the repo root).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
DATA_ROOT="${SVI_BENCH_DATA:-$REPO_ROOT/data}"

# Put the bundled DiffSynth slice on PYTHONPATH so `from diffsynth import ...`
# in train.py and validate.py resolves to ${HERE}/diffsynth.
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

SPORT_DIR="$DATA_ROOT/T8/basketball"

# Convert ID-only train split -> full bbox paths
SPLIT_IDS="$SPORT_DIR/splits/train.txt"
SPLIT_BBOX_LIST="$SPORT_DIR/splits/train.bbox_paths.txt"
if [ ! -f "$SPLIT_BBOX_LIST" ]; then
  python3 "$REPO_ROOT/scripts/build_split_bbox_list.py" \
    --ids "$SPLIT_IDS" \
    --root "$SPORT_DIR/bboxes" \
    --out "$SPLIT_BBOX_LIST"
fi

export VALIDATION_SCRIPT="$HERE/validate.py"
export VALIDATION_NUM_FRAMES=81
export VALIDATION_TIME_DIVISION_FACTOR=1

accelerate launch "$HERE/train.py" \
  --bbox_folder "$SPLIT_BBOX_LIST" \
  --video_base_path "$SPORT_DIR/clips" \
  --background_video_folder "$SPORT_DIR/backgrounds" \
  --captions "$SPORT_DIR/captions.json" \
  --bbox_channels 16 \
  --video_extension .mp4 \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_t8" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,bbox,background_video" \
  --use_overlay_method \
  --bbox_color_mode color \
  --bbox_first_last_only \
  --time_division_factor 1 \
  --save_steps 2000
