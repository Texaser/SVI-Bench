#!/usr/bin/env bash
# T8 — Goal-Conditioned Action Generation
# LoRA fine-tunes Wan2.1-Fun-V1.1-1.3B-Control with per-video polished
# captions and first/last-frame bbox conditioning (the "task2" variant of
# T7). Typically chained off a T7 checkpoint via --lora_checkpoint.
# Edit the data paths below to point at your local dataset before running.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED="$HERE/../_wan_shared"

# Put the vendored DiffSynth slice on PYTHONPATH so `from diffsynth import ...`
# in train.py and validate.py resolves to ${SHARED}/diffsynth.
export PYTHONPATH="$SHARED:${PYTHONPATH:-}"

export VALIDATION_SCRIPT="$HERE/validate.py"
export VALIDATION_NUM_FRAMES=81
export VALIDATION_TIME_DIVISION_FACTOR=1

accelerate launch "$SHARED/train.py" \
  --bbox_folder /mnt/bum/hanyi/repo/sports_detection/segment-anything-2-real-time/basketball_set/train_task2_final.txt \
  --video_base_path /mnt/bum/hanyi/data/basketball_fps_15_task2 \
  --background_video_folder /mnt/bum/hanyi/data/basketball_inpainting_video_task2 \
  --polished_captions \
    /mnt/bum/hanyi/repo/sports_detection/segment-anything-2-real-time/polished_captions_final.json \
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
  --output_path "./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_task2" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --lora_checkpoint "./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_task2/step-8000-run2.safetensors" \
  --extra_inputs "input_image,bbox,background_video" \
  --use_overlay_method \
  --bbox_color_mode color \
  --bbox_first_last_only \
  --time_division_factor 1 \
  --save_steps 2000
