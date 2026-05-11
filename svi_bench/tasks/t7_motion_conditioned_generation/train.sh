#!/usr/bin/env bash
# T7 — Motion-Conditioned Generation
# LoRA fine-tunes Wan2.1-Fun-V1.1-1.3B-Control on basketball video with
# bbox + background-video conditioning. Edit the data paths below to point
# at your local dataset before running.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Put the bundled DiffSynth slice on PYTHONPATH so `from diffsynth import ...`
# in train.py and validate.py resolves to ${HERE}/diffsynth.
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

export VALIDATION_SCRIPT="$HERE/validate.py"
export VALIDATION_NUM_FRAMES=81
export VALIDATION_TIME_DIVISION_FACTOR=1

accelerate launch "$HERE/train.py" \
  --bbox_folder /mnt/bum/hanyi/repo/sports_detection/segment-anything-2-real-time/basketball_set/train.txt \
  --video_base_path /mnt/bum/hanyi/data/basketball_fps_15 \
  --background_video_folder /mnt/bum/hanyi/data/basketball_inpainting_video \
  --bbox_channels 16 \
  --video_extension .mp4 \
  --prompt "a realistic basketball game video" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 3 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_full_scale" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,bbox,background_video" \
  --use_overlay_method \
  --bbox_color_mode color \
  --time_division_factor 1 \
  --save_steps 2000
