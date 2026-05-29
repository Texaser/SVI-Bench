#!/usr/bin/env bash
# ============================================================
# Qwen3-VL LoRA Fine-tuning for Outcome Forecasting (T5)
#
# Uses ms-swift (https://github.com/modelscope/ms-swift) for
# LoRA fine-tuning of Qwen3-VL on the forecasting training set.
#
# Prerequisites:
#   pip install ms-swift[llm]
#
# Usage:
#   # 1. Convert training JSON to Swift JSONL format:
#   python training/convert_train_to_jsonl.py \
#       --input data/basketball_train.json data/hockey_train.json data/soccer_train.json \
#       --output data/train.jsonl
#
#   # 2. Run training (adjust GPU count and paths as needed):
#   bash training/train_qwen.sh
#
#   # Or submit via SLURM:
#   sbatch --gpus=8 --job-name=t5-qwen-lora --wrap="bash training/train_qwen.sh"
# ============================================================
set -euo pipefail

# -------- GPU configuration ----------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

# -------- Qwen3-VL video settings ----------
export VIDEO_MAX_TOKEN_NUM=768   # Max visual tokens per video
export FPS=0.2                   # Frame sampling rate

# -------- Performance tuning ----------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=1
export DECORD_REWIND_RETRY_MAX=1
export DECORD_VIDEO_NON_KEY_FRAME_RETRY_MAX=1

# -------- Paths ----------
MODEL="Qwen/Qwen3-VL-8B-Instruct"
TRAIN_JSONL="data/train.jsonl"
OUT_DIR="output/qwen3-vl-8b-lora-t5-forecasting"

# -------- Training ----------
swift sft \
  --model "${MODEL}" \
  --dataset "${TRAIN_JSONL}" \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --max_length 50000 \
  --save_steps 200 \
  --logging_steps 10 \
  --output_dir "${OUT_DIR}" \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --deepspeed zero3 \
  --use_logits_to_keep true \
