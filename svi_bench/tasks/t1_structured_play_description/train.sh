#!/usr/bin/env bash
# T1 + T2 joint training: full fine-tune of LLaVA-Video-7B-Qwen2 on the
# combined caption + multi-choice QA pool. Same script trains the checkpoint
# used by both T1 (caption eval) and T2 (QA eval).
#
# Defaults to the joint pool (sports_100k.yaml). Override:
#   T1 only:   DATA_YAML=$HERE/configs/sports_caption_100k.yaml
#   T2 only:   DATA_YAML=$HERE/configs/sports_qa_100k.yaml
#   Per-sport: DATA_YAML=$HERE/configs/{basketball,hockey,soccer}_100k.yaml

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$HERE"
REPO_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
DATA_ROOT="${SVI_BENCH_DATA:-$REPO_ROOT/data}"

# `from llava import ...` / `from trl import ...` resolve to the vendored slices.
export PYTHONPATH="$TASK_DIR:${PYTHONPATH:-}"

IMAGE_FOLDER="${IMAGE_FOLDER:-$DATA_ROOT/T1}"
VIDEO_FOLDER="${VIDEO_FOLDER:-$DATA_ROOT/T1}"
DATA_YAML="${DATA_YAML:-$TASK_DIR/configs/sports_100k.yaml}"

VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-google/siglip-so400m-patch14-384}"
PROMPT_VERSION="${PROMPT_VERSION:-qwen_1_5}"
PREV_STAGE_CHECKPOINT="${PREV_STAGE_CHECKPOINT:-lmms-lab/LLaVA-Video-7B-Qwen2}"
RUN_NAME="${RUN_NAME:-sports_100k_f16_full_ft}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/work_dirs/$RUN_NAME}"

export WANDB_PROJECT="${WANDB_PROJECT:-svi-t1-t2}"

deepspeed --master_port "${MASTER_PORT:-30000}" \
    "$TASK_DIR/llava/train/train_mem.py" \
    --deepspeed "$TASK_DIR/configs/zero3.json" \
    --model_name_or_path "$PREV_STAGE_CHECKPOINT" \
    --version "$PROMPT_VERSION" \
    --data_path "$DATA_YAML" \
    --image_folder "$IMAGE_FOLDER" \
    --video_folder "$VIDEO_FOLDER" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower "$VISION_MODEL_VERSION" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 16 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --report_to "wandb"
