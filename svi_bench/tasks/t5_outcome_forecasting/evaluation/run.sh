#!/usr/bin/env bash
# ============================================================
# Example: Run Qwen3-VL inference on outcome forecasting data
#
# This script shows how to launch multi-GPU inference using
# torchrun. Adjust NUM_GPUS, paths, and SLURM flags to match
# your cluster setup.
#
# Usage:
#   # Direct launch (no SLURM)
#   bash evaluation/run.sh
#
#   # Via SLURM
#   sbatch --gpus=8 --job-name=t5-qwen-eval --wrap="bash evaluation/run.sh"
# ============================================================

NUM_GPUS=8

# ------- Adjust these paths -------
TEST_JSON="data/hockey_test.json"
OUTPUT="outputs/hockey_qwen.json"
VIDEO_ROOT="/path/to/video/root"           # Root directory for video files
ADAPTER="/path/to/lora/checkpoint"         # Set to "" to use base model

torchrun --nproc_per_node=$NUM_GPUS evaluation/infer_qwen.py \
    --test_json "$TEST_JSON" \
    --output "$OUTPUT" \
    --video_root "$VIDEO_ROOT" \
    --adapter "$ADAPTER"
