#!/usr/bin/env bash
# ============================================================
# T5 Outcome Forecasting — Inference Examples
#
# Adjust paths, GPU counts, and API keys to match your setup.
# Each section can be run independently.
#
# Usage:
#   bash evaluation/run.sh
#
#   # Via SLURM (example)
#   sbatch --gpus=8 --job-name=t5-eval --wrap="bash evaluation/run.sh"
# ============================================================

# ------- Common settings -------
NUM_GPUS=8
DATA_DIR="data/T5"
OUTPUT_DIR="outputs"
VIDEO_ROOT="/path/to/video/root"

mkdir -p "$OUTPUT_DIR"

# ============================================================
# Qwen3-VL (with LoRA adapter)
# ============================================================
torchrun --nproc_per_node=$NUM_GPUS evaluation/infer_qwen.py \
    --test_json "$DATA_DIR/basketball_test.json" \
    --output "$OUTPUT_DIR/basketball_qwen.json" \
    --video_root "$VIDEO_ROOT" \
    --sample_fps 0.2 \
    --adapter /path/to/lora/checkpoint

# ============================================================
# Molmo2
# ============================================================
# torchrun --nproc_per_node=$NUM_GPUS evaluation/infer_molmo.py \
#     --test_json "$DATA_DIR/basketball_test.json" \
#     --output "$OUTPUT_DIR/basketball_molmo.json" \
#     --sample_fps 0.2

# ============================================================
# GPT
# ============================================================
# export OPENAI_API_KEY="sk-..."
# python evaluation/infer_gpt.py \
#     --test_json "$DATA_DIR/basketball_test.json" \
#     --output "$OUTPUT_DIR/basketball_gpt.json" \
#     --model gpt-4o \
#     --frame_fps 0.5 \
#     --image_detail low

# ============================================================
# Gemini
# ============================================================
# export GEMINI_API_KEY="AIza..."
# python evaluation/infer_gemini.py \
#     --test_json "$DATA_DIR/soccer_test.json" \
#     --output "$OUTPUT_DIR/soccer_gemini.json" \
#     --model gemini-2.5-flash-preview

# ============================================================
# Calibration Error (after Qwen or Molmo inference)
# ============================================================
# python evaluation/calc_ce.py --results "$OUTPUT_DIR/basketball_qwen.json"
# python evaluation/calc_ce.py --results "$OUTPUT_DIR/basketball_qwen.json" --num_bins 5
