#!/usr/bin/env bash
# ============================================================
# Example: Run T4 Strategic Reasoning QA Evaluation
#
# Adjust the paths and model choice below to match your setup.
#
# Usage:
#   bash evaluation/run.sh
#
#   # Via SLURM
#   sbatch --gpus=1 --job-name=t4-eval --wrap="bash evaluation/run.sh"
# ============================================================

# ------- Adjust these -------
VIDEO_ROOT="/path/to/your/downloaded/data"   # Root directory for video files
MODEL="qwen"                                 # Options: qwen, molmo, gpt, gemini
MODEL_KEY=""                                 # API key for GPT/Gemini (leave empty for local models)
MAX_ANSWERS=5
# SUBSET="--subset"                          # Uncomment to use qa_subset.json

export OPENROUTER_API_KEY="your_openrouter_key_here"

python evaluation/evaluate.py \
    --video_root "$VIDEO_ROOT" \
    --model "$MODEL" \
    --model_key "$MODEL_KEY" \
    --max_answers "$MAX_ANSWERS" \
    ${SUBSET:-}
