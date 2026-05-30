#!/usr/bin/env bash
# ============================================================
# T6 Long-Form Narrative Synthesis — Evaluation Examples
#
# Runs the three evaluation metrics (factual accuracy, coverage,
# writing quality) on pre-computed model outputs.
#
# Prerequisites:
#   - vLLM installed with Qwen3-235B-A22B-Thinking-2507-FP8 model
#   - Pre-computed predictions in model_output/ (shipped with repo)
#
# GPU configurations:
#   A6000 (48GB): 8 GPUs, --tensor_parallel 4 --pipeline_parallel 2
#   H100  (80GB): 4 GPUs, --tensor_parallel 4 --pipeline_parallel 1
#
# Usage:
#   bash evaluation/run.sh
# ============================================================

# ------- Adjust these -------
SPORT="basketball"
MODEL="gpt"                              # Options: gpt, qwen, gemini

# GPU config — pick one:
# A6000 (8 GPUs)
TP=4
PP=2
# H100 (4 GPUs)
# TP=4
# PP=1

PREDICTIONS="model_output/${SPORT}_${MODEL}_zero_shot.json"
DATA_DIR="dataset/${SPORT}"
OUTPUT_DIR="outputs/${SPORT}_${MODEL}"

mkdir -p "$OUTPUT_DIR"

# ============================================================
# Factual Accuracy
# ============================================================
python evaluation/eval_factual.py \
    --sport "$SPORT" \
    --data_dir "$DATA_DIR" \
    --predictions "$PREDICTIONS" \
    --output "$OUTPUT_DIR/factual_eval.json" \
    --tensor_parallel $TP \
    --pipeline_parallel $PP

# ============================================================
# Saliency / Coverage
# ============================================================
# python evaluation/eval_coverage.py \
#     --sport "$SPORT" \
#     --data_dir "$DATA_DIR" \
#     --predictions "$PREDICTIONS" \
#     --output "$OUTPUT_DIR/coverage_eval.json" \
#     --tensor_parallel $TP \
#     --pipeline_parallel $PP

# ============================================================
# Writing Quality
# ============================================================
# python evaluation/eval_writing.py \
#     --data_dir "$DATA_DIR" \
#     --predictions "$PREDICTIONS" \
#     --output "$OUTPUT_DIR/writing_eval.json" \
#     --tensor_parallel $TP \
#     --pipeline_parallel $PP
