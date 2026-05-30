#!/usr/bin/env bash
# ============================================================
# T6 Evaluation — SLURM Submission
#
# Submits evaluation jobs using pre-computed model outputs.
# Adjust SPORT, MODEL, GPU config, and node as needed.
#
# GPU configurations:
#   A6000 (48GB): 8 GPUs, --tensor_parallel 4 --pipeline_parallel 2
#   H100  (80GB): 4 GPUs, --tensor_parallel 4 --pipeline_parallel 1
#
# Usage:
#   bash evaluation/run_slurm.sh
# ============================================================

# ------- Adjust these -------
SPORT="basketball"
MODEL="gpt"                              # Options: gpt, qwen, gemini
NODE="mirage.ib"

# GPU config — pick one:
# A6000 (8 GPUs)
NUM_GPUS=8
TP=4
PP=2
# H100 (4 GPUs)
# NUM_GPUS=4
# TP=4
# PP=1

PREDICTIONS="model_output/${SPORT}_${MODEL}_zero_shot.json"
DATA_DIR="dataset/${SPORT}"
OUTPUT_DIR="outputs/${SPORT}_${MODEL}"

mkdir -p "$OUTPUT_DIR"

# Factual Accuracy
sbatch --gpus=$NUM_GPUS --nodelist="$NODE" -J t6_factual_${SPORT}_${MODEL} --wrap "
python evaluation/eval_factual.py \
    --sport $SPORT \
    --data_dir $DATA_DIR \
    --predictions $PREDICTIONS \
    --output $OUTPUT_DIR/factual_eval.json \
    --tensor_parallel $TP \
    --pipeline_parallel $PP
"

# Coverage
# sbatch --gpus=$NUM_GPUS --nodelist="$NODE" -J t6_coverage_${SPORT}_${MODEL} --wrap "
# python evaluation/eval_coverage.py \
#     --sport $SPORT \
#     --data_dir $DATA_DIR \
#     --predictions $PREDICTIONS \
#     --output $OUTPUT_DIR/coverage_eval.json \
#     --tensor_parallel $TP \
#     --pipeline_parallel $PP
# "

# Writing Quality
# sbatch --gpus=$NUM_GPUS --nodelist="$NODE" -J t6_writing_${SPORT}_${MODEL} --wrap "
# python evaluation/eval_writing.py \
#     --data_dir $DATA_DIR \
#     --predictions $PREDICTIONS \
#     --output $OUTPUT_DIR/writing_eval.json \
#     --tensor_parallel $TP \
#     --pipeline_parallel $PP
# "
