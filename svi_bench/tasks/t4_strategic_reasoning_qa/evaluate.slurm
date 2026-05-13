#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4

export OPENROUTER_API_KEY=

python -m evaluation.evaluate
