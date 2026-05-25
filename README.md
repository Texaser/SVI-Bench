# SVI-Bench

**Project page:** https://svi-bench.github.io/  ·  **Dataset:**
[MVP-Group/SVI-Bench](https://huggingface.co/datasets/MVP-Group/SVI-Bench)

SVI-Bench is a multi-task benchmark for evaluating vision-language models on
sports video understanding. It spans four pillars — Perception, Reasoning,
Simulation, and Agency — across nine tasks, three sports (basketball, hockey,
soccer), and multiple seasons.

## Tasks

| Task | Pillar | Description |
|------|--------|-------------|
| T1 | Perception | Structured play-by-play description |
| T2 | Perception | Fine-grained action QA |
| T3 | Perception | Compositional video retrieval (R@K ranking) |
| T4 | Reasoning | Strategic reasoning QA (free-text, LLM-as-judge) |
| T5 | Reasoning | Outcome forecasting (multiple-choice) |
| T6 | Reasoning | Long-form narrative synthesis (report generation) |
| T7 | Simulation | Motion-conditioned video generation (LoRA fine-tune) |
| T8 | Simulation | Goal-conditioned action generation (LoRA fine-tune) |
| T9 | Agency | Cross-corpus agentic reasoning (search + QA) |

Each task has its own directory under `svi_bench/tasks/` with a dedicated
`README.md` covering setup, data format, and evaluation instructions.

## Quickstart (T7 generation demo)

```bash
git clone https://github.com/Texaser/SVI-Bench && cd SVI-Bench
pip install "svi-bench[t7]"
bash scripts/download_t7_t8.sh                      # ~50 GB; T7+T8 data + tracker weights
bash svi_bench/tasks/t7_motion_conditioned_generation/download_checkpoint.sh basketball

# Inference (test_100 split, 8 GPUs)
SPORT=basketball bash svi_bench/tasks/t7_motion_conditioned_generation/inference/infer.sh \
    ./checkpoints/T7/basketball

# Eval (Video mIoU + feature similarity)
STEP_DIR=./checkpoints/T7/basketball/validation/step-<N>
VALIDATION_DIR=$STEP_DIR bash svi_bench/tasks/t7_motion_conditioned_generation/eval/run_basketball.sh
bash svi_bench/tasks/t7_motion_conditioned_generation/eval/run_basketball_featsim.sh $STEP_DIR
```

## Quickstart (T8 generation demo)

```bash
git clone https://github.com/Texaser/SVI-Bench && cd SVI-Bench
pip install "svi-bench[t8]"
bash scripts/download_t7_t8.sh                      # ~65 GB; T7+T8 data, tracker weights, LLaVA-Qwen checkpoint
bash svi_bench/tasks/t8_goal_conditioned_action_generation/download_checkpoint.sh

# Inference (test_1000 split, 8 GPUs)
bash svi_bench/tasks/t8_goal_conditioned_action_generation/inference/infer.sh \
    ./checkpoints/T8/basketball

# Eval (last-frame mIoU + feature similarity + goal accuracy)
VIDEO_DIR=./checkpoints/T8/basketball/validation/step-<N>
bash svi_bench/tasks/t8_goal_conditioned_action_generation/eval/run_basketball.sh         $VIDEO_DIR
bash svi_bench/tasks/t8_goal_conditioned_action_generation/eval/run_basketball_featsim.sh $VIDEO_DIR
bash svi_bench/tasks/t8_goal_conditioned_action_generation/eval/run_basketball_goalacc.sh $VIDEO_DIR
```

For other tasks, see the per-task README.

## Data

Datasets are hosted on HuggingFace at
[MVP-Group/SVI-Bench](https://huggingface.co/datasets/MVP-Group/SVI-Bench).
Access is gated — agree to the terms on the dataset page once, then your HF
token unlocks all data. Large data are shipped as `.tar` bundles. See each
task's README for download and setup instructions.

## Repository layout

```
SVI-Bench/
├── pyproject.toml
├── svi_bench/
│   ├── core/                 # shared utilities
│   └── tasks/                # one subpackage per task
│       ├── t1_structured_play_description/
│       ├── t2_fine_grained_action_qa/
│       ├── t3_compositional_video_retrieval/
│       ├── t4_strategic_reasoning_qa/
│       ├── t5_outcome_forecasting/
│       ├── t6_long_form_narrative_synthesis/
│       ├── t7_motion_conditioned_generation/
│       ├── t8_goal_conditioned_action_generation/
│       └── t9_cross_corpus_agentic_reasoning/
├── configs/                  # YAML configs per task
└── scripts/                  # helper scripts (download, extract_tars, etc.)
```

## License

Code is MIT (see [`LICENSE`](LICENSE)). Data is governed by the gated-access
agreement on the HF dataset page.
