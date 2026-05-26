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
Per-task quickstarts (clone → install → download → infer → eval one-shot
recipes) live in those READMEs.

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
├── README.md
├── LICENSE
├── scripts/                  # benchmark-wide helpers (e.g. extract_tars.py)
└── svi_bench/
    ├── core/                 # shared utilities (config loader, model registry)
    └── tasks/                # one self-contained subpackage per task
        ├── t1_structured_play_description/
        ├── t2_fine_grained_action_qa/
        ├── t3_compositional_video_retrieval/
        ├── t4_strategic_reasoning_qa/
        ├── t5_outcome_forecasting/
        ├── t6_long_form_narrative_synthesis/
        ├── t7_motion_conditioned_generation/
        ├── t8_goal_conditioned_action_generation/
        └── t9_cross_corpus_agentic_reasoning/
```

Each task dir holds its own `train.sh` / `inference/` / `eval/` / `scripts/`
/ `configs/` / `figures/` as needed.

## License

Code is MIT (see [`LICENSE`](LICENSE)). Data is governed by the gated-access
agreement on the HF dataset page.
