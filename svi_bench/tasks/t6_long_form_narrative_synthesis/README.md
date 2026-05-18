# T6 — Long-Form Narrative Synthesis

**Pillar 2: Causal Reasoning** &nbsp;|&nbsp; Full-game video &nbsp;|&nbsp; Report generation &nbsp;|&nbsp; Factual accuracy / Saliency / Writing quality

## Task Overview

Given a full-game video (or multiple game videos) and a topic-specific writing prompt, the model must generate a detailed game report that accurately captures what happened, highlights the most important developments, and adheres to the specified writing constraints. Each prompt specifies a target word count, a narrative perspective (e.g., analyst, beat reporter), and formatting constraints.

Report prompts are organized into two settings and five topic categories:

- **Single-game reports** require the model to watch one full game and produce a report on a specified topic: an overall game summary, a targeted player performance analysis, a team strategy breakdown, a comparison of a player's performance across periods, or a narrative focused on the game's decisive moments (e.g., final minutes, key goals).
- **Multi-game reports** require synthesizing 2–10 games linked by a coherent theme — such as a recurring team, a head-to-head matchup, or a player's trajectory across consecutive games — into a single analytical piece. Topics include multi-game team summaries, player performance trends, matchup comparisons, and home-vs-away analyses.

### Report Templates

Each sport uses 10 templates (5 single-game, 5 multi-game). Eight templates are shared across all sports; two are sport-specific:

| # | Template | Scope |
|---|---|---|
| 1 | Game Summarization | Single-game |
| 2 | Player Performance Summarization | Single-game |
| 3 | Team Strategy Analysis | Single-game |
| 4 | Player Performance Period Comparison | Single-game |
| 5 | Game Final Moments (basketball) / Goal Development Analysis (hockey & soccer) | Single-game |
| 6 | Team Multi-Game Summarization | Multi-game |
| 7 | Player Multi-Game Performance Summarization | Multi-game |
| 8 | Two-Team Matchup Comparison | Multi-game |
| 9 | Player Performance Comparison | Multi-game |
| 10 | Player Face-to-Face Matchup (basketball) / Team Home-Away Performance (hockey & soccer) | Multi-game |

## Dataset

| Sport | Train | Test | Unique Videos | Video Hours | Avg. Video Length | Avg. Report Words |
|---|---|---|---|---|---|---|
| Basketball | 8,447 | 350 | 5,224 | 8,968 h | 103 min | 471 |
| Hockey | 7,780 | 350 | 1,863 | 3,326 h | 108 min | 467 |
| Soccer | 1,560 | 300 | 249 | 414 h | 100 min | 495 |

### Data Format

**Train/test lists** (`dataset/<sport>/train_list.json` and `test_list.json`) map question type keys to lists of integer sample IDs:

```json
{
  "single_Q1": [1, 5, 12, ...],
  "single_Q2": [...],
  "multi_Q1": [...],
  ...
}
```

`single_Q1`–`single_Q5` correspond to the 5 single-game templates; `multi_Q1`–`multi_Q5` to the 5 multi-game templates.

**Component folders** — each sample ID maps to a folder under `dataset/<sport>/single_game/Q<n>/<id>/` or `multi_game/Q<n>/<id>/`:

| File | Description |
|---|---|
| `prompt.txt` | Writing task prompt (topic, perspective, word count, formatting constraints) |
| `metadata.json` | Game ID(s), template source, and optional placeholder values |
| `stats.txt` | Period-by-period player/team box score statistics |
| `log.txt` | Chronological play-by-play event log with timestamps |
| `events.txt` | Structured event annotations (where available) |
| `ground_truth_report.txt` | GPT-5-generated ground-truth report following the prompt |
| `coverage_facts.txt` | Atomic factual claims extracted from the ground-truth report (for saliency evaluation) |

**Note:** `coverage_facts.txt` and `events.txt` are not present in every sample folder. The evaluation scripts handle missing files gracefully.

## Evaluation

We evaluate generated reports along three dimensions using an LLM-as-a-judge framework with **Qwen3-235B-A22B-Thinking** as the judge:

| Dimension | Metric | Script | Description |
|---|---|---|---|
| **Factual accuracy** | Supported / Total (%) | `evaluation/eval_factual.py` | Decompose the report into atomic verifiable facts and verify each against ground-truth resources (game report, stats, play-by-play logs). Score = proportion of *Supported* facts. |
| **Saliency (coverage)** | Covered / Total (%) | `evaluation/eval_coverage.py` | Decompose the ground-truth report into atomic statements and measure what proportion is covered by the generated report. Captures whether the model identifies and prioritizes the most important game developments. |
| **Writing quality** | 1–5 rubric score | `evaluation/eval_writing.py` | Rubric-based scoring for stylistic adherence, narrative coherence, and overall quality. |

### Running Evaluation

All three eval scripts take `--sport`, `--data_dir`, `--predictions` (aggregated JSON), and `--output`:

```bash
# Factual accuracy (sport-specific prompts for stat terminology)
python evaluation/eval_factual.py \
    --sport basketball \
    --data_dir dataset/basketball \
    --predictions outputs/basketball_qwen_aggregated.json \
    --output outputs/basketball_factual_eval.json

# Saliency / coverage (sport-specific prompts)
python evaluation/eval_coverage.py \
    --sport basketball \
    --data_dir dataset/basketball \
    --predictions outputs/basketball_qwen_aggregated.json \
    --output outputs/basketball_coverage_eval.json

# Writing quality (sport-agnostic prompt)
python evaluation/eval_writing.py \
    --data_dir dataset/basketball \
    --predictions outputs/basketball_qwen_aggregated.json \
    --output outputs/basketball_writing_eval.json
```

The predictions JSON should have the structure `{"Q1": {"<sample_id>": "<report_text>", ...}, "multi_Q1": {...}, ...}`. All inference scripts produce this format under the `"predictions"` key.

**Requirements:** `vllm`, `transformers` (for the Qwen3-235B thinking model judge)

## Inference

Three inference scripts are provided in `inference/`, one per model family. Each handles all three sports and both single/multi-game settings.

### Qwen3-VL (`inference/infer_qwen.py`)

Loads video directly using decord. Supports LoRA adapters and multi-GPU via torchrun.

```bash
python inference/infer_qwen.py \
    --test_list dataset/basketball/test_list.json \
    --data_dir dataset/basketball \
    --video_dir /path/to/full_game_videos \
    --output outputs/basketball_qwen.json \
    --sample_fps 1.0

# Multi-GPU
torchrun --nproc_per_node=4 inference/infer_qwen.py \
    --test_list dataset/hockey/test_list.json \
    --data_dir dataset/hockey \
    --video_dir /path/to/hockey_videos \
    --output outputs/hockey_qwen.json \
    --adapter /path/to/lora/checkpoint
```

**Dependencies:** `torch`, `transformers`, `peft`, `decord`, `numpy`, `tqdm`, `Pillow`

### GPT (`inference/infer_gpt.py`)

Uses pre-extracted video frames (JPEG) via the OpenAI Responses API. Resumable.

```bash
export OPENAI_API_KEY="sk-..."

python inference/infer_gpt.py \
    --test_list dataset/basketball/test_list.json \
    --data_dir dataset/basketball \
    --frames_dir /path/to/full_game_video_frames \
    --output outputs/basketball_gpt.json \
    --model gpt-4o --max_frames 500
```

**Dependencies:** `openai`, `numpy`, `tqdm`

### Gemini (`inference/infer_gemini.py`)

Compresses full game videos via ffmpeg and uploads to Gemini Files API. Resumable.

```bash
export GEMINI_API_KEY="AIza..."

python inference/infer_gemini.py \
    --test_list dataset/soccer/test_list.json \
    --data_dir dataset/soccer \
    --video_dir /path/to/soccer_videos \
    --video_cache_dir /path/to/compressed_cache \
    --output outputs/soccer_gemini.json \
    --model gemini-2.5-flash-preview --total_duration 3600
```

**Dependencies:** `google-genai`, `tqdm`, `ffmpeg` (system)

## Directory Structure

```
t6_long_form_narrative_synthesis/
├── __init__.py
├── README.md
├── dataset/
│   ├── basketball/
│   │   ├── train_list.json
│   │   ├── test_list.json
│   │   ├── single_game/Q1/<id>/{prompt.txt, metadata.json, stats.txt, ...}
│   │   └── multi_game/Q1/<id>/{...}
│   ├── hockey/
│   │   └── (same structure)
│   └── soccer/
│       └── (same structure)
├── evaluation/
│   ├── eval_factual.py
│   ├── eval_coverage.py
│   └── eval_writing.py
├── inference/
│   ├── infer_qwen.py
│   ├── infer_gpt.py
│   └── infer_gemini.py
├── logs/
└── outputs/
```

## Notes

- Data config on HF: `t6_long_form_narrative_synthesis`
- Default eval config: [`configs/t6.yaml`](../../../configs/t6.yaml)
