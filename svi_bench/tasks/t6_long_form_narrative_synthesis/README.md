# T6 — Long-Form Narrative Synthesis

Given a full-game video (or multiple game videos) and a topic-specific writing prompt, generate a detailed game report that accurately captures what happened, highlights key developments, and follows the specified writing constraints (word count, perspective, style). Metrics: Factual accuracy, Saliency/coverage, Writing quality (LLM-as-a-judge).

Reports span two settings: **single-game** (one full game) and **multi-game** (2–10 games linked by theme), each with 5 topic templates (game summary, player analysis, team strategy, period comparison, and a sport-specific template).

## 1. Install

```bash
conda env create -f environment.yaml && conda activate svi_t6
```

Or follow the official guides for the models you plan to use:
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) &nbsp;|&nbsp; [Molmo2](https://github.com/allenai/molmo)

Evaluation requires [vLLM](https://github.com/vllm-project/vllm) for the Qwen3-235B-A22B-Thinking judge.

## 2. Data

```bash
huggingface-cli download MVP-Group/SVI-Bench --repo-type dataset \
    --include "T6/**" --local-dir data/
```

Everything goes under `data/T6/`:

```
data/T6/
├── {basketball,hockey,soccer}/
│   ├── train_list.json
│   ├── test_list.json
│   ├── single_game/Q{1-5}/<id>/{prompt.txt, metadata.json, stats.txt, log.txt, ...}
│   └── multi_game/Q{1-5}/<id>/{...}
```

Each sample folder contains:

| File | Description |
|---|---|
| `prompt.txt` | Writing task prompt (topic, perspective, word count, formatting) |
| `metadata.json` | Game ID(s), video path(s), template info |
| `stats.txt` | Period-by-period box score statistics |
| `log.txt` | Chronological play-by-play event log |
| `ground_truth_report.txt` | Reference report |
| `coverage_facts.txt` | Atomic factual claims for saliency evaluation |

The `video_path` / `video_paths` fields in `metadata.json` are relative paths. Use `--video_dir` at inference time to point to your local video root.

## 3. Evaluate

Three evaluation dimensions using **Qwen3-235B-A22B-Thinking-2507-FP8** (via vLLM) as judge:

| Dimension | Metric | Script |
|---|---|---|
| Factual accuracy | Supported / (Supported + Contradicted) (%) | `evaluation/eval_factual.py` |
| Saliency (coverage) | Covered / Total ground-truth facts (%) | `evaluation/eval_coverage.py` |
| Writing quality | 1–5 rubric score | `evaluation/eval_writing.py` |

**GPU requirements:**

| GPU | Count | `--tensor_parallel` | `--pipeline_parallel` |
|---|---|---|---|
| A6000 (48GB) | 8 | 4 | 2 |
| H100 (80GB) | 4 | 4 | 1 |

```bash
# Factual accuracy (A6000 example)
python evaluation/eval_factual.py \
    --sport basketball \
    --data_dir dataset/basketball \
    --predictions outputs/basketball_qwen.json \
    --output outputs/basketball_factual_eval.json \
    --tensor_parallel 4 --pipeline_parallel 2

# Saliency / coverage
python evaluation/eval_coverage.py \
    --sport basketball \
    --data_dir dataset/basketball \
    --predictions outputs/basketball_qwen.json \
    --output outputs/basketball_coverage_eval.json \
    --tensor_parallel 4 --pipeline_parallel 2

# Writing quality
python evaluation/eval_writing.py \
    --data_dir dataset/basketball \
    --predictions outputs/basketball_qwen.json \
    --output outputs/basketball_writing_eval.json \
    --tensor_parallel 4 --pipeline_parallel 2
```

For H100, use `--pipeline_parallel 1` instead (no pipeline parallelism needed).

The predictions JSON should have the structure `{"predictions": {"Q1": {"<id>": "<report>", ...}, "multi_Q1": {...}, ...}}`.

Pre-computed baseline predictions are provided in `model_output/` for GPT, Qwen, and Gemini across all three sports. Example SLURM and local launch scripts are in `evaluation/run_slurm.sh` and `evaluation/run.sh`.

## 4. Inference

Three inference scripts in `inference/`, one per model family.

**Qwen3-VL** — supports LoRA adapter, multi-GPU via `torchrun`:

```bash
torchrun --nproc_per_node=8 inference/infer_qwen.py \
    --test_list dataset/basketball/test_list.json \
    --data_dir dataset/basketball \
    --video_dir /path/to/video/root \
    --output outputs/basketball_qwen.json \
    --sample_fps 1.0 \
    --adapter /path/to/lora/checkpoint
```

**GPT** — uses pre-extracted video frames (JPEG) via OpenAI Responses API:

```bash
export OPENAI_API_KEY="sk-..."
python inference/infer_gpt.py \
    --test_list dataset/basketball/test_list.json \
    --data_dir dataset/basketball \
    --frames_dir /path/to/extracted_frames \
    --output outputs/basketball_gpt.json \
    --model gpt-4o --max_frames 500
```

Pre-extract frames using: `python utils/extract_frames_uniform.py`

**Gemini** — compresses videos via ffmpeg and uploads to Files API:

```bash
export GEMINI_API_KEY="AIza..."
python inference/infer_gemini.py \
    --test_list dataset/soccer/test_list.json \
    --data_dir dataset/soccer \
    --video_dir /path/to/video/root \
    --video_cache_dir /tmp/compressed_cache \
    --output outputs/soccer_gemini.json \
    --model gemini-2.5-flash-preview --total_duration 3600
```

Compress videos ahead of time using: `python utils/condense_video_to_1h.py`

## Notes

- Data config on HF: `t6_long_form_narrative_synthesis`
- Default eval config: [`configs/t6.yaml`](../../../configs/t6.yaml)
