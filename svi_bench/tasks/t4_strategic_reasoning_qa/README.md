# T4 — Strategic Reasoning QA

Given a full-game video (~55–150 min) and an open-ended question, produce a free-text response explaining strategic reasoning behind game events. Unlike short-clip tasks, T4 requires reasoning over extended game portions: identifying strategic errors, evaluating tactical execution, and interpreting latent dynamics such as momentum shifts. Metric: LLM-as-a-judge score (0–5).

## 1. Install

Follow the official installation guides for the models you plan to use:
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) &nbsp;|&nbsp; [Molmo2](https://github.com/allenai/molmo)

API-based models (GPT, Gemini) require their respective API keys.

## 2. Data

```bash
huggingface-cli download MVP-Group/SVI-Bench --repo-type dataset \
    --include "T4/**" --local-dir data/
```

Everything goes under `data/T4/`:

```
data/T4/
├── qa.json            # 1,000 questions (full evaluation set)
└── qa_subset.json     # 300 questions (smaller evaluation set)
```

Each entry:

```json
{
  "league": "EuroLeague",
  "game_id": "191869",
  "question_type": "anomaly_novelty_detection",
  "question": "In the closing seconds of the fourth period, what happened on ...",
  "answer": "With about 0.4 seconds left, Fenerbahce converted a last-gasp ...",
  "video_path": "T4/basketball/videos/191869.mp4"
}
```

The `video_path` field is a relative path. Use `--video_root` at evaluation time to prepend your local root.

**Sports coverage:** Basketball (398 Qs), Hockey (333 Qs), Soccer (269 Qs).

## 3. Evaluate

Two components: (1) a model generates up to *k* candidate answers per question, then (2) an LLM judge (DeepSeek-V3 via OpenRouter) scores each candidate on a 0–5 scale. The best score among all candidates is reported.

```bash
export OPENROUTER_API_KEY="your_key_here"

python -m evaluation.evaluate \
    --video_root /path/to/video/root \
    --model qwen
```

Available models: `qwen`, `molmo`, `gpt`, `gemini`. For API-based models, pass `--model_key`:

```bash
python -m evaluation.evaluate \
    --video_root /path/to/video/root \
    --model gpt \
    --model_key "sk-..."
```

Additional flags: `--max_answers 5` (default), `--subset` (use `qa_subset.json`).

An example SLURM launch script is provided in `evaluation/run.sh`.

**Output files** (written to `outputs/<model>/k-<k>/`):
- `results.json` — full QA data with model responses and scores
- `llm_judge_trace.json` — detailed judge reasoning for each candidate
- `results.txt` — per-sport and overall average scores

## Notes

- No train split — T4 is evaluation-only
- Data config on HF: `t4_strategic_reasoning_qa`
- Default eval config: [`configs/t4.yaml`](../../../configs/t4.yaml)
