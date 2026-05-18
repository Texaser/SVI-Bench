# T4 — Strategic Reasoning QA

**Pillar 2: Causal Reasoning** &nbsp;|&nbsp; Full-game video &nbsp;|&nbsp; Open-ended QA &nbsp;|&nbsp; LLM-as-a-judge (0–5)

## Task Overview

Given a full-game video (~55–150 min) and an open-ended question, the model must produce a free-text response explaining strategic reasoning behind game events. Unlike short-clip perception tasks (T1–T3), T4 requires reasoning over extended game portions: identifying strategic errors, evaluating tactical execution, and interpreting latent dynamics such as momentum shifts. Evidence for a single answer may be spread across minutes of footage interleaved with irrelevant events.

## Question Categories

The dataset defines six question types:

| Category | Description | Count |
|---|---|---|
| Tactical & strategic analysis | Reasoning about tactics and strategies employed by teams and players, how decisions interact, and how they affect game state | 134 |
| Player role & skill assessment | Identifying player archetypes and evaluating execution quality of specific roles and skills | 119 |
| Causal & counterfactual reasoning | Understanding causal links between events and outcomes, and reasoning about alternative scenarios | 144 |
| Anomaly & novelty detection | Identifying what makes certain events, tactics, or outcomes unusual and explaining those novelties | 229 |
| Spatiotemporal & relational reasoning | Reasoning about spatial structures, player positions, and how these relate to specific outcomes or strategies | 203 |
| General | Questions requiring a combination of the above capabilities | 171 |

## Dataset

| Split | Questions | Description |
|---|---|---|
| Full (`dataset/qa.json`) | 1,000 | Complete evaluation set |
| Subset (`dataset/qa_subset.json`) | 300 | Smaller evaluation set |

Each question contains: `league`, `game_id`, `question_type`, `question`, and `answer`.

**Sports and league coverage:**

| Sport | Leagues | Questions |
|---|---|---|
| Basketball | NBA (133), NCAA (130), EuroLeague (135) | 398 |
| Hockey | NHL | 333 |
| Soccer | Premier League (169), La Liga (100) | 269 |

## Data Construction Pipeline

T4 demands the most rigorous quality control of any task in the benchmark because strategic reasoning questions are uniquely susceptible to language-prior shortcuts — a question that *sounds* like it requires game understanding may be answerable from general sports knowledge alone. The pipeline retains only ~2.4% of initial candidates through five stages:

1. **Initial construction** (`generation/generate.py`): GPT-5.2 generates candidate QA pairs from professional commentary and game reports, grounded against team rosters. Six category-specific prompt templates (`generation/prompt_templates.py`) guide generation.
2. **Revision** (`generation/revise.py`): GPT-5.2 rephrases each question so that it does not reveal its answer, removing superficial cues that could allow a model to guess correctly without watching the game.
3. **Automated quality check** (`generation/supported.py`): GPT-5-mini filters out candidates not fully supported by the evidence, removing those that draw conclusions not explicitly stated or reference information unavailable from the video alone (e.g., interviews, career statistics). Retains ~69.2% of the initial set.
4. **Language-bias filtering** (`generation/blind_filter.py`): Each candidate is presented to GPT-5.2 and Gemini-3-Flash in a *blind* setting — the models receive only the question, without any video or game context. If either model scores 3/5 or higher, the candidate is removed. Only ~5.2% survives this stage.
5. **Human review**: Expert annotators verify that each surviving candidate is (1) fully supported by evidence and (2) not answerable using general sports knowledge alone.

## Evaluation

### Protocol

Models are allowed up to *k* candidate answers per question (default *k*=5). Each candidate is independently scored by an LLM judge against the ground-truth answer, and the **top-*k* score** (best among all candidates) is reported. This accounts for the fact that strategic reasoning questions may admit more than one valid answer grounded in different aspects of the game.

### LLM Judge

The judge is **DeepSeek-V3** (via OpenRouter), selected for reproducibility as an open-source model. It scores each predicted answer on a 0–5 scale, prioritizing alignment on key ideas and reasoning traces over minor factual details. The judge outputs structured JSON with chain-of-thought analysis (`gt_analysis`, `pred_analysis`, `justification_cot`, `score`). See the full rubric and prompt in `evaluation/prompts.py`.

### Supported Models

Four model wrappers are provided in `evaluation/models/`:

| Model | Class | Temporal Sampling |
|---|---|---|
| GPT-5.2 | `gpt.GPT` | 500 uniformly sampled frames, resized to 400px short side |
| Gemini 3.1 Pro | `gemini.Gemini` | Full video compressed to 1 hr via ffmpeg, low resolution |
| Qwen3-VL-32B | `qwen.Qwen` | 768 frames via vLLM |
| Molmo 2-8B | `molmo.Molmo` | 300 uniformly sampled frames |

### Running Evaluation

**Requirements:**
- `OPENROUTER_API_KEY` environment variable (for the DeepSeek-V3 judge)
- Model-specific API keys or local GPU resources

```bash
# Via SLURM (see evaluate.slurm)
export OPENROUTER_API_KEY=<your_key>
python -m evaluation.evaluate
```

Uncomment the desired model in `evaluation/evaluate.py` and provide any required API keys. Results are written to `outputs/<model_name>/k-<k>/`.

Output files per model:
- `results.json` — full QA data with model responses and scores
- `llm_judge_trace.json` — detailed judge reasoning for each candidate
- `results.txt` — per-sport and overall average scores

## Notes

- Data config on HF: `t4_strategic_reasoning_qa`
- Default eval config: [`configs/t4.yaml`](../../../configs/t4.yaml)
- No train split — T4 is evaluation-only
