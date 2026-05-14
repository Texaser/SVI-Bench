# T5 — Outcome Forecasting

**Pillar 2: Causal Reasoning** &nbsp;|&nbsp; 3–15 min video segments &nbsp;|&nbsp; MCQ &nbsp;|&nbsp; Accuracy

## Task Overview

Given a video segment capturing a sequence of play (3–15 minutes) and a multiple-choice question about a future event, the model must predict the outcome by selecting the correct answer from a candidate set. The target event occurs *beyond* the input window, requiring the model to infer the most probable course of game development from visual evidence rather than trend extrapolation.

Questions span three forecasting categories:

| Category | Description | Example |
|---|---|---|
| **Performance forecasting** | Predicting future player- or team-level statistics | *How many points will this player score from the end of this segment until the end of the half?* |
| **Game state evolution** | Anticipating how the game state will change, including final scores, possession shifts, and overall outcomes | *Predict which team will win the game by the end of regulation.* |
| **Strategic intention** | Inferring the most likely tactics from observed play patterns | *Which play type will this team attempt most frequently?* |

## Question Types

We define 15 question types across three sports (6 shared, 4 basketball-specific, 2 hockey-specific, 3 soccer-specific):

| # | Question Type | Category | Sports |
|---|---|---|---|
| 1 | Player Statistics Prediction | Performance | All |
| 2 | Team Statistics Prediction | Performance | All |
| 3 | Player Statistics Milestone | Performance | All |
| 4 | Team Statistics Milestone | Performance | All |
| 5 | Game Outcome Prediction | Game state | All |
| 6 | Next Player Action Prediction | Game state | All |
| 7 | Player Most Attempted Play Type | Strategic intention | Basketball |
| 8 | Player Most Attempted Shot Type | Strategic intention | Basketball |
| 9 | Team Most Attempted Play Type | Strategic intention | Basketball |
| 10 | Team Most Attempted Shot Type | Strategic intention | Basketball |
| 11 | Player Most Attempted Shot Type | Strategic intention | Hockey |
| 12 | Team Most Attempted Shot Type | Strategic intention | Hockey |
| 13 | Team Most Attempted Attack Flank | Strategic intention | Soccer |
| 14 | Next Team to Score | Game state | Soccer |
| 15 | Team Most Possession | Strategic intention | Soccer |

Basketball has Q1–Q10, hockey has Q1–Q8 (mapped from type IDs 1–6, 11–12), and soccer has Q1–Q9 (mapped from type IDs 1–6, 13–15). The question type is encoded in each entry's `id` field (e.g., `Q1_0_404818`).

### Preventing Shortcut Solutions

Two design choices ensure the task requires genuine visual forecasting:

1. **Future-oriented questions.** All questions ask about *future* game progression relative to the observation window, not cumulative totals. This prevents models from exploiting seasonal statistical priors.
2. **Indirect player references.** Player names are never revealed. Instead, questions use indirect visual references grounded in the observation clip (e.g., *"the player who makes a 3-point shot during 4:13–4:23"*), forcing the model to visually identify the relevant player before any prediction.

## Dataset

| Sport | Train | Test | Video Hours | Avg. Clip Length |
|---|---|---|---|---|
| Basketball | 43,466 | 7,000 | 6,053 | 431.8 s |
| Hockey | 43,585 | 7,123 | 7,009 | 497.7 s |
| Soccer | 10,328 | 2,593 | 945 | 525.9 s |
| **Total** | **97,379** | **16,716** | **14,007** | |

### Data Format

Each JSON file contains a list of entries:

```json
{
  "id": "Q1_0_404818",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nAnalyze the 10-minute segment from the 3rd period. Focus on the player who makes a three-point attempt between time 1:19 and 1:29. How many total assists will this specific player record during the 4th period?\nA: 4\nB: 2\nC: 10\nD: 5\nE: 8"
    },
    {
      "from": "gpt",
      "value": "B"
    }
  ],
  "video": "/path/to/video.mp4"
}
```

- **`id`**: Encodes the question type and source game (e.g., `Q1_0_404818` is question type Q1).
- **`conversations`**: The `human` turn contains the question with `<video>` placeholder; the `gpt` turn contains the ground-truth answer letter.
- **`video`**: Absolute path to the video file. You will need to update these paths to match your local setup.

## Evaluation

### Metrics

- **Accuracy**: Top-1 accuracy of the predicted answer letter.
- **Calibration Error (CE)**: Predictions are grouped into *B*=5 equally spaced confidence bins. CE = (1/*B*) * sum |acc(*i*) - conf(*i*)| over all bins. CE = 0 indicates perfect calibration. (Applicable to models that output token-level probabilities, i.e., Qwen and Molmo.)

### Supported Models

Four inference scripts are provided in `evaluation/`, one per model family:

| Model | Script | Temporal Sampling | Dependencies |
|---|---|---|---|
| GPT | `evaluation/infer_gpt.py` | Configurable FPS (default 0.5), frames as base64 JPEG | `openai`, `decord`, `numpy`, `tqdm`, `Pillow` |
| Gemini | `evaluation/infer_gemini.py` | Direct video upload via Files API | `google-genai`, `tqdm` |
| Qwen3-VL | `evaluation/infer_qwen.py` | Configurable FPS (default 0.2), optional LoRA adapter | `torch`, `transformers`, `peft`, `decord`, `numpy`, `tqdm`, `Pillow` |
| Molmo2-8B | `evaluation/infer_molmo.py` | Configurable FPS (default 0.2) | `torch`, `transformers`, `molmo_utils`, `decord`, `numpy`, `tqdm` |

### Running Evaluation

**Qwen3-VL** (`evaluation/infer_qwen.py`) — supports optional LoRA adapter and multi-GPU distributed inference via `torchrun`:

```bash
# Single GPU
python evaluation/infer_qwen.py \
    --test_json dataset/basketball_test.json \
    --output outputs/basketball_qwen.json \
    --sample_fps 0.2

# With LoRA adapter
python evaluation/infer_qwen.py \
    --test_json dataset/hockey_test.json \
    --output outputs/hockey_qwen.json \
    --adapter /path/to/lora/checkpoint

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 evaluation/infer_qwen.py \
    --test_json dataset/soccer_test.json \
    --output outputs/soccer_qwen.json
```

**Molmo2** (`evaluation/infer_molmo.py`) — supports multi-GPU distributed inference via `torchrun`:

```bash
python evaluation/infer_molmo.py \
    --test_json dataset/basketball_test.json \
    --output outputs/basketball_molmo.json \
    --sample_fps 0.2
```

**GPT** (`evaluation/infer_gpt.py`) — uses OpenAI Responses API with resumption support:

```bash
export OPENAI_API_KEY="sk-..."

python evaluation/infer_gpt.py \
    --test_json dataset/basketball_test.json \
    --output outputs/basketball_gpt.json \
    --model gpt-4o \
    --frame_fps 0.5 \
    --image_detail low
```

**Gemini** (`evaluation/infer_gemini.py`) — uploads video directly via Files API with resumption support:

```bash
export GEMINI_API_KEY="AIza..."

python evaluation/infer_gemini.py \
    --test_json dataset/soccer_test.json \
    --output outputs/soccer_gemini.json \
    --model gemini-2.5-flash-preview
```

## Training

We provide a LoRA fine-tuning script for Qwen3-VL using [ms-swift](https://github.com/modelscope/ms-swift). First convert the training JSON to Swift's JSONL format, then run training:

```bash
# 1. Convert training data (can combine multiple sports)
python training/convert_train_to_jsonl.py \
    --input dataset/basketball_train.json dataset/hockey_train.json dataset/soccer_train.json \
    --output dataset/train.jsonl

# 2. Run LoRA fine-tuning (edit training/train_qwen.sh to adjust GPU count, paths, etc.)
bash training/train_qwen.sh
```

Key training settings (in `training/train_qwen.sh`):
- **LoRA** rank 8 / alpha 32 on all linear layers, ViT and aligner frozen
- **Video sampling**: 0.2 FPS with max 768 visual tokens per video
- **Sequence length**: 50,000 tokens
- **Optimizer**: AdamW with lr=1e-4, bfloat16 precision

The resulting LoRA checkpoint can be loaded for inference via `evaluation/infer_qwen.py --adapter <checkpoint_path>`.

## Directory Structure

```
t5_outcome_forecasting/
├── __init__.py
├── README.md
├── dataset/
│   ├── basketball_train.json
│   ├── basketball_test.json
│   ├── hockey_train.json
│   ├── hockey_test.json
│   ├── soccer_train.json
│   └── soccer_test.json
├── evaluation/
│   ├── infer_qwen.py
│   ├── infer_molmo.py
│   ├── infer_gpt.py
│   └── infer_gemini.py
├── training/
│   ├── convert_train_to_jsonl.py
│   └── train_qwen.sh
├── logs/
└── outputs/
```

## Notes

- Data config on HF: `t5_outcome_forecasting`
- Default eval config: [`configs/t5.yaml`](../../../configs/t5.yaml)
