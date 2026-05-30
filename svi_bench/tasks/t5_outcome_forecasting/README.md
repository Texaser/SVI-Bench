# T5 — Outcome Forecasting

Given a 3–15 min video segment and a multiple-choice question about a future event, predict the outcome by selecting the correct answer (A–E). The target event occurs *beyond* the input window. Metrics: Accuracy and Calibration Error (CE).

Two design choices prevent shortcuts: (1) questions ask about *future* progression only, blocking statistical priors; (2) players are referenced via visual cues (e.g., *"the player who makes a 3-point shot during 4:13–4:23"*), forcing visual identification.

## 1. Install

```bash
conda env create -f environment.yaml && conda activate svi_t5
```

Or follow the official guides for the models you plan to use:
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) &nbsp;|&nbsp; [Molmo2](https://github.com/allenai/molmo)

For training, additionally install [ms-swift](https://github.com/modelscope/ms-swift): `pip install ms-swift[llm]`

## 2. Data

```bash
huggingface-cli download MVP-Group/SVI-Bench --repo-type dataset \
    --include "T5/**" --local-dir data/
```

Everything goes under `data/T5/`:

```
data/T5/
├── {basketball,hockey,soccer}_train.json
└── {basketball,hockey,soccer}_test.json
```

Each entry:

```json
{
  "id": "Q1_2846637",
  "conversations": [
    {"from": "human", "value": "<video>\n[question with options A–E]"},
    {"from": "gpt", "value": "B"}
  ],
  "video": "T5/basketball/shards/shard_07/Q1_5526717.mp4"
}
```

The `video` field is a relative path. Use `--video_root` at inference time to prepend your local root.

## 3. Evaluate

Four inference scripts in `evaluation/`, one per model family. All support resumption.

**Qwen3-VL** — supports LoRA adapter, `--video_root`, multi-GPU via `torchrun`:

```bash
torchrun --nproc_per_node=8 evaluation/infer_qwen.py \
    --test_json data/T5/basketball_test.json \
    --output outputs/basketball_qwen.json \
    --video_root /path/to/video/root \
    --sample_fps 0.2 \
    --adapter /path/to/lora/checkpoint
```

**Molmo2** — multi-GPU via `torchrun`:

```bash
torchrun --nproc_per_node=8 evaluation/infer_molmo.py \
    --test_json data/T5/basketball_test.json \
    --output outputs/basketball_molmo.json \
    --sample_fps 0.2
```

**GPT** — OpenAI Responses API:

```bash
export OPENAI_API_KEY="sk-..."
python evaluation/infer_gpt.py \
    --test_json data/T5/basketball_test.json \
    --output outputs/basketball_gpt.json \
    --model gpt-4o --frame_fps 0.5 --image_detail low
```

**Gemini** — direct video upload via Files API:

```bash
export GEMINI_API_KEY="AIza..."
python evaluation/infer_gemini.py \
    --test_json data/T5/soccer_test.json \
    --output outputs/soccer_gemini.json \
    --model gemini-2.5-flash-preview
```

An example SLURM launch script is provided in `evaluation/run.sh`.

**Calibration Error** — after running inference with option logits available:

```bash
python evaluation/calc_ce.py --results outputs/basketball_qwen.json
python evaluation/calc_ce.py --results outputs/basketball_qwen.json --num_bins 5
```

## 4. Train

LoRA fine-tuning for Qwen3-VL using [ms-swift](https://github.com/modelscope/ms-swift). Runs on 8 GPUs with DeepSpeed ZeRO-3.

```bash
# 1. Convert training data to Swift JSONL format
python training/convert_train_to_jsonl.py \
    --input data/T5/basketball_train.json data/T5/hockey_train.json data/T5/soccer_train.json \
    --output data/T5/train.jsonl

# 2. Run LoRA fine-tuning (edit training/train_qwen.sh to adjust GPU count, paths, etc.)
bash training/train_qwen.sh
```

Key settings: LoRA rank 8 / alpha 32, frozen ViT + aligner, 0.2 FPS, 50k seq length, AdamW lr=1e-4, bfloat16.

The resulting checkpoint can be loaded via `evaluation/infer_qwen.py --adapter <path>`.
