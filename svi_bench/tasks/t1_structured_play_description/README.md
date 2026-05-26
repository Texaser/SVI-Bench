# T1 — Structured Play Description

## Task

Inputs:

- a 5–10 s clip from a basketball, soccer, or hockey game.

Output: a single-sentence caption that names the players involved, the
action they performed, the time on the game clock, and the current
score.

Reference captions look like:

> *With 1:06 left in the 4th quarter, Elisa Pinzan committed a foul
> during the matchup between South Florida and North Carolina State,
> with the current score standing at 63–73.*

## Metrics

| Metric | Definition |
|---|---|
| **LLM-as-a-judge score** | A judge model rates each generated caption against the reference on player-name accuracy, action correctness, and time/score fidelity. Per-clip scores are written to `..._scores_all.json` and averaged into `..._scores_avg.json`. |

Default judge: `gpt-5.2-2025-12-11`. Override via `JUDGE_MODEL`.

## Install

```bash
pip install "svi-bench[t1]"
```

The `[t1]` environment also covers T2; the two tasks share a checkpoint.

## Data

```bash
huggingface-cli download MVP-Group/SVI-Bench --repo-type dataset \
    --local-dir "$SVI_BENCH_DATA" --include 'T1/*'
```

Layout under `$SVI_BENCH_DATA/T1/` (`<sport>` ∈ `{basketball, hockey, soccer}`):

```
captions/<sport>_caption_train_<N>k.json   100k basketball + hockey, 80k soccer
captions/<sport>_caption_val_1k.json       1k val per sport
captions/<sport>_caption_test_5k.json      5k test per sport
<sport>/train/shard_{00..09}.zip           video shards
<sport>/val.zip
<sport>/test.zip
checkpoint/                                LLaVA-Video-7B fine-tune (~15 GB)
```

Each JSON entry:

```json
{
  "video": "basketball/val/65126250.mp4",
  "data_source": "65126250",
  "caption": "With 1:06 left in the 4th quarter, Elisa Pinzan committed a foul..."
}
```

`video` is relative to `$SVI_BENCH_DATA/T1/`. Unzip each sport's
`val.zip` / `test.zip` / `train/shard_*.zip` in place so the clips land
at the paths the JSON expects.

## Usage

```bash
HERE=svi_bench/tasks/t1_structured_play_description
```

### Train

```bash
bash $HERE/train.sh
```

Full fine-tune of `lmms-lab/LLaVA-Video-7B-Qwen2` (vision tower,
projector, and language model) under DeepSpeed ZeRO-3. 1 epoch, lr 1e-5,
checkpoint every 500 steps. Trains on the joint T1 caption + T2 QA pool
(`configs/sports_100k.yaml`). Override `DATA_YAML` for the
caption-only, QA-only, or per-sport variants:

```bash
DATA_YAML=$HERE/configs/sports_caption_100k.yaml bash $HERE/train.sh
DATA_YAML=$HERE/configs/basketball_100k.yaml bash $HERE/train.sh
```

Outputs to `./work_dirs/sports_100k_f16_full_ft/`.

### Inference

```bash
bash $HERE/eval/run_caption.sh [MODEL_PATH] [RESULTS_DIR]
```

Iterates over all three sports for the chosen `SPLIT` (default `val`,
set `SPLIT=test` for test). `MODEL_PATH` defaults to
`$SVI_BENCH_DATA/T1/checkpoint`.

Generated captions land at:

```
<RESULTS_DIR>/<sport>/<split>/caption_eval_f16_outputs.json
```

### Evaluation

```bash
bash $HERE/eval/run_llm_judge.sh [JUDGE_MODEL] [RESULTS_DIR]
```

Consumes the caption outputs from `### Inference` and writes:

```
<RESULTS_DIR>/<sport>/<split>/caption_eval_f16_outputs_<JUDGE>_scores_all.json
<RESULTS_DIR>/<sport>/<split>/caption_eval_f16_outputs_<JUDGE>_scores_avg.json
```

`*_scores_avg.json` carries the headline number; `*_scores_all.json`
has per-clip detail.

## Files

| Path | Role |
|---|---|
| `train.sh` | DeepSpeed launcher for the joint T1+T2 fine-tune |
| `eval/run_caption.sh` | caption generation per sport / split |
| `eval/run_llm_judge.sh` | LLM-as-a-judge scoring on generated captions |
| `llava/`, `trl/` | vendored LLaVA-NeXT + TRL slices (also used by T2 via `PYTHONPATH`) |
| `configs/` | data YAMLs + DeepSpeed `zero{2,3}.json` |
| `run.py` | `svi-bench evaluate --task t1` CLI entry |
