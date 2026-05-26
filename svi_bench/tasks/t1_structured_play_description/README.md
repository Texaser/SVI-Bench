# T1 — Structured Play Description

## Task

Given a 5–10 s clip from a basketball, soccer, or hockey game, produce a
single-sentence caption that names the players involved, the action they
performed, the time on the game clock, and the current score.

Reference captions look like:

> *With 1:06 left in the 4th quarter, Elisa Pinzan committed a foul
> during the matchup between South Florida and North Carolina State,
> with the current score standing at 63–73.*

The reference set is split across three sports: basketball, soccer,
hockey.

## Metric

LLM-as-a-judge scoring. The judge model rates each generated caption
along multiple axes (player-name accuracy, action correctness, time /
score fidelity) and returns a numeric overall score. Per-clip scores are
written to `..._scores_all.json` and averaged into `..._scores_avg.json`.

Default judge: `gpt-5.2-2025-12-11`. Swap via `JUDGE_MODEL` env var or
the first positional arg to `eval/run_llm_judge.sh`.

## Install

```bash
pip install "svi-bench[t1]"
```

T1 and T2 share this environment; the model is jointly trained on both.

## Data

```bash
huggingface-cli download MVP-Group/SVI-Bench \
    --include 'T1/*' --local-dir "$SVI_BENCH_DATA" --repo-type dataset
```

Layout under `$SVI_BENCH_DATA/T1/`:

```
captions/{basketball,hockey}_caption_train_100k.json   # 100k train samples
captions/soccer_caption_train_80k.json                 # 80k train samples
captions/{basketball,hockey,soccer}_caption_val_1k.json    # 1k val
captions/{basketball,hockey,soccer}_caption_test_5k.json   # 5k test
{basketball,hockey,soccer}/train/shard_{00..09}.zip    # video shards
{basketball,hockey,soccer}/val.zip
{basketball,hockey,soccer}/test.zip
checkpoint/                                            # fine-tuned LLaVA-Video-7B, ~15 GB
```

Each JSON entry:

```json
{
  "video":       "basketball/val/65126250.mp4",
  "data_source": "65126250",
  "caption":     "With 1:06 left in the 4th quarter, Elisa Pinzan committed a foul..."
}
```

`video` is relative to `$SVI_BENCH_DATA/T1/` — unzip the matching
`<sport>/<split>.zip` (or `<sport>/train/shard_*.zip`) to land each clip
at the path the JSON expects.

## Usage

```bash
HERE=svi_bench/tasks/t1_structured_play_description
```

### Train

```bash
bash $HERE/train.sh
```

Full fine-tune of `lmms-lab/LLaVA-Video-7B-Qwen2` (all parts: vision tower
+ projector + language model) under DeepSpeed ZeRO-3. 1 epoch, lr 1e-5,
save every 500 steps. Default training pool is the **combined T1 caption
+ T2 QA** set (`configs/sports_100k.yaml`); override via `DATA_YAML` for
single-task or per-sport variants:

```bash
DATA_YAML=$HERE/configs/sports_caption_100k.yaml bash $HERE/train.sh   # T1 only
DATA_YAML=$HERE/configs/basketball_100k.yaml      bash $HERE/train.sh  # one sport
```

Outputs to `./work_dirs/sports_100k_f16_full_ft/`.

### Inference + Eval

```bash
# 1. Generate captions (writes <sport>/<split>/caption_eval_f16_outputs.json)
bash $HERE/eval/run_caption.sh    [MODEL_PATH] [RESULTS_DIR]

# 2. LLM-as-a-judge scoring (consumes step 1's output)
bash $HERE/eval/run_llm_judge.sh  [JUDGE_MODEL] [RESULTS_DIR]
```

`MODEL_PATH` defaults to `$SVI_BENCH_DATA/T1/checkpoint`. Both wrappers
iterate over all three sports for the chosen split (`SPLIT=val` by
default, set `SPLIT=test` for the test set).

Results land at:

```
<RESULTS_DIR>/<sport>/<split>/caption_eval_f16_outputs.json
<RESULTS_DIR>/<sport>/<split>/caption_eval_f16_outputs_<JUDGE>_scores_all.json
<RESULTS_DIR>/<sport>/<split>/caption_eval_f16_outputs_<JUDGE>_scores_avg.json
```

## Files

| Path | Role |
|---|---|
| `train.sh` | DeepSpeed launcher for the joint T1+T2 fine-tune |
| `eval/run_caption.sh` | per-sport caption generation |
| `eval/run_llm_judge.sh` | LLM-as-a-judge scoring on generated captions |
| `llava/`, `trl/` | vendored LLaVA-NeXT + TRL slices (shared with T2 via PYTHONPATH) |
| `configs/` | data YAMLs + DeepSpeed `zero{2,3}.json` |
| `run.py` | `svi-bench evaluate --task t1` CLI entry |
