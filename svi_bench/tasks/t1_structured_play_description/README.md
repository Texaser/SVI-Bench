# T1 — Structured Play Description

## Task

Input: a 5–10 s sports clip.
Output: a short structured caption describing the play (who did what, when,
score state).

## Metric

LLM-as-a-judge scoring against reference captions. Default judge model
is `gpt-5.2-2025-12-11`; swap via `JUDGE_MODEL` env var or positional arg.

## Install

```bash
pip install "svi-bench[t1]"
```

The T1 environment also covers T2 (the model is jointly trained).

## Data

```bash
huggingface-cli download MVP-Group/SVI-Bench \
    --include 'T1/*' --include 'T2/data/*' \
    --local-dir "$SVI_BENCH_DATA" --repo-type dataset
```

Layout under `$SVI_BENCH_DATA/T1/`:

```
captions/{basketball,hockey,soccer}_caption_{train_100k,val_1k,test_5k}.json
{basketball,hockey,soccer}/train/shard_{00..09}.zip
{basketball,hockey,soccer}/val.zip
{basketball,hockey,soccer}/test.zip
checkpoint/                    # T1+T2 jointly-trained LLaVA-Video-7B (~15 GB)
```

JSON entries carry `video` (relative path to a `.mp4` inside the sport's
shards) and `caption` (reference text).

## Usage

```bash
HERE=svi_bench/tasks/t1_structured_play_description
```

### Train

```bash
bash $HERE/train.sh
```

Defaults: 1 epoch, lr 1e-5, full fine-tune of all parts (vision tower +
projector + LM). DeepSpeed ZeRO-3. Trains on the joint T1+T2 pool
(`configs/sports_100k.yaml`); override `DATA_YAML` to train on a single
sport or task. Outputs to `./work_dirs/sports_100k_f16_full_ft/`.

### Inference + Eval

```bash
bash $HERE/eval/run_caption.sh    [MODEL_PATH] [RESULTS_DIR]
bash $HERE/eval/run_llm_judge.sh  [JUDGE_MODEL] [RESULTS_DIR]
```

`run_caption.sh` writes per-clip captions to
`<results_dir>/<sport>/<split>/caption_eval_f16_outputs.json`.
`run_llm_judge.sh` scores those captions and writes
`..._scores_all.json` + `..._scores_avg.json` next to them.

`MODEL_PATH` defaults to `$SVI_BENCH_DATA/T1/checkpoint`.

## Files

| Path | Role |
|---|---|
| `train.sh` | DeepSpeed launcher for joint T1+T2 fine-tune |
| `eval/run_caption.sh` | caption generation per sport / split |
| `eval/run_llm_judge.sh` | LLM-as-a-judge scoring on generated captions |
| `llava/`, `trl/` | vendored LLaVA-NeXT + TRL slices |
| `configs/` | per-sport / per-task data YAMLs + DeepSpeed configs |
| `run.py` | `svi-bench evaluate --task t1` CLI entry |
