# T2 — Fine-Grained Action QA

## Task

Input: a 5–10 s sports clip + a multi-choice question about what's
happening (action category, player position, shot type, etc.).
Output: one of the provided answer letters.

## Metric

Multi-choice accuracy (overall + per `question_type`).

## Install

```bash
pip install "svi-bench[t2]"
```

T2 shares its environment + trained checkpoint with T1 (jointly trained
on the combined caption + QA pool). The Python deps are identical, and
the checkpoint lives at `T1/checkpoint/` on the HF dataset.

## Data

```bash
huggingface-cli download MVP-Group/SVI-Bench \
    --include 'T1/checkpoint/*' --include 'T2/*' \
    --local-dir "$SVI_BENCH_DATA" --repo-type dataset
```

Layout under `$SVI_BENCH_DATA/T2/`:

```
data/{basketball,hockey,soccer}_qa_{train_100k,val_10k,test_10k}.json
{basketball,hockey,soccer}/train/shard_{00..09}.zip
{basketball,hockey,soccer}/val.zip
{basketball,hockey,soccer}/test.zip
```

Each QA entry: `{id, video, question_type, question, options, answer}`.
`video` is a relative path to a `.mp4` inside the sport's shards.

## Usage

```bash
HERE=svi_bench/tasks/t2_fine_grained_action_qa
```

### Train

T1 and T2 are jointly trained — see
[`../t1_structured_play_description/README.md#train`](../t1_structured_play_description/README.md#train).
To train on T2 data only, point `train.sh` at the T2-only config:

```bash
DATA_YAML=$REPO_ROOT/svi_bench/tasks/t1_structured_play_description/configs/sports_qa_100k.yaml \
    bash $REPO_ROOT/svi_bench/tasks/t1_structured_play_description/train.sh
```

### Inference + Eval

```bash
bash $HERE/eval/run_qa.sh [MODEL_PATH] [RESULTS_DIR]
```

Writes `<results_dir>/<sport>/<split>/qa_eval_f16_{outputs,results}.json`.
`MODEL_PATH` defaults to `$SVI_BENCH_DATA/T1/checkpoint`.

## Files

| Path | Role |
|---|---|
| `eval/run_qa.sh` | multi-choice QA worker (uses T1's vendored llava/) |
| `run.py` | `svi-bench evaluate --task t2` CLI entry |
