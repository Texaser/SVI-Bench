# T2 — Fine-Grained Action QA

## Task

Given a 5–10 s clip from a basketball, soccer, or hockey game and a
multi-choice question about what's happening in the clip, output the
letter of the correct option.

Sample entry:

```json
{
  "id":            174849758,
  "video":         "basketball/val/85290206.mp4",
  "question_type": "player_position",
  "question":      "Based on the clip, identify the position of the player performing 3 Pt Made.",
  "options":       ["A: Guard", "B: Center", "C: Forward"],
  "answer":        "A",
  "data_source":   "85290206"
}
```

The val split carries 10k QA pairs per sport, drawn from 18 question
types:

| Group | Question types |
|---|---|
| Action | `atomic_action_recognition`, `play_type`, `shot_type`, `dribble_move`, `action_sequence`, `drive_direction`, `contested_shot`, `shooting_hand` |
| Player | `player_position`, `player_name`, `player_jersey_number`, `player_skill_level` |
| Spatial | `court_spatial_position` |
| Game state | `remaining_time`, `shot_clock`, `which_period`, `current_score`, `teams_identification` |

## Metric

Multi-choice accuracy. The eval pipeline writes both:

- `qa_eval_f16_outputs.json` — per-question prediction vs. ground truth
- `qa_eval_f16_results.json` — overall accuracy + breakdown by
  `question_type` (and an `overall` aggregate)

## Install

```bash
pip install "svi-bench[t2]"
```

`[t2]` aliases `[t1]` — same env covers both. The trained checkpoint at
`T1/checkpoint/` on HF serves both tasks.

## Data

```bash
huggingface-cli download MVP-Group/SVI-Bench \
    --include 'T1/checkpoint/*' --include 'T2/*' \
    --local-dir "$SVI_BENCH_DATA" --repo-type dataset
```

Layout under `$SVI_BENCH_DATA/T2/`:

```
data/{basketball,hockey,soccer}_qa_train_100k.json   # 100k train QA per sport
data/{basketball,hockey,soccer}_qa_val_10k.json      # 10k val
data/{basketball,hockey,soccer}_qa_test_10k.json     # 10k test
{basketball,hockey,soccer}/train/shard_{00..09}.zip  # video shards
{basketball,hockey,soccer}/val.zip
{basketball,hockey,soccer}/test.zip
```

Plus the shared checkpoint at `$SVI_BENCH_DATA/T1/checkpoint/` (~15 GB).

## Usage

```bash
HERE=svi_bench/tasks/t2_fine_grained_action_qa
T1=svi_bench/tasks/t1_structured_play_description
```

### Train

T1 and T2 are jointly trained. See
[T1's `## Train`](../t1_structured_play_description/README.md#train).
To train on T2 (QA) data only:

```bash
DATA_YAML=$T1/configs/sports_qa_100k.yaml bash $T1/train.sh
```

### Inference + Eval

```bash
bash $HERE/eval/run_qa.sh [MODEL_PATH] [RESULTS_DIR]
```

Iterates over all three sports for the chosen split (`SPLIT=val` by
default, set `SPLIT=test` for the test set). `MODEL_PATH` defaults to
`$SVI_BENCH_DATA/T1/checkpoint`. Results land at:

```
<RESULTS_DIR>/<sport>/<split>/qa_eval_f16_outputs.json
<RESULTS_DIR>/<sport>/<split>/qa_eval_f16_results.json
```

`qa_eval_f16_results.json` reports overall accuracy plus a per-
`question_type` breakdown.

## Files

| Path | Role |
|---|---|
| `eval/run_qa.sh` | multi-choice QA worker (calls T1's vendored `llava/eval/eval_sports.py`) |
| `run.py` | `svi-bench evaluate --task t2` CLI entry |
