# T2 — Fine-Grained Action QA

## Task

Inputs:

- a 5–10 s clip from a basketball, soccer, or hockey game,
- a multi-choice question about what's happening in the clip.

Output: the letter of the correct option.

Sample entry:

```json
{
  "id": 174849758,
  "video": "basketball/val/85290206.mp4",
  "question_type": "player_position",
  "question": "Based on the clip, identify the position of the player performing 3 Pt Made.",
  "options": ["A: Guard", "B: Center", "C: Forward"],
  "answer": "A",
  "data_source": "85290206"
}
```

Each `val` split carries 10k QA pairs per sport, drawn from 18 question
types grouped as:

| Group | Question types |
|---|---|
| Action | `atomic_action_recognition`, `play_type`, `shot_type`, `dribble_move`, `action_sequence`, `drive_direction`, `contested_shot`, `shooting_hand` |
| Player | `player_position`, `player_name`, `player_jersey_number`, `player_skill_level` |
| Spatial | `court_spatial_position` |
| Game state | `remaining_time`, `shot_clock`, `which_period`, `current_score`, `teams_identification` |

## Metrics

| Metric | Definition |
|---|---|
| **Multi-choice accuracy** | Fraction of QA pairs whose predicted letter matches `answer`. Reported overall and broken down by `question_type`. |

## Install

```bash
pip install "svi-bench[t2]"
```

`[t2]` aliases `[t1]`; the env covers both. The trained checkpoint at
`T1/checkpoint/` on HF serves both tasks.

## Data

```bash
huggingface-cli download MVP-Group/SVI-Bench --repo-type dataset \
    --local-dir "$SVI_BENCH_DATA" --include 'T1/checkpoint/*' --include 'T2/*'
```

Layout under `$SVI_BENCH_DATA/T2/` (`<sport>` ∈ `{basketball, hockey, soccer}`):

```
data/<sport>_qa_train_100k.json   100k train QA per sport
data/<sport>_qa_val_10k.json      10k val per sport
data/<sport>_qa_test_10k.json     10k test per sport
<sport>/train/shard_{00..09}.zip  video shards
<sport>/val.zip
<sport>/test.zip
```

Plus `$SVI_BENCH_DATA/T1/checkpoint/` (~15 GB) for the shared model.

## Usage

```bash
HERE=svi_bench/tasks/t2_fine_grained_action_qa
T1=svi_bench/tasks/t1_structured_play_description
```

### Train

T1 and T2 are jointly trained. See
[T1's `### Train`](../t1_structured_play_description/README.md#train).
To train on T2 (QA) data only:

```bash
DATA_YAML=$T1/configs/sports_qa_100k.yaml bash $T1/train.sh
```

### Inference + Evaluation

```bash
bash $HERE/eval/run_qa.sh [MODEL_PATH] [RESULTS_DIR]
```

Iterates over all three sports for the chosen `SPLIT` (default `val`,
set `SPLIT=test` for test). `MODEL_PATH` defaults to
`$SVI_BENCH_DATA/T1/checkpoint`. Results land at:

```
<RESULTS_DIR>/<sport>/<split>/qa_eval_f16_outputs.json   per-question predictions
<RESULTS_DIR>/<sport>/<split>/qa_eval_f16_results.json   accuracy overall + per question_type
```

## Files

| Path | Role |
|---|---|
| `eval/run_qa.sh` | multi-choice QA worker (calls T1's vendored `llava/eval/eval_sports.py`) |
| `run.py` | `svi-bench evaluate --task t2` CLI entry |
