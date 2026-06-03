# T8 — Goal-Conditioned Action Generation

![T7 (top) and T8 (bottom)](../t7_motion_conditioned_generation/figures/pillar3.png)

## Quickstart

> If your CUDA driver is < 13, pre-install a CUDA-matching torch wheel
> first (otherwise pip pulls the newest cu13-only wheel and GPU is
> disabled):
>
> ```bash
> # CUDA 12.1 example; swap cu121 for your stack
> pip install "torch>=2.0,<2.10" --index-url https://download.pytorch.org/whl/cu121
> ```

```bash
git clone https://github.com/Texaser/SVI-Bench && cd SVI-Bench
pip install "svi-bench[t8]"

HERE=svi_bench/tasks/t8_goal_conditioned_action_generation
T7=svi_bench/tasks/t7_motion_conditioned_generation
CKPT=$HERE/checkpoints/T8/basketball/checkpoint.safetensors

bash $T7/scripts/download_t7_t8.sh          # ~65 GB; T7+T8 data, tracker weights, LLaVA-Qwen checkpoint
bash $HERE/download_checkpoint.sh

# Inference (test_1000 split, 8 GPUs)
bash $HERE/inference/infer.sh $CKPT

# Eval (last-frame mIoU + feature similarity + goal accuracy)
VIDEO_DIR=$HERE/checkpoints/T8/basketball/validation/step-pretrained
bash $HERE/eval/run_basketball.sh         $VIDEO_DIR
bash $HERE/eval/run_basketball_featsim.sh $VIDEO_DIR
bash $HERE/eval/run_basketball_goalacc.sh $VIDEO_DIR
```

## Task

Inputs:

- an initial frame,
- a player-removed background video,
- start and end bounding boxes for each target player,
- a natural-language goal description.

Output: a 5–10 s video in which the specified players execute a coherent
action sequence that achieves the described objective.

## Metrics

| Metric | Definition |
|---|---|
| **Last-frame mIoU** | Bounding-box overlap between generated and target player positions at the last frame. |
| **Last-frame feature similarity** | SigLIP2 cosine similarity between per-player crops at the last frame, IoU-gated by the tracker. |
| **Goal accuracy** | Fraction of multi-choice QA pairs about the generated clip answered correctly by a fine-tuned LLaVA-Qwen QA model. Headline is the entry-weighted (micro) accuracy across 8 question types. |

## Install

```bash
pip install "svi-bench[t8]"
```

One environment covers training, inference, and all three evaluation
metrics.

## Data

```bash
bash svi_bench/tasks/t7_motion_conditioned_generation/scripts/download_t7_t8.sh
```

Layout under `$SVI_BENCH_DATA/T8/basketball/`:

```
clips/{bucket}/{ID}.mp4         5.4 s game clip, 832×480, 15 fps, 81 frames
bboxes/{bucket}/{ID}.txt        per-frame player bboxes (MOT-style, 10 cols)
backgrounds/{bucket}/{ID}.mp4   player-removed background, same shape as clip
splits/{train,val,test}.txt     one ID per line
splits/test_{100,1000}.txt      100- and 1000-clip evaluation subsets
captions.json                   ID -> {refined_instruction, player_specifications}
qa_test/Q*.json                 goal-accuracy question bank (8 question types)
```

`ID` is a zero-padded integer.

Other artifacts pulled by `download_t7_t8.sh`:

- `T8/llava_qa_checkpoint/` — fine-tuned LLaVA-Qwen QA model (~15 GB),
  used by goal accuracy.
- `T8/tracker_weights/` — YOLOX + MixFormer-ViT sports tracker (~1.2 GB),
  symlinked into `eval/pretrained/`.

### `bboxes/{ID}.txt` format

One detection per line, comma-separated:

```
frame_id,track_id,x1,y1,x2,y2,confidence,-1,-1,-1
```

Coordinates are normalized to `[0, 1]` (×width / ×height). The trailing
`-1,-1,-1` are MOT placeholders (visibility, world-x, world-y) kept for
format compatibility; the eval pipeline ignores them.

### `captions.json` schema

Top-level keys are sample IDs. Each value:

```json
{
  "0000000": {
    "refined_instruction": "Simulate Player #15 performing a Pick'n'Roll.",
    "player_specifications": [
      {
        "jersey_number": "#15",
        "start_bbox": {"x1": 0.540519, "y1": 0.403172, "x2": 0.620234, "y2": 0.651138},
        "end_bbox": {"x1": 0.475335, "y1": 0.455800, "x2": 0.544082, "y2": 0.695225},
        "action": "Pick'n'Roll"
      }
    ]
  }
}
```

- `refined_instruction` — goal instruction.
- `player_specifications` — target player(s); 1–3 entries. Bbox
  coordinates are normalized to [0, 1] (×width / ×height).

### `qa_test/Q*.json` schema

Eight question types, one JSON file each, holding the LLM-as-a-judge
multi-choice questions used by goal accuracy:

```
Q1_atomic_action_recognition.json
Q3_contested_shot.json
Q3_dribble_move.json
Q3_drive_direction.json
Q3_play_type.json
Q3_shooting_hand.json
Q3_shot_type.json
Q4_spatial_position.json
```

Each file is a JSON list whose entries follow the LLaVA-OneVision
conversation format:

```json
{
  "id": "0069003_player0",
  "video": "clips/93/0069003.mp4",
  "start_bbox": {"x1": 0.576766, "y1": 0.578147, "x2": 0.644978, "y2": 0.818599},
  "jersey_number": "#11",
  "normalized_action": "3 PT Missed",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat atomic action is being defended by the player in the red bounding box? ...\nA: Screen\nB: Free Throw Missed\nC: Rebound\nD: 3 PT Missed\nE: 2 PT Shot"},
    {"from": "gpt",   "value": "D"}
  ]
}
```

- `id` — `<sample_id>_<suffix>` where `<suffix>` is the player slot
  (`player0..2`) or the QA category (`shot_type`, `play_type`, ...).
- `video` — clip path relative to `$SVI_BENCH_DATA/T8/basketball/`.
- `start_bbox` — first-frame bbox of the target player, used by
  `eval/prepare_qa_for_method.py` to render the red overlay on each
  generated clip.
- `jersey_number`, `normalized_action` — provenance metadata; the eval
  worker doesn't read them, but they're handy for slicing results.
- `conversations` — single-turn human / GPT pair. The human prompt
  starts with the `<image>` placeholder (replaced with the video) and
  ends with five `A:`–`E:` options; the GPT value is the gold letter.

## Usage

```bash
HERE=svi_bench/tasks/t8_goal_conditioned_action_generation
```

### Train

```bash
bash $HERE/train.sh
```

Defaults: 5 epochs, lr 1e-4, save every 2000 steps. LoRA rank 32 on the
DiT side. `train.sh` resumes from a T7 checkpoint via `--lora_checkpoint`.
Outputs to
`./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_t8/`.

### Inference

Pre-trained T8 LoRA checkpoint lives on
[`MVP-Group/SVI-Bench`](https://huggingface.co/datasets/MVP-Group/SVI-Bench/tree/main/T8).
Fetch it and run inference against it directly:

```bash
bash $HERE/download_checkpoint.sh
bash $HERE/inference/infer.sh $HERE/checkpoints/T8/basketball/checkpoint.safetensors
```

To run against your own training output instead, point `$1` at the LoRA
output dir; `infer.sh` picks the latest `step-*.safetensors`:

```bash
bash $HERE/inference/infer.sh ./models/train/<your-lora-dir>
```

Either form runs the `test_1000` split sharded across `NUM_GPUS=8`.
Per-clip generated videos land at

```
<checkpoint_dir>/validation/step-<N>/<clip>/generated.mp4
```

`<checkpoint_dir>` is the directory containing the loaded checkpoint;
`<N>` is the training step (`pretrained` for the HF checkpoint). That
path is `VIDEO_DIR` for the eval wrappers below.

### Evaluation

```bash
VIDEO_DIR=<output_dir>/validation/step-<N>

# 1. Last-frame mIoU
bash $HERE/eval/run_basketball.sh         $VIDEO_DIR

# 2. Last-frame feature similarity (reuses tracker output from step 1)
bash $HERE/eval/run_basketball_featsim.sh $VIDEO_DIR

# 3. Goal accuracy (LLaVA-Qwen QA)
bash $HERE/eval/run_basketball_goalacc.sh $VIDEO_DIR
```

Results:

```
$VIDEO_DIR/video_miou_results/summary.json
$VIDEO_DIR/feature_sim/summary.json
$VIDEO_DIR/goal_accuracy_results/summary.json     # per-type + micro + macro
```

## Files

| Path | Role |
|---|---|
| `train.sh` | training entry |
| `inference/infer.sh` | multi-GPU inference dispatcher |
| `eval/run_basketball.sh` | tracker + last-frame mIoU |
| `eval/run_basketball_featsim.sh` | feature similarity |
| `eval/run_basketball_goalacc.sh` | goal-accuracy QA |
| `run.py` | `svi-bench evaluate --task t8` CLI entry (dispatches to `inference/infer.sh`) |
