# T8 — Goal-Conditioned Action Generation

![T7 (top) and T8 (bottom)](../../../docs/figures/pillar3.png)

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
| **Goal accuracy** | Fraction of multiple-choice QA pairs (8 question types covering atomic action, play type, shot type, contested-ness, etc.) answered correctly by a fine-tuned LLaVA-Qwen video-language model run on the generated clip. Headline number is the entry-weighted (micro) accuracy. |

## Install

```bash
pip install "svi-bench[t8]"
```

One environment covers training, inference, and all three evaluation
metrics.

## Data

```bash
bash scripts/download_t7_t8.sh
```

Layout under `$SVI_BENCH_DATA/T8/basketball/`:

```
clips/{bucket}/{ID}.mp4         5 s game clip, 832×480, 15 fps
bboxes/{bucket}/{ID}.txt        per-frame player bboxes
backgrounds/{bucket}/{ID}.mp4   player-removed background
splits/{train,val,test}.txt    one ID per line
splits/test_{100,1000}.txt    100- and 1000-clip evaluation subsets
captions.json                 ID -> {refined_instruction, player_specifications}
qa_test/Q*.json               goal-accuracy question bank
```

`ID` is a zero-padded integer; `bucket` is `ID // 741`.

`captions.json` schema (top-level keys are sample IDs):

```jsonc
{
  "0000000": {
    "refined_instruction": "Simulate Player #15 performing a Pick'n'Roll.",
    "player_specifications": [
      {
        "jersey_number": "#15",
        "action":        "Pick'n'Roll",
        "start_bbox":    {"x1": 0.540, "y1": 0.403, "x2": 0.620, "y2": 0.651},
        "end_bbox":      {"x1": 0.475, "y1": 0.456, "x2": 0.544, "y2": 0.695}
      }
    ]
  }
}
```

- `refined_instruction` — generation prompt.
- `player_specifications` — target player(s) for this clip (one entry per
  player). Bbox coordinates are normalized to [0, 1] (×width / ×height).
  `action`, `jersey_number`, and the bboxes are referenced by the
  evaluation pipeline.

Additional artifacts pulled by `download_t7_t8.sh`:

- `T8/llava_qa_checkpoint/` — fine-tuned LLaVA-Qwen QA model (~15 GB),
  used by goal accuracy.
- `shared/tracker_weights/` — YOLOX + MixFormer-ViT sports tracker
  (~1.2 GB, shared with T7), symlinked into `eval/pretrained/`.

## Train

```bash
bash svi_bench/tasks/t8_goal_conditioned_action_generation/train.sh
```

Defaults: 5 epochs, lr 1e-4, save every 2000 steps. LoRA rank 32 on the
DiT side. `train.sh` resumes from a T7 checkpoint via `--lora_checkpoint`.
Outputs to
`./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_t8/`.

## Inference

```bash
bash svi_bench/tasks/t8_goal_conditioned_action_generation/inference/infer.sh
```

Picks up the latest `step-*.safetensors` checkpoint under the LoRA output
dir and runs `test_1000` sharded across `NUM_GPUS=8`. Pass an alternate
checkpoint dir as `$1`.

Pre-trained T8 LoRA checkpoint is on
[`MVP-Group/SVI-Bench`](https://huggingface.co/datasets/MVP-Group/SVI-Bench/tree/main/T8):

```bash
bash svi_bench/tasks/t8_goal_conditioned_action_generation/download_checkpoint.sh
```

## Evaluation

```bash
HERE=svi_bench/tasks/t8_goal_conditioned_action_generation

# 1. Last-frame mIoU
bash $HERE/eval/run_basketball.sh         <VIDEO_DIR>

# 2. Last-frame feature similarity (reuses tracker output from step 1)
bash $HERE/eval/run_basketball_featsim.sh <VIDEO_DIR>

# 3. Goal accuracy (LLaVA-Qwen QA)
bash $HERE/eval/run_basketball_goalacc.sh <VIDEO_DIR>
```

Results:

```
<VIDEO_DIR>/video_miou_results/summary.json
<VIDEO_DIR>/feature_sim/summary.json
<VIDEO_DIR>/goal_accuracy_results/summary.json    # per-type + micro + macro
```

## Files

| Path | Role |
|---|---|
| `train.sh`, `train.py` | training entry |
| `inference/infer.{sh,py}` | multi-GPU inference dispatcher |
| `inference/split_validation_set.py` | shards a split file across GPUs |
| `validate.py` | in-training validation hook |
| `eval/run_basketball.sh` | tracker + last-frame mIoU |
| `eval/run_basketball_featsim.sh` | feature similarity |
| `eval/run_basketball_goalacc.sh` | goal-accuracy QA wrapper |
| `eval/prepare_qa_for_method.py` | filters QA bank for the user's videos (auto-invoked) |
| `eval/test_llavaov.py` | QA inference worker |
| `eval/video_miou.py`, `eval/feature_sim.py`, `eval/eval_generated_videos.py` | metric workers |
| `eval/yolox/`, `eval/MixViT/`, `eval/exps/`, `eval/llava/` | tracker + QA model code |
| `diffsynth/` | Wan2.1-Fun pipeline |
| `evaluate.py` | `svi-bench evaluate --task t8` CLI entry |
