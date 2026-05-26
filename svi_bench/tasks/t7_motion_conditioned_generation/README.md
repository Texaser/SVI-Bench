# T7 — Motion-Conditioned Generation

![T7 (top) and T8 (bottom)](../../../docs/figures/pillar3.png)

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
pip install "svi-bench[t7]"
bash scripts/download_t7_t8.sh                                # ~50 GB; T7+T8 data + tracker weights
bash svi_bench/tasks/t7_motion_conditioned_generation/download_checkpoint.sh basketball

HERE=svi_bench/tasks/t7_motion_conditioned_generation

# Inference (test_100 split, 8 GPUs)
SPORT=basketball bash $HERE/inference/infer.sh ./checkpoints/T7/basketball

# Eval (Video mIoU + feature similarity)
STEP_DIR=./checkpoints/T7/basketball/validation/step-<N>
VALIDATION_DIR=$STEP_DIR bash $HERE/eval/run_basketball.sh
bash $HERE/eval/run_basketball_featsim.sh $STEP_DIR
```

## Task

Inputs:

- an initial frame,
- a player-removed background video,
- per-player bounding-box trajectories (one box per player per frame).

Output: a 5–10 s video in which players follow the prescribed
trajectories with physical and temporal coherence.

T7 targets multi-agent settings with 10+ players moving simultaneously,
interacting, and occluding one another.

## Metrics

| Metric | Definition |
|---|---|
| **Video mIoU** | Spatiotemporal alignment between generated and reference trajectories, accumulated across all frames and matched track pairs. |
| **Temporal feature similarity** | SigLIP2 cosine similarity between per-player crops in generated and reference videos, IoU-gated by the tracker. |

## Install

```bash
pip install "svi-bench[t7]"
```

## Data

```bash
bash scripts/download_t7_t8.sh
```

Layout under `$SVI_BENCH_DATA/T7/{basketball,soccer}/`:

```
clips/{bucket}/{ID}.mp4         5 s game clip, 832×480, 15 fps
bboxes/{bucket}/{ID}.txt        per-frame player bboxes
backgrounds/{bucket}/{ID}.mp4   player-removed background
splits/{train,val,test}.txt     one ID per line
splits/test_100.txt             100-clip evaluation subset
```

`ID` is a zero-padded integer. `bucket` is `ID // 1668` (basketball) or
`ID // 1236` (soccer).

## Usage

```bash
HERE=svi_bench/tasks/t7_motion_conditioned_generation
```

### Train

```bash
SPORT=basketball bash $HERE/train.sh
SPORT=soccer     bash $HERE/train.sh
```

Defaults: 3 epochs, lr 1e-4, save every 2000 steps. LoRA rank 32 on the
DiT side (targets `q,k,v,o,ffn.0,ffn.2`). Outputs to
`./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_${SPORT}/`.

### Inference

```bash
SPORT=basketball bash $HERE/inference/infer.sh
SPORT=soccer     bash $HERE/inference/infer.sh
```

Picks up the latest `step-*.safetensors` under the LoRA output dir and
runs `test_100` sharded across `NUM_GPUS=8`. Pass an alternate output dir
as `$1`. Per-clip generated videos land at

```
<output_dir>/validation/step-<N>/<clip>/generated.mp4
```

The `<output_dir>/validation/step-<N>` path is the `STEP_DIR` consumed by
the eval wrappers below.

Pre-trained T7 LoRA checkpoints (basketball + soccer) are on
[`MVP-Group/SVI-Bench`](https://huggingface.co/datasets/MVP-Group/SVI-Bench/tree/main/T7):

```bash
bash $HERE/download_checkpoint.sh basketball
bash $HERE/download_checkpoint.sh soccer
```

### Evaluation

```bash
STEP_DIR=<output_dir>/validation/step-<N>

# 1. Video mIoU (tracker + holistic mIoU)
VALIDATION_DIR=$STEP_DIR bash $HERE/eval/run_basketball.sh
VALIDATION_DIR=$STEP_DIR bash $HERE/eval/run_soccer.sh

# 2. Feature similarity (reuses tracker output from step 1)
bash $HERE/eval/run_basketball_featsim.sh $STEP_DIR
bash $HERE/eval/run_soccer_featsim.sh     $STEP_DIR
```

`VALIDATION_DIR` makes `run_*.sh` auto-flatten the per-clip `generated.mp4`
files into a flat `<step_dir>/generated_flat/` before tracking. The
featsim wrappers read the per-clip layout directly.

Results:

```
$STEP_DIR/generated_flat/video_miou_results/summary.json
$STEP_DIR/feature_sim/summary.json
```

## Files

| Path | Role |
|---|---|
| `train.sh` | training entry |
| `inference/infer.sh` | multi-GPU inference dispatcher |
| `eval/run_{basketball,soccer}.sh` | tracker + Video mIoU |
| `eval/run_{basketball,soccer}_featsim.sh` | feature similarity |
| `run.py` | `svi-bench evaluate --task t7` CLI entry (dispatches to `inference/infer.sh`) |
