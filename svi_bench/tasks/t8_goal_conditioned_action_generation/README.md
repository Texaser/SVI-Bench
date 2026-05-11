# T8 — Goal-Conditioned Action Generation

LoRA fine-tune of **Wan2.1-Fun-V1.1-1.3B-Control** that builds on T7 but
swaps the static prompt for **per-video polished captions** and switches
the bbox conditioning to **first/last-frame only** — i.e. the model has to
fill in the in-between motion that takes the players from their start
positions to their end positions. Typically chained off a T7 checkpoint via
`--lora_checkpoint`.

## Install

```bash
pip install "svi-bench[t8]"
```

Same dep set as T7 (torch / accelerate / peft / transformers / einops /
modelscope / imageio / ...).

## Run

### Train

```bash
bash svi_bench/tasks/t8_goal_conditioned_action_generation/train.sh
```

### Inference

T8 only covers basketball. Loads the latest `step-*.safetensors`
checkpoint under the LoRA output dir and generates video samples for the
task2 basketball test set, sharded across 8 GPUs by default:

```bash
bash svi_bench/tasks/t8_goal_conditioned_action_generation/inference/basketball.sh
```

Pass a different output dir as `$1` to override the default. Edit the data
paths inside the script to match your local layout.

The unified CLI is equivalent:

```bash
svi-bench evaluate --task t8 --model wan2.1-fun
```

## Files

- [`train.sh`](train.sh) — `accelerate launch` wrapper around the bundled
  `train.py`; passes `--polished_captions` and `--bbox_first_last_only`.
- [`train.py`](train.py) — training entry point, vendored verbatim from
  `DiffSynth-Studio/examples/wanvideo/model_training/train.py` (same file
  T7 ships).
- [`validate.py`](validate.py) — **in-training** task2 validation hook
  invoked by `train.py` via `$VALIDATION_SCRIPT`. Renders a small number
  of samples each save step.
- [`inference/`](inference/) — multi-GPU inference pipeline that loads
  the trained LoRA and generates video samples:
  - `basketball.{sh,py}` — full task2 test-set run, default 8 GPUs.
  - `split_validation_set.py` — helper that shards a test-set listing
    into N per-GPU split files.
- [`diffsynth/`](diffsynth/) — slimmed copy of the Wan2.1-Fun-related
  closure from upstream DiffSynth-Studio. T7 ships an identical copy.
- [`evaluate.py`](evaluate.py) — Python wrapper exposed via
  `svi-bench evaluate --task t8`.

## Data

- Bbox folder: `train_task2_final.txt` (`--bbox_folder`)
- Source video clips: 15 fps basketball clips for task2
  (`--video_base_path`)
- Background-inpainted clips: matching task2 versions
  (`--background_video_folder`)
- Polished per-video captions: `polished_captions_final.json`
  (`--polished_captions`)

## Notes

- Data config on HF: `t8_goal_conditioned_action_generation`
- Default training hyperparameters: 5 epochs, lr 1e-4, save every 2000 steps
- `train.sh` resumes from a step-8000 T7 checkpoint by default — adjust
  `--lora_checkpoint` if your run path differs.
- Output dir: `./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_task2`
## Vendored DiffSynth-Studio slice

`diffsynth/` is a slimmed copy of upstream
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Non-Wan
model families and their registry entries were dropped to keep the slice
under ~1.4 MB. T7 carries an identical copy — keep the two in sync when
re-syncing from upstream. See T7's README for the re-sync checklist.
