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

```bash
bash svi_bench/tasks/t8_goal_conditioned_action_generation/train.sh
```

Or via the unified CLI:

```bash
svi-bench evaluate --task t8 --model wan2.1-fun
```

## Files

- [`train.sh`](train.sh) — `accelerate launch` wrapper around the shared
  vendored `train.py`; passes `--polished_captions` and
  `--bbox_first_last_only`.
- [`validate.py`](validate.py) — task2 validation hook. Loads per-video
  prompts from the polished captions JSON and renders samples using
  first/last bbox conditioning.
- [`evaluate.py`](evaluate.py) — Python wrapper invoked by
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
- See [`../_wan_shared/README.md`](../_wan_shared/README.md) for what was
  vendored from upstream DiffSynth-Studio and how to re-sync.
