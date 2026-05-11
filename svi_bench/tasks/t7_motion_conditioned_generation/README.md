# T7 — Motion-Conditioned Generation

LoRA fine-tune of **Wan2.1-Fun-V1.1-1.3B-Control** conditioned on per-frame
player bounding boxes and a background-video stream. Trains the DiT side of
the pipeline with `q,k,v,o,ffn.0,ffn.2` LoRA targets at rank 32.

## Install

```bash
pip install "svi-bench[t7]"
```

This pulls torch, accelerate, peft, transformers, einops, modelscope,
imageio, pandas, ftfy, and the rest of the deps required by the vendored
DiffSynth-Studio slice at [`../_wan_shared/`](../_wan_shared/).

## Run

The training+validation pipeline lives in `train.sh`. Edit the data paths
at the top before launching:

```bash
bash svi_bench/tasks/t7_motion_conditioned_generation/train.sh
```

Or via the unified CLI:

```bash
svi-bench evaluate --task t7 --model wan2.1-fun
```

## Files

- [`train.sh`](train.sh) — `accelerate launch` wrapper around the shared
  vendored `train.py`; sets `PYTHONPATH` so `from diffsynth import ...`
  resolves to the bundled slice.
- [`validate.py`](validate.py) — periodic-validation hook invoked by
  `train.py` via the `$VALIDATION_SCRIPT` env var. Samples short video
  clips at each save step using the bbox + background-video conditioning.
- [`evaluate.py`](evaluate.py) — thin Python wrapper that just shells out
  to `train.sh`; exposed via `svi-bench evaluate --task t7`.

## Data

- Bbox folder: `train.txt` listing per-clip bbox `.npz` files (set via
  `--bbox_folder` in `train.sh`)
- Source video clips: 15 fps basketball footage (`--video_base_path`)
- Background-inpainted video: matching clips with players masked out
  (`--background_video_folder`)
- Default conditioning prompt: `"a realistic basketball game video"`

## Notes

- Data config on HF: `t7_motion_conditioned_generation`
- Default training hyperparameters: 3 epochs, lr 1e-4, save every 2000 steps
- Output dir: `./models/train/Wan2.1-Fun-V1.1-1.3B-Control-lora_with_bboxs_color_background_81frames_full_scale`
- See [`../_wan_shared/README.md`](../_wan_shared/README.md) for what was
  vendored from upstream DiffSynth-Studio and how to re-sync.
