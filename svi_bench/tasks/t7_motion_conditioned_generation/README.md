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
DiffSynth-Studio slice bundled inside this task at [`diffsynth/`](diffsynth/).

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

- [`train.sh`](train.sh) — `accelerate launch` wrapper around the bundled
  `train.py`; sets `PYTHONPATH` so `from diffsynth import ...` resolves
  to [`diffsynth/`](diffsynth/).
- [`train.py`](train.py) — training entry point, vendored verbatim from
  `DiffSynth-Studio/examples/wanvideo/model_training/train.py`.
- [`diffsynth/`](diffsynth/) — slimmed copy of the Wan2.1-Fun-related
  closure from upstream DiffSynth-Studio. T8 ships an identical copy.
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
## Vendored DiffSynth-Studio slice

`diffsynth/` is a slimmed copy of upstream
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Non-Wan
model families (SD / SDXL / SD3 / Flux / HunyuanDiT / HunyuanVideo / SVD /
Cog / OmniGen / StepVideo / QwenImage) and their registry entries were
dropped so the slice fits in ~1.4 MB. T8 carries an identical copy.

If upstream adds capabilities the slice should adopt, re-copy:
1. `diffsynth/pipelines/wan_video_new.py`
2. Any new `diffsynth/models/wan_video_*.py` it imports
3. `diffsynth/trainers/{utils,unified_dataset}.py` (preserving the
   bbox / polished-caption / overlay extensions — those are local
   additions, not upstream)

Then re-run the slimming checklist:
- Remove non-Wan imports from `diffsynth/models/model_manager.py`
- Trim non-Wan rows from `diffsynth/configs/model_config.py`
- Stub any new non-Wan classes referenced by `diffsynth/models/lora.py`
- `python -c "from diffsynth.pipelines.wan_video_new import WanVideoPipeline"`
  must succeed using the local slice alone.

Apply the same updates to T8's copy.
