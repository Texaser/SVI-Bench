# `_wan_shared/` — vendored DiffSynth-Studio slice

T7 (Motion-Conditioned Generation) and T8 (Goal-Conditioned Action
Generation) are both LoRA fine-tunes of **Wan2.1-Fun-V1.1-1.3B-Control** with
bounding-box + background-video conditioning. They share the same training
entry point and the same upstream `diffsynth` package — only the shell args
and per-task validation script differ. To keep T7 and T8 functionally
identical to the original upstream flow without forcing users to clone the
~10 GB `DiffSynth-Studio` repo, the minimum Wan-related closure is vendored
here.

## What's in here

- `diffsynth/` — a slimmed copy of the upstream package. Non-Wan model
  families (SD / SDXL / SD3 / Flux / HunyuanDiT / HunyuanVideo / SVD / Cog /
  OmniGen / StepVideo / QwenImage) were removed along with their entries in
  `configs/model_config.py` and `models/model_manager.py`. The remaining
  surface area is the Wan2.1 pipeline, its model components, the training
  utilities, and the bbox/background-video dataset loaders.
- `train.py` — shared `accelerate launch` entry point. Copied verbatim from
  `DiffSynth-Studio/examples/wanvideo/model_training/train.py`.
- `__init__.py` — wires the vendored `diffsynth/` onto `sys.path` so the
  unmodified `from diffsynth import ...` lines keep working.

## Upstream provenance

The slice was extracted at:

- Source repo: `DiffSynth-Studio` (https://github.com/modelscope/DiffSynth-Studio)
- Sport-specific additions (bbox folder loading, polished captions,
  `--bbox_color_mode`, `--use_overlay_method`, `--bbox_first_last_only`,
  `--background_video_folder`) live in `diffsynth/trainers/utils.py` and
  `diffsynth/trainers/unified_dataset.py` and are not part of upstream
  DiffSynth-Studio.

## When to re-sync

If upstream DiffSynth-Studio adds capabilities to the Wan2.1 pipeline that
T7/T8 should adopt, re-copy:

1. `diffsynth/pipelines/wan_video_new.py`
2. Any new `diffsynth/models/wan_video_*.py` it imports
3. `diffsynth/trainers/{utils,unified_dataset}.py` (preserving the sport
   extensions — these are local additions, not upstream)

Then re-run the slimming checklist:
- Remove non-Wan imports from `diffsynth/models/model_manager.py`
- Remove non-Wan entries from `diffsynth/configs/model_config.py`
- Stub any new non-Wan classes referenced by `diffsynth/models/lora.py`
- `python -c "from diffsynth.pipelines.wan_video_new import WanVideoPipeline"`
  must succeed against the vendored slice alone.
