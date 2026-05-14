# T7 — Motion-Conditioned Generation

![Pillar 3 figure: T7 (top) and T8 (bottom)](../../../docs/figures/pillar3.png)

_The top row of the figure illustrates T7: an initial frame plus the
player-removed background video with per-player bbox motion trajectories
become the conditioning inputs; the target output is a 5–10 s video where
players follow those trajectories. Vector source:
[`docs/figures/pillar3.pdf`](../../../docs/figures/pillar3.pdf)._

Part of SVI-Bench **Pillar 3: Strategic Simulation**, which evaluates whether
video generation models can simulate alternative futures while respecting the
physical constraints of real multi-agent play.

The implementation in this directory is a LoRA fine-tune of
**Wan2.1-Fun-V1.1-1.3B-Control** conditioned on per-frame player bounding
boxes and a background-video stream. It trains the DiT side of the pipeline
with `q,k,v,o,ffn.0,ffn.2` LoRA targets at rank 32.

## Task

Given:

- an **initial frame** showing all players in their starting positions,
- a **player-removed background video** — the original footage with all
  players digitally erased via video inpainting, leaving only the court or
  pitch and static elements, and
- a set of **player motion trajectories** specified as time-aligned
  bounding-box sequences,

the model must generate a 5–10 s video in which players follow the
prescribed trajectories while remaining visually, physically, and temporally
coherent.

Unlike prior trajectory-conditioned generation that typically handles one or
two objects in simple scenes, T7 targets multi-agent coordination where 10+
players move simultaneously, interact physically, and occlude one another.

## Data construction

Each instance consists of:

1. an initial frame,
2. per-player motion trajectories as bounding-box sequences, and
3. a player-removed background video generated via video inpainting.

Explicit quality filtering removes instances with unstable tracking, severe
occlusion, or visible inpainting artifacts (residual player silhouettes,
texture bleeding) so generation operates on clean background inputs.

## Evaluation metrics

Two metrics specified in the paper:

- **Video mIoU** — spatiotemporal alignment between player trajectories in
  the generated and reference videos. **Implementation bundled** at
  [`eval/`](eval/) (slim copy of `MixSort`: YOLOX detector + MixFormer-ViT
  tracker + holistic-video-mIoU). Run via
  `bash eval/run_basketball.sh` or `bash eval/run_soccer.sh` after the
  generation step.
- **Temporal feature similarity** — SigLIP features from corresponding
  player regions across frames, measuring visual consistency.
  _Not yet bundled_ — plug in your own pass over the generated videos.

## Install

```bash
pip install "svi-bench[t7]"
```

This pulls torch, accelerate, peft, transformers, einops, modelscope,
imageio, pandas, ftfy, and the rest of the deps required by the vendored
DiffSynth-Studio slice bundled inside this task at [`diffsynth/`](diffsynth/).

## Run

### Train

Edit the data paths at the top of `train.sh` first, then launch:

```bash
bash svi_bench/tasks/t7_motion_conditioned_generation/train.sh
```

### Inference

T7 covers two domains. Each loads the latest `step-*.safetensors`
checkpoint under the LoRA output dir and generates video samples for every
clip in the corresponding test set, sharded across GPUs:

```bash
# Basketball (default 8 GPUs)
bash svi_bench/tasks/t7_motion_conditioned_generation/inference/basketball.sh

# Soccer (default 4 GPUs)
bash svi_bench/tasks/t7_motion_conditioned_generation/inference/soccer.sh
```

You can override the output directory by passing it as `$1`. Edit the
`TEST_SUBSET` / `VALIDATION_VIDEO_BASE` / `VALIDATION_BACKGROUND_VIDEO_BASE`
lines inside each script to point at your data.

The unified CLI dispatches to `inference/basketball.sh` by default and
accepts `domain=soccer` via the config:

```bash
svi-bench evaluate --task t7 --model wan2.1-fun
```

## Files

- [`train.sh`](train.sh) — `accelerate launch` wrapper around the bundled
  `train.py`; sets `PYTHONPATH` so `from diffsynth import ...` resolves
  to [`diffsynth/`](diffsynth/).
- [`train.py`](train.py) — training entry point, vendored verbatim from
  `DiffSynth-Studio/examples/wanvideo/model_training/train.py`.
- [`validate.py`](validate.py) — **in-training** validation hook invoked
  by `train.py` via the `$VALIDATION_SCRIPT` env var. Samples a small
  number of video clips at each save step as a sanity check.
- [`inference/`](inference/) — multi-GPU inference pipeline that loads
  the trained LoRA and generates video samples:
  - `basketball.{sh,py}` — full basketball test-set run, default 8 GPUs.
  - `soccer.{sh,py}` — full soccer test-set run, default 4 GPUs.
  - `split_validation_set.py` — helper that shards a test-set listing
    into N per-GPU split files.
- [`diffsynth/`](diffsynth/) — slimmed copy of the Wan2.1-Fun-related
  closure from upstream DiffSynth-Studio. T8 ships an identical copy.
- [`evaluate.py`](evaluate.py) — Python wrapper exposed via
  `svi-bench evaluate --task t7`. Dispatches to `inference/<domain>.sh`.

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
