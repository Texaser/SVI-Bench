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
  **Implementation bundled** at [`eval/feature_sim.py`](eval/feature_sim.py)
  (IoU-gated mode: tracker output → per-pair SigLIP2 cosine sim). Run
  after the mIoU pipeline via `bash eval/run_basketball_featsim.sh
  <STEP_DIR>` or `bash eval/run_soccer_featsim.sh <STEP_DIR>`.

## Install

```bash
pip install "svi-bench[t7]"
```

This pulls torch, accelerate, peft, transformers, einops, modelscope,
imageio, pandas, ftfy, and the rest of the deps required by the vendored
DiffSynth-Studio slice bundled inside this task at [`diffsynth/`](diffsynth/).

## Run

### Checkpoints

Pre-trained LoRA checkpoints (basketball + soccer, ~84 MB each) are
published on the HF dataset
[`MVP-Group/SVI-Bench`](https://huggingface.co/datasets/MVP-Group/SVI-Bench/tree/main/T7).
Download into the task directory before running inference:

```bash
cd svi_bench/tasks/t7_motion_conditioned_generation
bash download_checkpoint.sh basketball   # → checkpoints/T7/basketball/checkpoint.safetensors
bash download_checkpoint.sh soccer       # → checkpoints/T7/soccer/checkpoint.safetensors
```

Each checkpoint is a LoRA adapter (rank 32) for `Wan2.1-Fun-V1.1-1.3B-Control`;
load it via `--lora_checkpoint <path>` or pass the path as `argv[1]` to
`inference/infer.py`.

### Download data

T7's basketball and soccer videos / bboxes / inpainted backgrounds / splits
are hosted on
[`MVP-Group/SVI-Bench`](https://huggingface.co/datasets/MVP-Group/SVI-Bench/tree/main/T7).
Run the helper from the repo root to fetch and extract everything into
`./data/T7/` (or set `SVI_BENCH_DATA` first to use a different location):

```bash
bash scripts/download_t7_t8.sh
```

After the download, the layout is:

```
$SVI_BENCH_DATA/T7/{soccer,basketball}/
├── clips/{00..99}/{ID}.mp4           # original 5 s game clips (832×480, 15 fps)
├── bboxes/{00..99}/{ID}.txt          # per-frame player bboxes
├── backgrounds/{00..99}/{ID}.mp4     # player-removed inpainted backgrounds
└── splits/{train,val,test}.txt # one sample ID per line
```

`train.sh` and the inference scripts read this layout by default (via
`SVI_BENCH_DATA`); no path editing required.

### Train

```bash
# basketball (default)
bash svi_bench/tasks/t7_motion_conditioned_generation/train.sh
# soccer
SPORT=soccer bash svi_bench/tasks/t7_motion_conditioned_generation/train.sh
```

### Inference

T7 covers two domains. The single `inference/infer.sh` entry loads the
latest `step-*.safetensors` checkpoint under the LoRA output dir and
generates video samples for every clip in the chosen sport's test set,
sharded across 8 GPUs by default. Pick the sport via the `SPORT` env var:

```bash
# Basketball (default)
SPORT=basketball bash svi_bench/tasks/t7_motion_conditioned_generation/inference/infer.sh

# Soccer
SPORT=soccer     bash svi_bench/tasks/t7_motion_conditioned_generation/inference/infer.sh
```

Override the output directory by passing it as `$1`, change the shard
count via `NUM_GPUS=...`, or point at a different data root via
`SVI_BENCH_DATA=/path/to/dir`.

The unified CLI dispatches to `inference/infer.sh` (sport=basketball by default) and
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
  - `infer.{sh,py}` — unified multi-GPU inference (8 GPUs by default for
    both sports); pick the test set via `SPORT={basketball,soccer}`.
  - `split_validation_set.py` — helper that shards a test-set listing
    into N per-GPU split files.
- [`diffsynth/`](diffsynth/) — slimmed copy of the Wan2.1-Fun-related
  closure from upstream DiffSynth-Studio. T8 ships an identical copy.
- [`evaluate.py`](evaluate.py) — Python wrapper exposed via
  `svi-bench evaluate --task t7`. Dispatches to `inference/infer.sh` with
  the chosen sport.

## Data

Downloaded via `scripts/download_t7_t8.sh` from
[`MVP-Group/SVI-Bench`](https://huggingface.co/datasets/MVP-Group/SVI-Bench)
into `$SVI_BENCH_DATA/T7/{soccer,basketball}/`.

Each sample is identified by a zero-padded numeric ID (e.g. `0000000`) that
appears in `splits/` and as the basename of its three artifacts:

- `clips/{bucket}/{ID}.mp4` — original 5 s game clip (832×480, 15 fps)
- `bboxes/{bucket}/{ID}.txt` — per-frame player bbox annotations
- `backgrounds/{bucket}/{ID}.mp4` — player-removed inpainted background

`bucket` is the first two digits of `ID // 1668` (basketball) or
`ID // 1236` (soccer) — i.e. samples are sharded into ≤100 directories of
≤1700 files each to keep the HF repo under per-folder limits.

Splits live at `splits/{train,val,test}.txt` (one ID per line). The
shell helpers in `scripts/build_split_bbox_list.py` convert these into the
full-path bbox lists the dataset loader expects; `train.sh` /
`inference/infer.sh` invokes it automatically the first time.

Default conditioning prompt: `"a realistic basketball game video"` (or
`"a realistic soccer game video"` when `SPORT=soccer`).

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
