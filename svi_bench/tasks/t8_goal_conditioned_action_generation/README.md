# T8 — Goal-Conditioned Action Generation

![Pillar 3 figure: T7 (top) and T8 (bottom)](../../../docs/figures/pillar3.png)

_The bottom row of the figure illustrates T8: an initial frame plus the
player-removed background video annotated with the spatial target (red
bbox at the goal location) and a goal instruction (e.g. "Simulate the
player in the blue box attempting and missing a shot from the location
marked by the red box") become the conditioning inputs; the target output
is a 5–10 s video where the specified player executes the action. Vector
source: [`docs/figures/pillar3.pdf`](../../../docs/figures/pillar3.pdf)._

Part of SVI-Bench **Pillar 3: Strategic Simulation**, sibling task to T7.
Where T7 prescribes exact trajectories, T8 forces the model to **plan**
intermediate actions toward a high-level goal under explicit spatial
constraints.

The implementation in this directory is a LoRA fine-tune of
**Wan2.1-Fun-V1.1-1.3B-Control** that builds on T7 but swaps the static
prompt for **per-video polished captions** and switches the bbox
conditioning to **first/last-frame only** — i.e. the model has to fill in
the in-between motion that takes the players from their start positions to
their end positions. Typically chained off a T7 checkpoint via
`--lora_checkpoint`.

## Task

Given:

- an **initial frame**,
- a **player-removed background video** (same construction as T7), and
- a **textual instruction** specifying target player(s), spatial constraints
  (start and end bounding boxes), and a desired action outcome — e.g. a
  rebound, a contested layup,

the model must generate a 5–10 s video in which the specified players
execute a coherent action sequence that achieves the described objective.

Unlike T7's trajectory-following, T8 requires implicit understanding of
environment dynamics and goal-directed reasoning that goes beyond
open-ended text-conditioned generation.

## Data construction

Curated basketball video clips paired with structured goal specifications
derived from annotated actions, covering diverse goal-conditioned behaviors:
completing plays at designated locations, executing specific moves, and
interaction-aware scenarios. Per-video prompts are stored in
`polished_captions_final.json` (one entry per clip); start/end bbox
constraints come from the same bbox listings used by T7.

## Evaluation metrics

Three metrics specified in the paper:

- **Final-frame mIoU** — bounding-box overlap between generated and target
  player positions at the last frame. **Implementation bundled** at
  [`eval/`](eval/) (slim copy of `MixSort` + last-frame matcher). Run via
  `bash eval/run_basketball.sh` after the generation step.
- **Final-frame feature similarity** — visual fidelity of the realized
  outcome. **Implementation bundled** at [`eval/feature_sim.py`](eval/feature_sim.py)
  (last-frame IoU-gated SigLIP2 cosine sim). Run after the mIoU
  pipeline via `bash eval/run_basketball_featsim.sh <VIDEO_DIR>`.
- **Goal accuracy** — fraction judged successful by a fine-tuned
  video-language QA model that asks whether the generated video achieves
  the specified objective. **Implementation bundled** at
  [`eval/test_llavaov.py`](eval/test_llavaov.py) + the vendored
  [`eval/llava/`](eval/llava/) package. Run after the mIoU pipeline via
  `bash eval/run_basketball_goalacc.sh <VIDEO_DIR> <QA_SOURCE> <MODEL_PATH>`.
  Requires a fine-tuned LLaVA-Qwen checkpoint (~15 GB, user-supplied via
  `MODEL_PATH`).

## Install

```bash
pip install "svi-bench[t8]"
```

A single `[t8]` env covers **train + inference + all three evals**
(mIoU, feature similarity, goal accuracy). It bundles torch, accelerate,
peft, transformers, einops, modelscope, imageio, plus the eval-only
extras (yolox is vendored under `eval/`, LLaVA is vendored under
`eval/llava/`, and `decord` / `pycocotools` are pulled in via `[t8]`).

`eval/llava_requirements.txt` is kept around as a reference for the
exact pin set that the LLaVA-Qwen authors developed against. You only
need to fall back to it if the unified env causes loader / attention
errors; in practice `flash-attn` is the one optional install you may
want on top (it's CUDA-version-sensitive, so it's not in `[t8]`).

## Run

### Checkpoints

Pre-trained T8 LoRA checkpoint (basketball, ~84 MB) is published on the HF
dataset
[`MVP-Group/SVI-Bench`](https://huggingface.co/datasets/MVP-Group/SVI-Bench/tree/main/T8).
Download into the task directory before running inference:

```bash
cd svi_bench/tasks/t8_goal_conditioned_action_generation
bash download_checkpoint.sh   # → checkpoints/T8/basketball/checkpoint.safetensors
```

It's a LoRA adapter (rank 32) for `Wan2.1-Fun-V1.1-1.3B-Control`. Load via
`--lora_checkpoint <path>` or pass the path as `argv[1]` to
`inference/infer.py`.

### Download data

T8's basketball clips / bboxes / inpainted backgrounds / splits / per-video
captions are hosted on
[`MVP-Group/SVI-Bench`](https://huggingface.co/datasets/MVP-Group/SVI-Bench/tree/main/T8).
Run the helper from the repo root to fetch and extract everything into
`./data/T8/` (or set `SVI_BENCH_DATA` first to use a different location):

```bash
bash scripts/download_t7_t8.sh
```

After the download, the layout is:

```
$SVI_BENCH_DATA/T8/basketball/
├── clips/{00..99}/{ID}.mp4               # original 5 s game clips
├── bboxes/{00..99}/{ID}.txt              # per-frame player bboxes
├── backgrounds/{00..99}/{ID}.mp4         # player-removed inpainted backgrounds
├── splits/{train,val,test}_task2_final.txt    # one sample ID per line
└── captions.json                          # id -> refined_instruction + player_specifications
```

`captions.json` is keyed by sample ID and contains only the two fields the
trainer actually consumes (`refined_instruction`, `player_specifications`).
`train.sh` / `inference/infer.sh` read this layout by default (via
`SVI_BENCH_DATA`); no path editing required.

### Train

```bash
bash svi_bench/tasks/t8_goal_conditioned_action_generation/train.sh
```

### Inference

T8 only covers basketball. Loads the latest `step-*.safetensors`
checkpoint under the LoRA output dir and generates video samples for the
task2 basketball test set, sharded across 8 GPUs by default:

```bash
bash svi_bench/tasks/t8_goal_conditioned_action_generation/inference/infer.sh
```

Pass a different output dir as `$1` to override the default. To use a
different data root, export `SVI_BENCH_DATA=/path/to/dir` before running.

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
  - `infer.{sh,py}` — full task2 basketball test-set run, default 8 GPUs.
  - `split_validation_set.py` — helper that shards a test-set listing
    into N per-GPU split files.
- [`diffsynth/`](diffsynth/) — slimmed copy of the Wan2.1-Fun-related
  closure from upstream DiffSynth-Studio. T7 ships an identical copy.
- [`evaluate.py`](evaluate.py) — Python wrapper exposed via
  `svi-bench evaluate --task t8`.

## Data

Downloaded via `scripts/download_t7_t8.sh` from
[`MVP-Group/SVI-Bench`](https://huggingface.co/datasets/MVP-Group/SVI-Bench)
into `$SVI_BENCH_DATA/T8/basketball/`.

Each sample is identified by a zero-padded numeric ID (e.g. `0000000`).
For each ID there are four artifacts:

- `clips/{bucket}/{ID}.mp4` — original 5 s basketball clip (832×480, 15 fps)
- `bboxes/{bucket}/{ID}.txt` — per-frame player bbox annotations
- `backgrounds/{bucket}/{ID}.mp4` — player-removed inpainted background
- `captions.json[{ID}]` — `{refined_instruction, player_specifications}`

`bucket` is the first two digits of `ID // 741` — samples are sharded into
100 directories of ≤741 files each to stay under HF per-folder limits.

Splits live at `splits/{train,val,test}_task2_final.txt` (one ID per line),
plus `test_task2_final_{100,1000}.txt` for the small/medium eval subsets.
`scripts/build_split_bbox_list.py` converts these into the full-path bbox
lists the dataset loader expects; `train.sh` / `inference/infer.sh`
invoke it automatically the first time.

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
