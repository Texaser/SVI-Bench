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
  `MODEL_PATH`) and the LLaVA-NeXT conda env (`llava_requirements.txt`
  inside `eval/`).

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
