# T8 Evaluation — Last-Frame mIoU + SigLIP2 Feature Similarity + LLaVA Goal Accuracy

End-to-end scoring pipeline for the task2 basketball videos produced by
`inference/infer.sh`:

1. Track every generated clip with **MixSort** (YOLOX-sportsmot detector
   + MixFormer-ViT tracker) to extract per-frame player bboxes.
2. Compute **last-frame mIoU** between the tracker's last-frame
   predictions and the `end_bbox` for each target player specified in
   `captions.json`.
3. Compute **SigLIP2 last-frame IoU-gated feature similarity** for the
   same target players at frame 80 (matched pred-vs-GT crops).
4. Compute **LLaVA-Qwen goal accuracy** — run a fine-tuned video-language
   QA model on each generated clip and check whether the rendered Q/A
   pairs (one per QA-type) get answered correctly. Reports per-QA-type
   and aggregate accuracy.

T8 only covers basketball (the goal-conditioned variant of T7's training flow),
so there is only one wrapper here. T7 ships an identical-purpose `eval/` with
both basketball and soccer scripts plus the holistic-video-mIoU variant.

All three wrappers below default `GT_LIST` and `CAPTIONS` to the
HF-downloaded layout under `$SVI_BENCH_DATA/T8/basketball/`, so the common
case is just to pass `VIDEO_DIR`. Override positional args 2/3 if your
layout differs.

## Run

### 1. Tracker + last-frame mIoU

```bash
# 8 GPUs by default. Override via NUM_GPUS=4 etc.
bash eval/run_basketball.sh /path/to/generated_videos_dir
```

### 2. SigLIP2 last-frame feature similarity (after step 1)

Re-uses the tracker output from step 1 (`${VIDEO_DIR}/eval_results`).

```bash
bash eval/run_basketball_featsim.sh /path/to/generated_videos_dir
```

Results land at `${VIDEO_DIR}/feature_sim/{summary.json, per_clip_metrics.csv}`.

### 3. LLaVA goal accuracy

Runs in the **same `[t8]` env** as steps 1-2 — `decord` and
`pycocotools` (the only LLaVA-only runtime deps) are pulled in by the
`[t8]` extras. `flash-attn` is optional: install it manually if your
CUDA stack matches `flash-attn==2.5.7`; otherwise the vendored
LLaVA-Qwen loader falls back to eager attention. The full pinned set
from upstream LLaVA-NeXT is preserved in `llava_requirements.txt` for
reference only.

The HF dataset ships two pieces under `T8/`:

- `T8/llava_qa_checkpoint/` — fine-tuned LLaVA-Qwen QA model (~15 GB).
- `T8/basketball/qa_test/` — anonymized **master** QA bank
  (8 question types, 8720 QA pairs covering 4994/5000 test clips).
  Each entry carries `id`, the original `start_bbox`, ground-truth answer,
  and the question text — but **no rendered videos**: the per-method
  bbox-overlay videos are produced on the fly from your generated outputs.

`scripts/download_t7_t8.sh` pulls both into `$SVI_BENCH_DATA`.

The wrapper auto-runs [`prepare_qa_for_method.py`](prepare_qa_for_method.py)
on first invocation for a given `VIDEO_DIR`: it filters the master QA to
entries whose clip exists in your generated outputs, ffmpeg-renders a red
bbox overlay on frame 0 of each matched video using `start_bbox`, and writes
the per-method Q*.json into `${VIDEO_DIR}/qa_prepared/qa_json/`. Subsequent
invocations skip prepare if that dir already exists. Common case:

```bash
bash eval/run_basketball_goalacc.sh /path/to/generated_videos_dir
```

Override `QA_MASTER` / `MODEL_PATH` (positional args 2-3 or env vars) if
your layout differs.

`test_llavaov.py` internally fans out across visible GPUs via
`torch.multiprocessing`; the wrapper just iterates Q*.json files
sequentially. Results land at
`${VIDEO_DIR}/goal_accuracy_results/<qa_type>/qa_eval_f16_results.json`,
with a summary table printed at the end.

If your generated videos are under DiffSynth's per-clip layout
(`validation/step-N/<clip>/generated.mp4`), point `VALIDATION_DIR` env
var at it and the wrapper will auto-flatten symlinks:

```bash
VALIDATION_DIR=.../validation/step-16000 bash eval/run_basketball.sh
```

Results land at `${VIDEO_DIR}/video_miou_results/{summary.json,per_video_metrics.csv}`.

## Files

- `run_basketball.sh` — tracker + mIoU orchestration wrapper.
- `run_basketball_featsim.sh` — SigLIP2 last-frame feature similarity
  wrapper (consumes tracker output from `run_basketball.sh`).
- `eval_generated_videos.py`, `miou_metric.py`, `yolox/`, `MixViT/`,
  `exps/` — identical role to T7's copies (per-task isolation duplicates
  the tracker stack).
- `video_miou.py` — last-frame mIoU scorer (T7's same-named file
  computes holistic across-frame mIoU instead; the implementation
  differs by metric definition).
- `feature_sim.py` — SigLIP2-only last-frame IoU-gated feature similarity
  scorer (DINOv3 removed).
- `run_basketball_goalacc.sh` — LLaVA-Qwen QA orchestration wrapper. Calls
  `prepare_qa_for_method.py` on first run, then iterates the rendered
  `Q*.json` files through `test_llavaov.py`.
- `prepare_qa_for_method.py` — filters the anonymized master QA to entries
  whose clip exists in `VIDEO_DIR`, renders red-bbox overlays on frame 0
  of each matched video (using the master's `start_bbox`), and writes the
  per-method Q*.json that `test_llavaov.py` consumes.
- `test_llavaov.py` — LLaVA-Qwen QA worker. Multi-GPU via internal
  `torch.multiprocessing`.
- `llava/` — slimmed copy of the LLaVA-NeXT package (model loader,
  vision tower, conversation templates, mm utils, constants). Upstream
  `train/` / `serve/` / `eval/` subpackages are dropped since they're
  not used at inference.
- `llava_requirements.txt` — upstream LLaVA-NeXT's pinned dep set, kept
  as a reference only. The `[t8]` extras in `pyproject.toml` already
  cover the runtime deps the bundled `llava/` slice actually imports
  (`decord`, `pycocotools`, plus the transformers / torch stack shared
  with training).

## Required external assets

Same as T7 — see [T7's `eval/README.md`](../../t7_motion_conditioned_generation/eval/README.md#required-external-assets)
for the checkpoint / data list. Also requires
`captions.json` (per-clip player target end_bboxes).

## Slim provenance

Same as T7's slice. The only differences in T8's `eval/`:
- `video_miou.py` computes last-frame (frame 80) mIoU instead of holistic
  across-frame mIoU (T7's same-named file)
- only `exps/example/mot/yolox_x_sportsmot.py` (no soccernet variant)
- only `run_basketball.sh` (no soccer)

If upstream MixSort changes, re-apply the same slim recipe to both task
copies. See T7's `eval/README.md` for the full provenance checklist.
