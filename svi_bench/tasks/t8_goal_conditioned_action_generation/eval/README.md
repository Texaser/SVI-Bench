# T8 Evaluation — Last-Frame mIoU + SigLIP2 Feature Similarity

End-to-end scoring pipeline for the task2 basketball videos produced by
`inference/basketball.sh`:

1. Track every generated clip with **MixSort** (YOLOX-sportsmot detector
   + MixFormer-ViT tracker) to extract per-frame player bboxes.
2. Compute **last-frame mIoU** between the tracker's last-frame
   predictions and the `end_bbox` for each target player specified in
   `polished_captions_final.json`.
3. Compute **SigLIP2 last-frame IoU-gated feature similarity** for the
   same target players at frame 80 (matched pred-vs-GT crops).

T8 only covers basketball (the task2 variant of T7's training flow), so
there is only one wrapper here. T7 ships an identical-purpose `eval/` with
both basketball and soccer scripts plus the holistic-video-mIoU variant.

## Run

### 1. Tracker + last-frame mIoU

```bash
# 8 GPUs by default. Override via NUM_GPUS=4 etc.
bash eval/run_basketball.sh \
    /path/to/generated_videos_dir \
    /path/to/test_task2_final_1000.txt \
    /path/to/polished_captions_final.json \
    /path/to/yolox_x_sports_train.pth.tar
```

### 2. SigLIP2 last-frame feature similarity (after step 1)

Re-uses the tracker output from step 1 (`${VIDEO_DIR}/eval_results`).

```bash
bash eval/run_basketball_featsim.sh \
    /path/to/generated_videos_dir \
    /path/to/test_task2_final_1000.txt \
    /path/to/polished_captions_final.json
```

Results land at `${VIDEO_DIR}/feature_sim_task2/{summary.json, per_clip_metrics.csv}`.

If your generated videos are under DiffSynth's per-clip layout
(`validation/step-N/<clip>/generated.mp4`), point `VALIDATION_DIR` env
var at it and the wrapper will auto-flatten symlinks:

```bash
VALIDATION_DIR=.../validation/step-16000 bash eval/run_basketball.sh
```

Results land at `${VIDEO_DIR}/video_miou_task2_results/{summary.json,per_video_metrics.csv}`.

## Files

- `run_basketball.sh` — tracker + mIoU orchestration wrapper.
- `run_basketball_featsim.sh` — SigLIP2 last-frame feature similarity
  wrapper (consumes tracker output from `run_basketball.sh`).
- `eval_generated_videos.py`, `miou_metric.py`, `yolox/`, `MixViT/`,
  `exps/` — identical role to T7's copies (per-task isolation duplicates
  the tracker stack).
- `video_miou_task2.py` — last-frame mIoU scorer.
- `feature_sim.py` — SigLIP2-only last-frame IoU-gated feature similarity
  scorer (DINOv3 removed).

## Required external assets

Same as T7 — see [T7's `eval/README.md`](../../t7_motion_conditioned_generation/eval/README.md#required-external-assets)
for the checkpoint / data list. Also requires
`polished_captions_final.json` (per-clip player target end_bboxes).

## Slim provenance

Same as T7's slice. The only differences in T8's `eval/`:
- ships `video_miou_task2.py` instead of `video_miou.py`
- only `exps/example/mot/yolox_x_sportsmot.py` (no soccernet variant)
- only `run_basketball.sh` (no soccer)

If upstream MixSort changes, re-apply the same slim recipe to both task
copies. See T7's `eval/README.md` for the full provenance checklist.
