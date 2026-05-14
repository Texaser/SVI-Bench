# T7 Evaluation — Video mIoU + SigLIP2 Feature Similarity

End-to-end scoring pipeline for the videos produced by `inference/*.sh`:

1. Track every generated clip with the **MixSort** (YOLOX-sportsmot detector
   + MixFormer-ViT tracker) to extract per-frame player bboxes.
2. Compute **Holistic Video mIoU** by accumulating `intersection / union`
   across all frames and matched track pairs against ground-truth bboxes.
3. Compute **SigLIP2 IoU-gated feature similarity** — for each GT bbox,
   find the best-IoU pred bbox; if matched, crop both and take cosine sim
   of SigLIP2 vision features.

Both pieces are vendored from
[`MixSort`](https://github.com/MCG-NJU/MixSort) and slimmed down to the
inference path only (no training code, no non-sports tracker variants).

## Run

### 1. Tracker + mIoU

```bash
# Basketball (8 GPUs by default, override with NUM_GPUS=4 etc.)
bash eval/run_basketball.sh \
    /path/to/generated_videos_dir \
    /path/to/test_subset.txt \
    /path/to/yolox_x_sports_train.pth.tar

# Soccer
bash eval/run_soccer.sh \
    /path/to/generated_videos_dir \
    /path/to/test_subset_soccer.txt \
    /path/to/yolox_x_sports_train.pth.tar
```

Results land at `${VIDEO_DIR}/video_miou_results/{summary.json,per_video_metrics.csv}`.

### 2. SigLIP2 feature similarity (after step 1)

Re-uses the tracker output from step 1. SigLIP2 model is auto-downloaded
from HF (`google/siglip2-so400m-patch14-384`, ~3 GB on first run).

```bash
# Basketball
bash eval/run_basketball_featsim.sh \
    /path/to/step_dir \
    /path/to/test_subset.txt

# Soccer
bash eval/run_soccer_featsim.sh \
    /path/to/step_dir \
    /path/to/test_subset_soccer.txt
```

`STEP_DIR` is the directory containing both the per-clip subdirs and a
`miou_results_all/` produced by the mIoU pipeline. Results land at
`${STEP_DIR}/feature_sim/{summary.json, per_clip_metrics.csv}`.

Override mode via env: `MODE=both bash eval/run_basketball_featsim.sh ...`
to additionally report the no-tracker baseline ("gt_box" mode).

## Files

- `run_basketball.sh` / `run_soccer.sh` — orchestration wrappers
  (multi-GPU shard → tracker → mIoU).
- `run_basketball_featsim.sh` / `run_soccer_featsim.sh` — orchestration
  wrappers for SigLIP2 feature similarity (assumes the tracker pass
  has already populated `miou_results_all/`).
- `eval_generated_videos.py` — per-GPU worker: loads YOLOX detector,
  runs MixSort tracker on each video in the shard, writes per-clip bbox
  output.
- `video_miou.py` — final scorer: ingests tracker output + GT bbox files,
  emits Holistic Video mIoU.
- `feature_sim.py` — SigLIP2-only IoU-gated feature similarity scorer.
  DINOv3 was removed; SigLIP2 only.
- `miou_metric.py` — shared module providing the `Predictor` class +
  `imageflow_demo_allframes()` runtime (used by `eval_generated_videos`)
  and the helper functions (`load_bbox_file`, `establish_track_id_mapping`,
  `find_first_appearance_frames`) used by `video_miou.py`.
- `yolox/` — slimmed YOLOX (~37 .py, inference path only — detector
  forward, postprocess, MixSort tracker, kalman + matching).
- `MixViT/` — slimmed MixFormer-ViT tracker (~48 .py, only the
  `mixformer_deit` variant the runtime loads).
- `exps/example/mot/yolox_x_sportsmot.py`, `yolox_x_soccernet.py` —
  YOLOX experiment configs; the shell wrapper picks one based on domain.

## Required external assets

Not vendored; the user downloads / supplies separately:

- `yolox_x_sports_train.pth.tar` — YOLOX detector checkpoint
  (pass via the 3rd positional arg of the wrapper).
- MixFormer-ViT checkpoint — `miou_metric.py`'s tracker auto-discovers it
  from the path it expects; see upstream MixSort docs.
- Generated test videos (from `inference/basketball.sh` / `soccer.sh`).
- GT bbox listing (`test_subset.txt`, `test_subset_soccer.txt`).

## Slim provenance

If you re-sync the slice from upstream MixSort:

- `yolox/`: kept `models/`, `exp/`, `utils/`, `mixsort_tracker/`,
  `tracking_utils/`, `data/{__init__,data_augment,dataloading,samplers}`.
  Dropped `byte_tracker/`, `deepsort_tracker/`, `sort_tracker/`,
  `motdt_tracker/`, `ocsort_tracker/`, `mixsort_oc_tracker/`,
  `core/`, `evaluators/`, `layers/`, `data/datasets/`, `data/data_prefetcher.py`,
  `tracking_utils/evaluation.py`, and the precompiled `_C.cpython-*.so`
  (training-only).
- `MixViT/`: kept `lib/{models/{mixformer_deit,losses},config/mixformer_deit,utils,train/{data,admin,base_functions.py}}`
  and `experiments/mixformer_deit/`. Dropped all other mixformer variants
  in `lib/models/mixformer_vit/__init__.py`, the full `experiments/`
  tree except `mixformer_deit/`, `lib/test/`, `lib/train/*` except
  data/admin/base_functions.py, and `external/`, `segmentation/`,
  `tracking/`.
- Misc edits: `yolox/utils/model_utils.py` makes `thop` import lazy
  (FLOP profiling, not needed for inference). `yolox/data/__init__.py`
  drops the imports of removed training-only submodules. `MixViT/lib/models/mixformer/__init__.py`
  rewritten to be empty (only `head.py` + `utils.py` are needed).
- Path rewrites: `../MixViT` → `MixViT` (now sibling, not parent).
  `from tools.miou_metric` → `from miou_metric` (no longer a package).
