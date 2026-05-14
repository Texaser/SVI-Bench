# T7 Evaluation — Video mIoU

End-to-end scoring pipeline for the videos produced by `inference/*.sh`:

1. Track every generated clip with the **MixSort** (YOLOX-sportsmot detector
   + MixFormer-ViT tracker) to extract per-frame player bboxes.
2. Compute **Holistic Video mIoU** by accumulating `intersection / union`
   across all frames and matched track pairs against ground-truth bboxes.

Both pieces are vendored from
[`MixSort`](https://github.com/MCG-NJU/MixSort) and slimmed down to the
inference path only (no training code, no non-sports tracker variants).

## Run

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

## Files

- `run_basketball.sh` / `run_soccer.sh` — orchestration wrappers
  (multi-GPU shard → tracker → mIoU).
- `eval_generated_videos.py` — per-GPU worker: loads YOLOX detector,
  runs MixSort tracker on each video in the shard, writes per-clip bbox
  output.
- `video_miou.py` — final scorer: ingests tracker output + GT bbox files,
  emits Holistic Video mIoU.
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
