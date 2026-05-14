# Pretrained checkpoints

Shared location for model checkpoints used by SVI-Bench tasks. **Files in
this directory are gitignored** (binaries don't belong in `git`), so each
machine must populate it locally.

## Currently required

| File | Size | Used by | Notes |
|---|---|---|---|
| `yolox_x_sports_train.pth.tar` | 757 MB | T7 + T8 `eval/run_*.sh` (MixSort detector) | YOLOX-X detector fine-tuned on sports MOT data. Same file for basketball and soccer — the exp config (`yolox_x_sportsmot.py` vs `yolox_x_soccernet.py`) controls per-domain wiring, not the weights. |

## Possibly required (not yet wired)

| File | Size | Used by | Notes |
|---|---|---|---|
| `MixFormer_sports_train.pth.tar` | 376 MB | MixSort tracker association step | Needed if the MixFormer-ViT tracker path is invoked. The YOLOX detector alone produces per-frame bboxes; MixFormer is the cross-frame matcher. Check whether `miou_metric.py`'s tracker auto-discovers this file at runtime; if you hit a "checkpoint not found" error during inference, drop this file alongside `yolox_x_sports_train.pth.tar`. |

## How to populate

If you have access to the original MixSort working directory:

```bash
cp /path/to/MixSort/pretrained/yolox_x_sports_train.pth.tar ./
# Optionally:
cp /path/to/MixSort/pretrained/MixFormer_sports_train.pth.tar ./
```

Otherwise grab them from the upstream MixSort release (see
<https://github.com/MCG-NJU/MixSort>).

## Per-task referencing

Both T7 and T8 `eval/run_*.sh` resolve this directory via
`$REPO_ROOT/pretrained/` (computed relative to the script location). You
can override per invocation:

```bash
# T7 basketball, custom checkpoint path
bash svi_bench/tasks/t7_motion_conditioned_generation/eval/run_basketball.sh \
    /path/to/generated_videos \
    /path/to/test_subset.txt \
    /custom/path/to/yolox_x_sports_train.pth.tar
```
