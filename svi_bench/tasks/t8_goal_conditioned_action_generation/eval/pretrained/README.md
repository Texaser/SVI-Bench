# T8 eval/ checkpoints

Model checkpoints used by `run_basketball.sh`. **Files in this directory
are gitignored** — each machine populates them locally.

## Required files

| File | Size | Used by | Notes |
|---|---|---|---|
| `yolox_x_sports_train.pth.tar` | 757 MB | YOLOX-X sports MOT detector | Loaded via the `--ckpt` arg of `eval_generated_videos.py`. |
| `MixFormer_sports_train.pth.tar` | 376 MB | MixFormer-ViT tracker | Loaded automatically by `mixformer_deit.py` via `track.yaml`'s `MODEL.BACKBONE.PRETRAINED_PATH = pretrained/MixFormer_sports_train.pth.tar`. The shell wrapper `cd`s into this `eval/` dir before launching python so that relative path resolves here. |

Both required. T7 ships an independent copy under its own
`eval/pretrained/` (per-task isolation).

## How to populate

```bash
cp /path/to/MixSort/pretrained/yolox_x_sports_train.pth.tar  ./
cp /path/to/MixSort/pretrained/MixFormer_sports_train.pth.tar ./
```

Or grab from the upstream MixSort release
(<https://github.com/MCG-NJU/MixSort>).
