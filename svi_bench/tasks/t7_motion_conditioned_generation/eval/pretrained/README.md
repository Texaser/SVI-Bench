# T7 eval/ checkpoints

Model checkpoints used by `run_basketball.sh` and `run_soccer.sh`. **Files
in this directory are gitignored** — each machine populates them locally.

## Required files

| File | Size | Used by | Notes |
|---|---|---|---|
| `yolox_x_sports_train.pth.tar` | 757 MB | YOLOX-X sports MOT detector | Loaded via the `--ckpt` arg of `eval_generated_videos.py`. Same file works for basketball and soccer; the exp config (`yolox_x_sportsmot.py` vs `yolox_x_soccernet.py`) selects per-domain wiring, not weights. |
| `MixFormer_sports_train.pth.tar` | 376 MB | MixFormer-ViT tracker | Loaded automatically by `mixformer_deit.py` via `track.yaml`'s `MODEL.BACKBONE.PRETRAINED_PATH = pretrained/MixFormer_sports_train.pth.tar`. The shell wrapper `cd`s into this `eval/` dir before launching python so that relative path resolves here. |

Both checkpoints are required for the tracker to load — missing either
crashes startup.

T8 ships an independent copy of these files under its own
`eval/pretrained/` (per-task isolation).

## How to populate

If you have access to the original upstream MixSort working directory:

```bash
cp /path/to/MixSort/pretrained/yolox_x_sports_train.pth.tar  ./
cp /path/to/MixSort/pretrained/MixFormer_sports_train.pth.tar ./
```

Otherwise grab them from the upstream MixSort release
(<https://github.com/MCG-NJU/MixSort>).
