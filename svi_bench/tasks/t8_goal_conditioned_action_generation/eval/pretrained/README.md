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

## How to download

The checkpoints come from upstream
[MixSort](https://github.com/MCG-NJU/MixSort#model-zoo). They aren't
hosted with stable direct-download URLs, so populate this directory by
one of:

### Option 1 — Google Drive (manual)

<https://drive.google.com/drive/folders/1pQs1gFC_jG0TlGIUMgf3E0I3OztCvgxI>

Pick `yolox_x_sports_train.pth.tar` and `MixFormer_sports_train.pth.tar`.

### Option 2 — Baidu Pan

<https://pan.baidu.com/s/1YAP1zKtx-M_ay6uZINoCHg> (code: `7438`)

### Option 3 — gdown CLI

```bash
pip install gdown
gdown --id <YOLOX_FILE_ID>     -O yolox_x_sports_train.pth.tar
gdown --id <MIXFORMER_FILE_ID> -O MixFormer_sports_train.pth.tar
```

### Option 4 — copy from a local MixSort checkout

```bash
cp /path/to/MixSort/pretrained/yolox_x_sports_train.pth.tar  ./
cp /path/to/MixSort/pretrained/MixFormer_sports_train.pth.tar ./
```

## Verifying

```bash
ls -lh
# yolox_x_sports_train.pth.tar    757M
# MixFormer_sports_train.pth.tar  376M
```

Tracker startup fails if either file is missing.
