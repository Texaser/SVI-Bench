# T7 eval/ checkpoints

Model checkpoints used by `run_basketball.sh` and `run_soccer.sh`. **Files
in this directory are gitignored** — each machine populates them locally.

## Required files

| File | Size | Used by | Notes |
|---|---|---|---|
| `yolox_x_sports_train.pth.tar` | 757 MB | YOLOX-X sports MOT detector | Loaded via the `--ckpt` arg of `eval_generated_videos.py`. Same file works for basketball and soccer; the exp config (`yolox_x_sportsmot.py` vs `yolox_x_soccernet.py`) selects per-domain wiring, not weights. |
| `MixFormer_sports_train.pth.tar` | 376 MB | MixFormer-ViT tracker | Loaded automatically by `mixformer_deit.py` via `track.yaml`'s `MODEL.BACKBONE.PRETRAINED_PATH = pretrained/MixFormer_sports_train.pth.tar`. The shell wrapper `cd`s into this `eval/` dir before launching python so that relative path resolves here. |

Both required. T8 ships an independent copy under its own
`eval/pretrained/` (per-task isolation).

## How to download

The checkpoints come from upstream
[MixSort](https://github.com/MCG-NJU/MixSort#model-zoo). They aren't
hosted with stable direct-download URLs, so populate this directory by
one of:

### Option 1 — Google Drive (manual)

Open the upstream model-zoo folder and download both files into this
directory:

<https://drive.google.com/drive/folders/1pQs1gFC_jG0TlGIUMgf3E0I3OztCvgxI>

You want `yolox_x_sports_train.pth.tar` and `MixFormer_sports_train.pth.tar`.

### Option 2 — Baidu Pan

<https://pan.baidu.com/s/1YAP1zKtx-M_ay6uZINoCHg> (extraction code: `7438`)

### Option 3 — gdown (CLI, if you know the file IDs)

```bash
pip install gdown
# Replace <FILE_ID> with the per-file ID from the Google Drive folder above.
gdown --id <YOLOX_FILE_ID>     -O yolox_x_sports_train.pth.tar
gdown --id <MIXFORMER_FILE_ID> -O MixFormer_sports_train.pth.tar
```

### Option 4 — local copy from another working tree

If you already have a MixSort checkout with the files in place:

```bash
cp /path/to/MixSort/pretrained/yolox_x_sports_train.pth.tar  ./
cp /path/to/MixSort/pretrained/MixFormer_sports_train.pth.tar ./
```

## Verifying

After download, both files should be in this directory:

```bash
ls -lh
# yolox_x_sports_train.pth.tar    757M
# MixFormer_sports_train.pth.tar  376M
```

The tracker startup will fail with "checkpoint not found" if either is
missing.
