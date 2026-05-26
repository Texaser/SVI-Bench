# T3 — Compositional Video Retrieval

Given a natural-language query specifying entity / dynamics / context /
spatial attributes, rank the correct clip among 1 positive + 5,000
same-sport negatives. Metrics: R@K and median rank.

---

## 1. Install

```bash
conda create -n svi-bench-t3 python=3.11 -y && conda activate svi-bench-t3
pip install -e ".[t3]"     # from the SVI-Bench/ repo root
```

---

## 2. Data

```bash
huggingface-cli download MVP-Group/SVI-Bench --repo-type dataset \
    --include "T3/**" --local-dir data/
```

Video clips are shipped as `.tar` bundles. After downloading, extract them:

```bash
python3 scripts/extract_tars.py --root data/T3/clips
```

Everything goes under `<repo>/data/T3/`:

```
data/T3/
├── clips/{sport}/{bucket}/*.mp4        # extracted from tars
├── data/{train,val,test}/*.json
├── compositions/{*.json, mappings/*.json}
├── embeds/embeds_{val|test}_{sport}_{full|partial}.pt
└── ckpts/
    ├── internvideo2_1b_sports_{full,partial}.pth   # for evaluation
    └── InternVideo2-stage2_1b-224p-f4.pt           # for training (§4)
```

---

## 3. Evaluate

Two steps: 1. **extract embeddings** 2. **run retrieval**

### 3.1 Extract embeddings

For the provided baselines, embeddings are already at
`data/T3/embeds/embeds_{val|test}_{sport}_{full|partial}.pt` from
`svi-bench download`. Skip to 3.2.

For your own checkpoint, point the eval scripts at it and choose a short
tag for the output filename:

```bash
cd svi_bench/tasks/t3_compositional_video_retrieval/internvideo2

export EVAL_CKPT=$T3_ROOT/results/finetune_full_caption/ckpt_latest.pth
export EVAL_SUFFIX=myrun     # any short tag

for sport in basketball hockey soccer; do
  for split in val test; do
    bash scripts/finetuning/1B/eval_full_caption_${split}_${sport}.sh
  done
done
```

Embeddings are written to `data/T3/embeds/embeds_{split}_{sport}_${EVAL_SUFFIX}.pt`.

### 3.2 Run retrieval

```bash
svi-bench evaluate --task t3 --model <model-id>
```

Built-in model ids:

- `internvideo2-1b-sports-full`
- `internvideo2-1b-sports-partial`
- `all`  — runs every built-in model

For your own checkpoint, add one line to `KNOWN_MODELS` in
[`evaluate.py`](evaluate.py):

```python
"my-model": {"suffix": "myrun"},     # match the EVAL_SUFFIX from 3.1
```

Then:

```bash
svi-bench evaluate --task t3 --model my-model
```

---

## 4. Train

```bash
cd svi_bench/tasks/t3_compositional_video_retrieval/internvideo2
bash scripts/finetuning/1B/finetune_full_caption.sh
bash scripts/finetuning/1B/finetune_attribute_dropout.sh
```

Both scripts run on SLURM with 8 GPUs on one node. Per-epoch checkpoints
and `train.log` are written under `data/T3/results/finetune_<regime>/`.

Per-epoch evaluation is off by default. To enable val-set R@K after every
epoch (across all three sports), set `EVAL_DURING_TRAINING=1`.

Your fine-tuned checkpoint is at
`data/T3/results/finetune_<regime>/ckpt_latest.pth` when `save_latest=True`,
otherwise `ckpt_NN.pth` per epoch. To evaluate it, follow §3.
