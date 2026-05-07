# SVI-Bench

A multi-task benchmark aggregating tasks T1–T9 into a single installable package
with per-task dependency isolation.

## Quick start

Install only the task(s) you care about:

```bash
pip install "svi-bench[t3]"          # just T3's dependencies
pip install "svi-bench[t3,t7]"       # T3 + T7
pip install "svi-bench[all]"         # everything
```

Run an evaluation:

```bash
# Evaluate GPT-4o on T3 only
svi-bench evaluate --task t3 --model gpt-4o

# Evaluate all available tasks
svi-bench evaluate --task all --model gpt-4o
```

## Data access

Datasets are hosted on HuggingFace at
[`svi-bench/svi-bench`](https://huggingface.co/datasets/svi-bench/svi-bench)
with one config per task. Access is gated — you'll need to agree to the
non-commercial / no-redistribution terms on the dataset page once, after which
your HF token unlocks all configs:

```python
from datasets import load_dataset

# Only T3 is downloaded
ds = load_dataset("svi-bench/svi-bench", "t3_action_recognition")

# Or download everything
ds = load_dataset("svi-bench/svi-bench", "all")
```

A convenience CLI wraps this for users who don't want the HF API:

```bash
svi-bench download --tasks t3 t7
```

The dataset card template (gated-access YAML header) lives at
[`dataset_card.md`](dataset_card.md) — copy it into the HF dataset repo's
`README.md` when publishing.

## Repository layout

```
svi-bench/
├── pyproject.toml            # single package, optional deps per task
├── svi_bench/
│   ├── core/                 # shared utilities (data, metrics, models, config)
│   ├── tasks/                # one subpackage per task, lazy-imported
│   │   ├── t1_scene_recognition/
│   │   ├── t3_action_recognition/
│   │   ├── t7_deep_game_analysis/
│   │   └── ...
│   └── cli.py                # unified entry point
├── configs/                  # YAML hyperparameters per task
└── scripts/                  # helper scripts (download, etc.)
```

Each `tasks/t<N>_*/` directory has its own `README.md` covering setup,
expected results, and contributor notes.

## Adding a new task

1. Create `svi_bench/tasks/t<N>_<slug>/` with `evaluate.py`, `baseline.py`,
   and a `README.md`.
2. Add an optional-dependency group to `pyproject.toml` listing only the deps
   *that task* needs.
3. Use **lazy imports** inside functions for any heavy dep — the package must
   import cleanly even when those deps are absent.
4. Register the task in `svi_bench/tasks/__init__.py`.
5. Add a YAML config under `configs/t<N>.yaml`.
6. Upload data to the HF dataset repo as a new config.

## License

Code is MIT (see [`LICENSE`](LICENSE)). Data is governed by the gated-access
agreement on the HF dataset page.
