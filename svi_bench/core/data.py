"""HuggingFace dataset wrapper for SVI-Bench.

Each task is a named config in a single gated HF repo. Loading a config
downloads only that task's shards.
"""

from __future__ import annotations

from typing import Any

DATASET_ID = "svi-bench/svi-bench"


def load_task(
    task: str,
    split: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
):
    """Load a single task's data from the SVI-Bench HF dataset.

    Streaming defers download until iteration — useful for large video shards.
    """
    from datasets import load_dataset

    return load_dataset(DATASET_ID, task, split=split, streaming=streaming, **kwargs)


def load_shared(split: str | None = None, **kwargs: Any):
    """Load cross-task shared metadata (e.g. game info, rosters)."""
    return load_task("shared", split=split, **kwargs)
