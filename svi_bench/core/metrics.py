"""Common evaluation metrics shared by multiple tasks.

Task-specific metrics belong in the task module, not here.
"""

from __future__ import annotations

from collections.abc import Sequence


def accuracy(preds: Sequence, refs: Sequence) -> float:
    if len(preds) != len(refs):
        raise ValueError(f"length mismatch: {len(preds)} vs {len(refs)}")
    if not preds:
        return 0.0
    return sum(p == r for p, r in zip(preds, refs)) / len(preds)
