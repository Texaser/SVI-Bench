"""YAML config loader.

Configs live in `configs/t<N>.yaml` and specify default hyperparameters,
prompt templates, metrics, and dataset splits per task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = _REPO_ROOT / "configs"


def load_config(task: str) -> dict[str, Any]:
    """Load `configs/<task>.yaml` (or the short `t<N>` form) as a dict."""
    candidates = [CONFIG_DIR / f"{task}.yaml", CONFIG_DIR / f"{task.split('_')[0]}.yaml"]
    for path in candidates:
        if path.exists():
            with path.open() as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(f"no config found for task {task!r} in {CONFIG_DIR}")
