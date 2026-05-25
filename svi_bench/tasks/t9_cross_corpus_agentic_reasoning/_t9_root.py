"""T9 data-root resolution — single source of truth.

Used by ``run_agent.py``, ``run_batch.py``, ``tools/extract_embeddings.py``
(and any future tool that needs to find the T9 data directory).

Resolution order:
  1. ``T9_ROOT`` env var, if set.
  2. Walk up from this file to find ``pyproject.toml``; if a ``data/t9/``
     dir lives next to it, use that.
  3. Sentinel ``T9_ROOT_NOT_SET`` so module import never fails — callers
     that actually need the dir should use ``require_t9_data_root()``
     which raises a clear error.
"""

from __future__ import annotations

import os


T9_ROOT_NOT_SET = "T9_ROOT_NOT_SET"


def resolve_t9_data_root() -> str:
    """Return the T9 data root path or the ``T9_ROOT_NOT_SET`` sentinel."""
    if v := os.environ.get("T9_ROOT"):
        return v
    parent = os.path.dirname(os.path.abspath(__file__))
    while parent != "/" and not os.path.isfile(os.path.join(parent, "pyproject.toml")):
        parent = os.path.dirname(parent)
    for name in ("T9", "t9"):
        candidate = os.path.join(parent, "data", name)
        if os.path.isdir(candidate):
            return candidate
    return T9_ROOT_NOT_SET


def require_t9_data_root() -> str:
    """Like ``resolve_t9_data_root`` but raises a clear error if T9 data
    can't be found. Call at any site that actually needs to open a file
    under the data root."""
    root = resolve_t9_data_root()
    if root == T9_ROOT_NOT_SET:
        raise FileNotFoundError(
            "T9 data root not found. Either set the T9_ROOT env var or run "
            "`svi-bench download --tasks t9` to populate <repo>/data/t9/."
        )
    return root


def resolve_t9_results_dir() -> str:
    """Return the directory where experiment results / logs are written.

    Resolution order:
      1. ``T9_RESULTS`` env var, if set.
      2. ``<task_dir>/results/`` (next to this file).
    """
    if v := os.environ.get("T9_RESULTS"):
        return v
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
