"""T8: Goal-Conditioned Action Generation.

LoRA fine-tune of Wan2.1-Fun-V1.1-1.3B-Control with per-video polished
captions and first/last-frame bbox conditioning. The training entry point
(`train.py`) and the slimmed `diffsynth/` package it depends on are
vendored alongside this module so the task is fully self-contained — T7
ships an identical copy of the same slice.

Importing this module adds the bundled `diffsynth/` directory to `sys.path`
so that any subsequent `from diffsynth import ...` (whether in code under
this package or in subprocesses launched by `train.sh`) resolves to the
local copy.

Heavy deps (torch, accelerate, peft, transformers, einops, modelscope,
imageio, ...) live in pyproject.toml [project.optional-dependencies] t8 and
must be imported lazily.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
