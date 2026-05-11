"""Vendored slice of DiffSynth-Studio.

Shared between T7 (motion-conditioned generation) and T8 (goal-conditioned
action generation): both tasks are LoRA fine-tunes of Wan2.1-Fun-V1.1-1.3B
with bbox + background-video conditioning, driven by the same upstream
training entry point and differing only in shell args and per-task
validation logic.

Importing this package places the bundled `diffsynth` directory on
``sys.path`` so the unmodified upstream `from diffsynth import ...` lines in
``train.py`` and the validation scripts resolve here rather than depending on
the user having a separate DiffSynth-Studio checkout.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

TRAIN_SCRIPT = os.path.join(_HERE, "train.py")
"""Absolute path to the shared `train.py` entry point that `accelerate launch`
invokes from each task's ``train.sh``."""
