"""T7: Motion-Conditioned Generation.

LoRA fine-tune of Wan2.1-Fun-V1.1-1.3B-Control conditioned on per-frame
player bounding boxes and a background-video stream. The training and
validation logic lives in the shared vendored DiffSynth-Studio slice at
``svi_bench/tasks/_wan_shared``. Importing this module also imports the
shared slice, which adds the vendored ``diffsynth`` package to ``sys.path``.

Heavy deps (torch, accelerate, peft, transformers, einops, modelscope,
imageio, ...) live in pyproject.toml [project.optional-dependencies] t7 and
must be imported lazily inside evaluate.py / validate.py.
"""

from __future__ import annotations

# Side-effecting import: adds the vendored diffsynth slice to sys.path so
# that any subsequent `from diffsynth import ...` in this package or in
# scripts launched from `train.sh` resolves to the bundled copy.
from svi_bench.tasks import _wan_shared as _wan_shared  # noqa: F401
