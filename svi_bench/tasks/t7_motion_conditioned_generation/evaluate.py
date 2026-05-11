"""T7 entry point exposed to ``svi-bench evaluate``.

T7 is a generation task — there is no closed-form accuracy metric to return.
The CLI hook just dispatches to ``train.sh``, which runs LoRA fine-tuning
followed by periodic validation through ``validate.py``. Per-video sample
outputs land in the ``--output_path`` configured inside ``train.sh``.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

TASK = "t7_motion_conditioned_generation"
HERE = os.path.dirname(os.path.abspath(__file__))


def run(model_name: str | None = None, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Launch the T7 training+validation pipeline.

    ``model_name`` is accepted for CLI uniformity but ignored: the model is
    fixed to Wan2.1-Fun-V1.1-1.3B-Control by the shell wrapper.
    """
    script = os.path.join(HERE, "train.sh")
    proc = subprocess.run(["bash", script], check=False)
    return {
        "task": TASK,
        "script": script,
        "returncode": proc.returncode,
    }
