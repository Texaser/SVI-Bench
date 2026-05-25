"""T8 entry point exposed to ``svi-bench evaluate``.

T8 is a generation task — the CLI hook dispatches to
``inference/infer.sh``, which shards the task2 basketball test set
across GPUs and runs the trained Wan2.1-Fun LoRA on each shard to produce
video samples. Per-sample outputs land under the checkpoint's
``validation/step-<N>/`` directory.

Unlike T7, T8 only covers the basketball domain.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

TASK = "t8_goal_conditioned_action_generation"
HERE = os.path.dirname(os.path.abspath(__file__))


def run(
    model_name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run T8 post-training evaluation on the latest checkpoint.

    ``model_name`` is accepted for CLI uniformity but ignored. ``output_path``
    overrides the default task2 LoRA checkpoint directory.
    """
    if config:
        output_path = config.get("output_path", output_path)

    script = os.path.join(HERE, "inference", "infer.sh")
    cmd = ["bash", script]
    if output_path:
        cmd.append(output_path)
    proc = subprocess.run(cmd, check=False)
    return {
        "task": TASK,
        "domain": "basketball",
        "script": script,
        "returncode": proc.returncode,
    }
