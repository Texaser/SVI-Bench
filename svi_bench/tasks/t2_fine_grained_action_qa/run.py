"""T2 entry point exposed to ``svi-bench evaluate``.

Dispatches to ``eval/run_qa.sh`` which runs the T1+T2 jointly-trained
LLaVA-Video checkpoint on the T2 multi-choice QA splits.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

TASK = "t2_fine_grained_action_qa"
HERE = os.path.dirname(os.path.abspath(__file__))


def run(
    model_name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    model_path: str | None = None,
    results_dir: str | None = None,
) -> dict[str, Any]:
    if config:
        model_path = config.get("model_path", model_path)
        results_dir = config.get("results_dir", results_dir)

    cmd = ["bash", os.path.join(HERE, "eval", "run_qa.sh")]
    if model_path:
        cmd.append(model_path)
    if results_dir:
        cmd.append(results_dir)
    proc = subprocess.run(cmd, check=False)
    return {"task": TASK, "returncode": proc.returncode}
