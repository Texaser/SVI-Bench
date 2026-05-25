"""T7 entry point exposed to ``svi-bench evaluate``.

T7 is a generation task — the CLI hook just dispatches to one of the
``inference/*.sh`` scripts, which shard the test set across GPUs and run
the trained Wan2.1-Fun LoRA on each shard to produce video samples.
Per-sample outputs land under the checkpoint's ``validation/step-<N>/``
directory.

Domains:
    - basketball  (default)
    - soccer
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

TASK = "t7_motion_conditioned_generation"
HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS = ("basketball", "soccer")


def run(
    model_name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    domain: str = "basketball",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run T7 post-training evaluation on the latest checkpoint.

    ``model_name`` is accepted for CLI uniformity but ignored (the model is
    fixed to Wan2.1-Fun-V1.1-1.3B-Control by the shell wrapper).
    ``domain`` selects ``basketball`` or ``soccer``. ``output_path`` is
    forwarded as the first positional arg of the shell script and overrides
    the default LoRA checkpoint directory.
    """
    if config:
        domain = config.get("domain", domain)
        output_path = config.get("output_path", output_path)
    if domain not in DOMAINS:
        raise ValueError(f"unknown T7 domain {domain!r}; pick one of {DOMAINS}")

    script = os.path.join(HERE, "inference", "infer.sh")
    cmd = ["bash", script]
    if output_path:
        cmd.append(output_path)
    env = os.environ.copy()
    env["SPORT"] = domain
    proc = subprocess.run(cmd, check=False, env=env)
    return {
        "task": TASK,
        "domain": domain,
        "script": script,
        "returncode": proc.returncode,
    }
