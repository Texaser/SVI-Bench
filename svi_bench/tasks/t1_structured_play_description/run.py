"""T1 entry point exposed to ``svi-bench evaluate``.

Dispatches to ``eval/run_caption.sh`` to generate per-clip captions, then
to ``eval/run_llm_judge.sh`` for the LLM-as-a-judge scoring pass.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

TASK = "t1_structured_play_description"
HERE = os.path.dirname(os.path.abspath(__file__))


def run(
    model_name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    model_path: str | None = None,
    results_dir: str | None = None,
    skip_judge: bool = False,
    judge_model: str = "gpt-5.2-2025-12-11",
) -> dict[str, Any]:
    """Run T1 caption generation + LLM judge."""
    if config:
        model_path = config.get("model_path", model_path)
        results_dir = config.get("results_dir", results_dir)
        skip_judge = config.get("skip_judge", skip_judge)
        judge_model = config.get("judge_model", judge_model)

    cap_cmd = ["bash", os.path.join(HERE, "eval", "run_caption.sh")]
    if model_path:
        cap_cmd.append(model_path)
    if results_dir:
        cap_cmd.append(results_dir)
    cap_proc = subprocess.run(cap_cmd, check=False)

    judge_rc = None
    if not skip_judge:
        judge_cmd = ["bash", os.path.join(HERE, "eval", "run_llm_judge.sh"), judge_model]
        if results_dir:
            judge_cmd.append(results_dir)
        judge_rc = subprocess.run(judge_cmd, check=False).returncode

    return {
        "task": TASK,
        "caption_returncode": cap_proc.returncode,
        "judge_returncode": judge_rc,
    }
