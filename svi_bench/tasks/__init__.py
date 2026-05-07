"""Task registry for SVI-Bench.

Maps the short slug (`t1`) and the full slug (`t1_scene_recognition`) to the
import path of the task module. The CLI uses this to dispatch `evaluate`.

Per-task heavy deps must be imported lazily *inside* the task module's
`evaluate.py`, never here. This keeps `import svi_bench` cheap and crash-free
even when a particular task's optional deps are missing.
"""

from __future__ import annotations

# Short-slug -> full-slug mapping. Update both when a task is renamed.
TASK_REGISTRY: dict[str, str] = {
    "t1": "t1_scene_recognition",
    "t2": "t2_placeholder",
    "t3": "t3_action_recognition",
    "t4": "t4_placeholder",
    "t5": "t5_placeholder",
    "t6": "t6_placeholder",
    "t7": "t7_deep_game_analysis",
    "t8": "t8_placeholder",
    "t9": "t9_placeholder",
}


def resolve(task: str) -> str:
    """Accept either `t3` or `t3_action_recognition` and return the full slug."""
    if task in TASK_REGISTRY:
        return TASK_REGISTRY[task]
    if task in TASK_REGISTRY.values():
        return task
    raise KeyError(f"unknown task {task!r}; known: {sorted(TASK_REGISTRY)}")


def all_tasks() -> list[str]:
    return list(TASK_REGISTRY.values())
