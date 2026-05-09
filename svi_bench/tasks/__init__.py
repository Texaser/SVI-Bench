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
    "t1": "t1_structured_play_description",
    "t2": "t2_fine_grained_action_qa",
    "t3": "t3_compositional_video_retrieval",
    "t4": "t4_strategic_reasoning_qa",
    "t5": "t5_outcome_forecasting",
    "t6": "t6_long_form_narrative_synthesis",
    "t7": "t7_motion_conditioned_generation",
    "t8": "t8_goal_conditioned_action_generation",
    "t9": "t9_cross_corpus_agentic_reasoning",
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
