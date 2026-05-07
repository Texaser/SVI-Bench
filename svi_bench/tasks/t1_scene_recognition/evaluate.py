"""Evaluation entry point for T1 (scene recognition)."""

from __future__ import annotations

from typing import Any

TASK = "t1_scene_recognition"


def run(model_name: str, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run T1 evaluation. Returns a metrics dict."""
    from svi_bench.core.config import load_config
    from svi_bench.core.data import load_task
    from svi_bench.core.metrics import accuracy
    from svi_bench.core.models import get_model

    cfg = config or load_config(TASK)
    model = get_model(model_name)
    ds = load_task(TASK, split=cfg.get("split", "test"))

    preds, refs = [], []
    for example in ds:
        prompt = cfg.get("prompt_template", "{question}").format(**example)
        out = model.generate(prompt)
        preds.append(out.strip())
        refs.append(example.get("answer"))

    return {"accuracy": accuracy(preds, refs), "n": len(preds)}
