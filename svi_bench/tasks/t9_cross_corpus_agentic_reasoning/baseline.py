"""Reference baselines for T9 (Cross-Corpus Agentic Reasoning).

The published T9 baselines are agent architectures — each defined by a
single YAML under ``archs/`` — that wire one of several LLMs into a
ReAct-style loop over the bundled retrieval + VQA tools.

Bundled agent architectures (see ``evaluate.py:KNOWN_ARCHS``):

- ``gpt5`` / ``gpt5_oracle``
- ``qwen3_32b`` / ``qwen3_32b_oracle``
- ``qwen3_omni_30b`` / ``qwen3_omni_30b_oracle``
- ``qwen3_235b`` / ``qwen3_235b_oracle``
- ``minimax_m2_5`` / ``minimax_m2_5_oracle``

(``*_oracle`` variants get gold-source clips directly, bypassing retrieval.)

T9 evaluation drives a live agent loop using shipped retrieval embeddings
under ``data/t9/embeds/``. See ``evaluate.py`` for the entry point.
"""

from __future__ import annotations


def run_baseline() -> None:
    """No-op entry point.

    The baselines are the bundled arch YAMLs. Users run them via
    ``svi-bench evaluate --task t9 --model <arch-id>``. For full setup
    (Elasticsearch, vLLM, API keys) see the task README.
    """
    raise NotImplementedError(
        "T9 baselines = bundled agent arch YAMLs. "
        "Use `svi-bench evaluate --task t9 --model <arch-id>` to run."
    )
