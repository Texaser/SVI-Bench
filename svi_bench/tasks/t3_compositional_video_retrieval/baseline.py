"""Reference baseline for T3 (Compositional Video Retrieval).

The published T3 baseline is **InternVideo2-1B fine-tuned on the SVI sports
retrieval training set**. We ship the fine-tuned ckpts on Hugging Face under:

    svi-bench/internvideo2-1b-sports-full      (full-caption training regime)
    svi-bench/internvideo2-1b-sports-partial   (attribute-dropout / "concept" regime)

Each ckpt produces a corresponding set of cached embeddings. T3 evaluation
reads those embeddings directly via ``retrieval`` — no model inference at
eval time. See ``evaluate.py``.

To re-encode new queries or videos with the fine-tuned model, see the
bundled InternVideo2 fork under
``svi_bench/tasks/t3_compositional_video_retrieval/internvideo2/scripts/finetuning/1B/``.
"""

from __future__ import annotations


def run_baseline() -> None:
    """No-op entry point.

    The baseline is the precomputed embeddings + cached ckpts on HF. Users who
    want to re-encode should invoke the bundled InternVideo2 fork after setting
    ``T3_ROOT`` to a directory containing data/, embeds/, ckpts/, compositions/.
    """
    raise NotImplementedError(
        "T3 baseline = pre-extracted embeddings on HF. "
        "Use `svi-bench evaluate --task t3` to run R@K eval. "
        "For re-encoding, see svi_bench/tasks/t3_compositional_video_retrieval/internvideo2/scripts/."
    )
