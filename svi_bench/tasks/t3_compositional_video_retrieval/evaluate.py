"""Evaluation entry point for T3 (Compositional Video Retrieval).

T3 is a retrieval task: given a natural-language query, rank the correct clip
among 1 positive + 5,000 same-sport negatives. ``run()`` computes R@K
(K = 1, 5, 10, 50, 100, ...) on cached, pre-extracted embeddings.

If embeddings for the requested ``--model`` are missing, ``run()`` will
encode them first using the bundled InternVideo2 fork — confirming with the
user beforehand (TTY) or via the ``T3_AUTO_ENCODE=1`` env var bypass (non-TTY).

See ``KNOWN_MODELS`` below for bundled baselines and how to register your own.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import Any

TASK = "t3_compositional_video_retrieval"

SPLITS = ["val", "test"]
SPORTS = ["basketball", "hockey", "soccer"]

# Public registry. Each entry maps a user-facing model id to a dict carrying:
#   - "suffix":         the embedding-filename suffix; the file scored is
#                       ``embeds_{split}_{sport}_{suffix}.pt``.
#   - "regime":         optional training-regime name (``"full_caption"`` or
#                       ``"attribute_dropout"``) selecting which bundled
#                       re-encoding shell script to invoke when embeddings
#                       are missing. Omit (or set to ``None``) for
#                       user-trained models that bring their own embeddings.
#
# Adding a user-trained model: pick a suffix, place your embeddings at
# ``data/T3/embeds/embeds_{split}_{sport}_<suffix>.pt``, then register here:
#
#     "my-model-v3": {"suffix": "my-suffix"}        # BYO embeddings
#
# The bundled checkpoints have re-encoder scripts (regime below selects the
# matching ``internvideo2/scripts/finetuning/1B/eval_{regime}_{split}_{sport}.sh``).
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "internvideo2-1b-sports-full":    {"suffix": "full",    "regime": "full_caption"},
    "internvideo2-1b-sports-partial": {"suffix": "partial", "regime": "attribute_dropout"},
}

# Sentinel meaning "evaluate every known model".
ALL_MODELS_TOKEN = "all"


def _resolve_models(
    model_name: str | None,
    models_arg: list[str] | None,
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve ``--model`` CLI input + ``models=`` kwarg + config default
    into a list of ``(model_id, info_dict)`` pairs.
    """
    if models_arg:
        ids: list[str] = list(models_arg)
    elif model_name and model_name != ALL_MODELS_TOKEN:
        ids = [t.strip() for t in model_name.split(",") if t.strip()]
    elif model_name == ALL_MODELS_TOKEN:
        ids = list(KNOWN_MODELS)
    elif cfg_models := config.get("models"):
        ids = list(cfg_models) if isinstance(cfg_models, list) else [cfg_models]
    else:
        ids = list(KNOWN_MODELS)

    unknown = [i for i in ids if i not in KNOWN_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown T3 model id(s): {unknown}. "
            f"Known: {sorted(KNOWN_MODELS)}. Register custom models by "
            f"adding to KNOWN_MODELS in {__name__}."
        )
    return [(i, KNOWN_MODELS[i]) for i in ids]


def _resolve_data_root(local_data_root: str | pathlib.Path | None) -> pathlib.Path:
    """Pick the on-disk root that contains data/, embeds/, compositions/, ckpts/."""
    if local_data_root is not None:
        return pathlib.Path(local_data_root)

    if v := os.environ.get("T3_ROOT"):
        return pathlib.Path(v)

    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            for name in ("T3", "t3"):
                candidate = parent / "data" / name
                if candidate.exists():
                    return candidate
            break

    from huggingface_hub import snapshot_download

    cached = snapshot_download(
        repo_id="MVP-Group/SVI-Bench",
        repo_type="dataset",
        allow_patterns=[f"{TASK}/*"],
    )
    return pathlib.Path(cached) / TASK


def _embed_path(embed_dir: pathlib.Path, sport: str, split: str, suffix: str) -> pathlib.Path:
    return embed_dir / f"embeds_{split}_{sport}_{suffix}.pt"


def _internvideo2_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "internvideo2"


def _missing_combinations(
    combos: list[tuple[str, str, str, dict[str, Any]]],
    embed_dir: pathlib.Path,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    return [c for c in combos
            if not _embed_path(embed_dir, c[0], c[1], c[3]["suffix"]).exists()]


def _format_missing(missing, embed_dir):
    lines = []
    for sport, split, model_id, info in missing:
        path = _embed_path(embed_dir, sport, split, info["suffix"])
        lines.append(f"       - {sport}/{split}  -> {path}")
    return "\n".join(lines)


def _check_user_models_present(missing, embed_dir) -> None:
    """If any missing combo belongs to a user-trained (no-encoder) model, raise."""
    user_only = [c for c in missing if c[3].get("regime") is None]
    if not user_only:
        return
    raise RuntimeError(
        "Embeddings missing for user-trained model(s) without a bundled encoder. "
        "Place pre-computed embeddings at the expected paths:\n"
        + _format_missing(user_only, embed_dir)
    )


def _confirm_or_exit(missing, embed_dir, ckpt_dir) -> None:
    """TTY: prompt y/N. Non-TTY: hard error unless T3_AUTO_ENCODE=1."""
    if os.environ.get("T3_AUTO_ENCODE") == "1":
        print(f"[t3] T3_AUTO_ENCODE=1; encoding {len(missing)} missing combination(s).")
        return

    if not sys.stdin.isatty():
        sys.stderr.write(
            f"\n[t3] Embeddings missing for {len(missing)} combination(s) "
            f"and stdin is not a TTY (no interactive prompt available).\n"
            f"[t3] Set T3_AUTO_ENCODE=1 to bypass the confirm, or place "
            f"pre-computed embeddings at the expected paths:\n"
            + _format_missing(missing, embed_dir) + "\n"
        )
        sys.exit(1)

    print(f"[t3] Embeddings missing for {len(missing)} combination(s):")
    print(_format_missing(missing, embed_dir))
    print(f"[t3] Encoding requires GPU + checkpoints under {ckpt_dir}/.")
    print()
    try:
        answer = input("Proceed with encoding? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("[t3] Aborted. No changes made.")
        sys.exit(1)


def _encode_one(sport: str, split: str, model_info: dict[str, Any]) -> None:
    """Subprocess into the bundled InternVideo2 eval shell for this combo."""
    regime = model_info.get("regime")
    if regime is None:
        raise RuntimeError("internal error: _encode_one called without a regime")
    script_rel = f"scripts/finetuning/1B/eval_{regime}_{split}_{sport}.sh"
    iv2 = _internvideo2_root()
    script_abs = iv2 / script_rel
    if not script_abs.exists():
        raise RuntimeError(f"Encoder script not found: {script_abs}")
    print(f"[t3] Encoding {sport}/{split} via {script_rel} ...")
    subprocess.run(["bash", script_rel], cwd=str(iv2), check=True)


def run(
    model_name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    local_data_root: str | pathlib.Path | None = None,
    sports: list[str] | None = None,
    splits: list[str] | None = None,
    models: list[str] | None = None,
    output_dir: str | pathlib.Path | None = None,
    save_results: bool = False,
) -> dict[str, Any]:
    """Run T3 R@K evaluation. If embeddings are missing, encode them first.

    Missing-embeddings behavior:
        - If any missing combo belongs to a user-trained model without an
          encoder, raise immediately with the expected paths.
        - Otherwise, prompt the user (TTY) or check ``T3_AUTO_ENCODE=1``
          (non-TTY), then subprocess into the bundled InternVideo2 fork to
          encode the missing combos before scoring.

    Returns a dict keyed by ``"{sport}/{split}/{model_id}"`` whose values are
    the per-combination metric summary (R@1, R@10, R@100, MedR, n).
    """
    if config is None:
        config = {}

    sports = sports or config.get("sports") or SPORTS
    splits = splits or config.get("splits") or SPLITS
    selected_models = _resolve_models(model_name, models, config)

    root = _resolve_data_root(local_data_root)
    data_dir = pathlib.Path(os.environ.get("T3_DATA_DIR", root / "data"))
    embed_dir = pathlib.Path(os.environ.get("T3_EMBED_DIR", root / "embeds"))
    mappings_dir = pathlib.Path(
        os.environ.get("T3_MAPPINGS_DIR", root / "compositions" / "mappings")
    )
    ckpt_dir = root / "ckpts"

    if save_results and output_dir is None:
        default_results_dir = pathlib.Path(__file__).resolve().parent / "results"
        output_dir = pathlib.Path(os.environ.get("T3_OUTPUT_DIR", default_results_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
    elif output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build the (sport, split, model_id, info) grid.
    combos = [
        (sport, split, model_id, info)
        for sport in sports
        for split in splits
        for model_id, info in selected_models
    ]

    # Encoding gate: handle missing embeddings before scoring.
    missing = _missing_combinations(combos, embed_dir)
    if missing:
        _check_user_models_present(missing, embed_dir)
        _confirm_or_exit(missing, embed_dir, ckpt_dir)
        for sport, split, _, info in missing:
            _encode_one(sport, split, info)

    # Lazy import — torch only needed when actually scoring.
    from svi_bench.tasks.t3_compositional_video_retrieval.retrieval import evaluate_one

    results: dict[str, Any] = {}
    for sport, split, model_id, info in combos:
        key = f"{sport}/{split}/{model_id}"
        out = evaluate_one(
            sport,
            split,
            info["suffix"],
            data_dir=data_dir,
            embed_dir=embed_dir,
            mappings_dir=mappings_dir,
            output_dir=output_dir,
        )
        if out is not None:
            results[key] = {
                "R@1": out["overall"]["R@1"],
                "R@10": out["overall"]["R@10"],
                "R@100": out["overall"]["R@100"],
                "MedR": out["overall"]["MedR"],
                "n": out["overall"]["n"],
            }
    return results
