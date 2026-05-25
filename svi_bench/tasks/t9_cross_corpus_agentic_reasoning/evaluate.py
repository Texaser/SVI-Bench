"""Evaluation entry point for T9 (Cross-Corpus Agentic Reasoning).

T9 is an agentic task: given a sports question, a ReAct-style agent drives
search + question-answering tools over a video + document corpus, then an
LLM judge scores the answer against ground truth.

``run()`` is the judge step only — it never re-runs the agent. The agent is
produced separately by ``scripts/submit_experiment.sh`` (SLURM) or
``run_agent.py`` (interactive); both write to ``$T9_ROOT/results/<run-id>/``.

What ``run()`` does:
  1. Discovers every completed run dir matching the requested arch.
  2. Shows a numbered list and asks which to send to the OpenAI Batch judge.
  3. Reuses ``batch_eval_output.jsonl`` if it already exists; otherwise
     submits a fresh batch and waits.
  4. Returns per-run accuracy.

See ``KNOWN_ARCHS`` below for the bundled agent architectures and how to
register your own.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any

TASK = "t9_cross_corpus_agentic_reasoning"

DATASETS = ["basketball", "hockey", "soccer"]
SPLITS = ["train", "val", "test", "all"]

# Public registry. Each entry maps a user-facing arch id to a dict carrying:
#   - "yaml":  the path (relative to this file) of the arch config YAML.
#
# Adding a custom architecture: drop ``archs/my_arch.yaml`` next to the
# bundled ones, then register here:
#
#     "my-arch": {"yaml": "archs/my_arch.yaml"}
#
# `run()` will then accept ``--model my-arch``.
KNOWN_ARCHS: dict[str, dict[str, Any]] = {
    "gpt5":                  {"yaml": "archs/gpt5.yaml"},
    "gpt5_oracle":           {"yaml": "archs/gpt5_oracle.yaml"},
    "qwen3_32b":             {"yaml": "archs/qwen3_32b.yaml"},
    "qwen3_32b_oracle":      {"yaml": "archs/qwen3_32b_oracle.yaml"},
    "qwen3_omni_30b":        {"yaml": "archs/qwen3_omni_30b.yaml"},
    "qwen3_omni_30b_oracle": {"yaml": "archs/qwen3_omni_30b_oracle.yaml"},
    "qwen3_235b":            {"yaml": "archs/qwen3_235b.yaml"},
    "qwen3_235b_oracle":     {"yaml": "archs/qwen3_235b_oracle.yaml"},
    "qwen3_235b_tools":      {"yaml": "archs/qwen3_235b_tools.yaml"},     # tools-only services-node for 235b
    "minimax_m2_5":          {"yaml": "archs/minimax_m2_5.yaml"},
    "minimax_m2_5_oracle":   {"yaml": "archs/minimax_m2_5_oracle.yaml"},
    "minimax_m2_5_tools":    {"yaml": "archs/minimax_m2_5_tools.yaml"},   # tools-only services-node for minimax
}

# Sentinel meaning "evaluate every known arch".
ALL_ARCHS_TOKEN = "all"


def _resolve_archs(
    model_name: str | None,
    archs_arg: list[str] | None,
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve ``--model`` + ``archs=`` kwarg + config default into a list
    of ``(arch_id, info_dict)`` pairs."""
    if archs_arg:
        ids: list[str] = list(archs_arg)
    elif model_name and model_name != ALL_ARCHS_TOKEN:
        ids = [t.strip() for t in model_name.split(",") if t.strip()]
    elif model_name == ALL_ARCHS_TOKEN:
        ids = list(KNOWN_ARCHS)
    elif cfg_archs := config.get("archs"):
        ids = list(cfg_archs) if isinstance(cfg_archs, list) else [cfg_archs]
    else:
        ids = ["gpt5"]

    unknown = [i for i in ids if i not in KNOWN_ARCHS]
    if unknown:
        raise ValueError(
            f"Unknown T9 arch id(s): {unknown}. "
            f"Known: {sorted(KNOWN_ARCHS)}. Register custom archs by adding "
            f"to KNOWN_ARCHS in {__name__}."
        )
    return [(i, KNOWN_ARCHS[i]) for i in ids]


def _resolve_data_root(local_data_root: str | pathlib.Path | None) -> pathlib.Path:
    """Pick the on-disk root holding data/, embeds/, storage/, results/, ckpts/."""
    if local_data_root is not None:
        return pathlib.Path(local_data_root)

    if v := os.environ.get("T9_ROOT"):
        return pathlib.Path(v)

    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            for name in ("T9", "t9"):
                candidate = parent / "data" / name
                if candidate.exists():
                    return candidate
            break

    from huggingface_hub import snapshot_download

    cached = snapshot_download(
        repo_id="MVP-Group/SVI-Bench",
        repo_type="dataset",
        allow_patterns=["T9/**"],
    )
    return pathlib.Path(cached) / "T9"


def _check_required_env_for_arch(arch_id: str, arch_info: dict[str, Any]) -> None:
    """Fail fast when an arch needs an API key we don't have."""
    # The default LLM judge (configs/models.yaml) is GPT-4o → OpenAI is
    # effectively always required.
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for T9 (the default LLM judge is "
            "GPT-4o). Export it before running, or swap the judge in "
            f"configs/models.yaml. arch={arch_id!r}"
        )


# -----------------------------------------------------------------------------
# Run discovery + interactive picker for the judge step
# -----------------------------------------------------------------------------

# submit_experiment.sh names runs ``<arch>_<sport>_<gpu>_<YYYYMMDD>_<HHMMSS>``.
# Anchoring sport between underscores prevents ``minimax_m2_5`` from matching
# ``minimax_m2_5_oracle_basketball_...`` (and vice versa).
_RUN_RE = re.compile(
    r"^(?P<arch>.+?)_(?P<sport>basketball|hockey|soccer)_(?P<gpu>[^_]+)_(?P<ts>\d{8}_\d{6})$"
)


@dataclass
class RunInfo:
    dir: pathlib.Path        # $T9_ROOT/results/<name>
    name: str                # <arch>_<sport>_<gpu>_<ts>
    arch: str
    sport: str
    timestamp: str           # YYYYMMDD_HHMMSS
    judged: bool             # batch_eval_output.jsonl exists
    covered: int             # unique question_ids across results_worker_*.jsonl
    total: int               # rows in questions_file (from experiment_metadata.json)

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.covered == self.total


def _count_questions(p: pathlib.Path) -> int:
    """Row count of a questions JSON (list or {questions: list})."""
    if not p.exists():
        return 0
    d = json.loads(p.read_text())
    if isinstance(d, list):
        return len(d)
    return len(d.get("questions", d))


def _count_unique_qids(results_dir: pathlib.Path) -> int:
    """Unique question_ids across all worker JSONLs in a run's results/ dir."""
    qids: set[str] = set()
    if not results_dir.is_dir():
        return 0
    for f in results_dir.glob("results_worker_*.jsonl"):
        try:
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = row.get("question_id") or (row.get("question") or {}).get("id")
                if qid is not None:
                    qids.add(str(qid))
        except OSError:
            continue
    return len(qids)


def _inspect_run(run_dir: pathlib.Path, m: re.Match) -> RunInfo | None:
    """Build a RunInfo from a results dir. Returns None if metadata is missing."""
    meta_path = run_dir / "experiment_metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    q_path = pathlib.Path(meta.get("questions_file", ""))
    total = _count_questions(q_path)
    results_subdir = run_dir / "results"
    covered = _count_unique_qids(results_subdir)
    judged = (results_subdir / "batch_eval_output.jsonl").exists()
    return RunInfo(
        dir=run_dir,
        name=run_dir.name,
        arch=m["arch"],
        sport=m["sport"],
        timestamp=m["ts"],
        judged=judged,
        covered=covered,
        total=total,
    )


def _discover_runs(arch_id: str) -> list[RunInfo]:
    """Find every results/{arch_id}_{sport}_{gpu}_{ts}/ dir.

    Returns runs sorted by timestamp (oldest first).
    """
    from svi_bench.tasks.t9_cross_corpus_agentic_reasoning._t9_root import (
        resolve_t9_results_dir,
    )
    results_root = pathlib.Path(resolve_t9_results_dir())
    if not results_root.is_dir():
        return []
    runs: list[RunInfo] = []
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        m = _RUN_RE.match(d.name)
        if not m or m["arch"] != arch_id:
            continue
        info = _inspect_run(d, m)
        if info is not None:
            runs.append(info)
    runs.sort(key=lambda r: r.timestamp)
    return runs


def _picker(runs: list[RunInfo]) -> tuple[list[RunInfo], list[RunInfo]] | None:
    """Print the run table and prompt for which to evaluate.

    Two prompts:
      1. Which complete runs to evaluate. Default ``Enter``/``all`` selects
         every complete run (both un-judged and already-judged); typing
         specific indices selects only those. Incomplete runs are never
         selectable.
      2. Which of the **selected** already-judged runs to re-run the LLM
         judge for (skipped if no selected run is ``reuse``-tagged).

    Returns ``(selected, rejudge)`` — both ``list[RunInfo]``. ``selected`` is
    the full set of runs to process; ``rejudge`` is the subset of those whose
    cached judge output should be discarded and re-submitted. Returns
    ``None`` if the user typed ``q``/``quit`` at either prompt.

    Raises RuntimeError on a non-TTY or an unparseable / out-of-range
    selection.
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "svi-bench evaluate --task t9 needs an interactive shell. "
            "Re-run from a terminal."
        )

    arch = runs[0].arch if runs else "?"
    print(f"\nDiscovered {len(runs)} run(s) for {arch}:\n")
    name_w = max((len(r.name) for r in runs), default=20)

    eligible: list[int] = []   # complete runs (un-judged or reuse)
    for i, r in enumerate(runs, start=1):
        if not r.complete:
            tag = "incomplete"
        else:
            eligible.append(i)
            tag = "reuse" if r.judged else ""
        coverage = f"{r.covered}/{r.total}"
        print(f"  [{i}] {r.name:<{name_w}}   {coverage:>10}   {tag}")

    if not eligible:
        print("\nNothing to evaluate.\n")
        return [], []

    print()
    print("At any prompt, enter 'q' or 'quit' to abort.")

    # ---- Prompt 1: which complete runs to evaluate ----
    print()
    print("Choose runs to evaluate with an LLM judge.")
    print( "  [Enter|all]    evaluate all complete runs")
    print( "  1,3,5          evaluate specific indices")
    raw = input("> ").strip().lower()

    if raw in ("q", "quit"):
        return None
    if raw == "" or raw == "all":
        selected_idxs = list(eligible)
    else:
        try:
            selected_idxs = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            raise RuntimeError(f"Invalid selection: {raw!r}")
        bad = [i for i in selected_idxs if i not in eligible]
        if bad:
            raise RuntimeError(
                f"Index {bad} not eligible (out of range or incomplete); "
                f"eligible indices: {eligible}"
            )

    # ---- Prompt 2: which of the SELECTED runs to re-judge ----
    selected_reuse = [i for i in selected_idxs if runs[i - 1].judged]
    rejudge_idxs: list[int] = []
    if selected_reuse:
        print()
        print(f"Rerun the LLM judge for runs that already have results? ({selected_reuse})")
        print( "  [Enter]        keep cached results")
        print( "  1,4            re-evaluate specific indices")
        raw = input("> ").strip().lower()

        if raw in ("q", "quit"):
            return None
        if raw == "":
            rejudge_idxs = []
        else:
            try:
                rejudge_idxs = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                raise RuntimeError(f"Invalid selection: {raw!r}")
            ineligible = [i for i in rejudge_idxs if i not in selected_reuse]
            if ineligible:
                raise RuntimeError(
                    f"Index {ineligible} not eligible for re-judging; "
                    f"eligible indices: {selected_reuse}"
                )

    selected = [runs[i - 1] for i in selected_idxs]
    rejudge = [runs[i - 1] for i in rejudge_idxs]
    return selected, rejudge


def run(
    model_name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    local_data_root: str | pathlib.Path | None = None,
    archs: list[str] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Score completed T9 agent runs with the OpenAI Batch judge.

    Does not run the agent. The agent runs are produced separately by
    ``scripts/submit_experiment.sh`` (SLURM) or ``run_agent.py`` (interactive),
    and land under ``$T9_ROOT/results/<arch>_<sport>_<gpu>_<ts>/``.

    For each arch in ``model_name``:
      1. Discover every completed run directory.
      2. Show a numbered list and ask which to send to the judge.
      3. Re-aggregate worker outputs, reuse ``batch_eval_output.jsonl`` if it
         exists, otherwise submit a fresh Batch judge job and wait.
      4. Compute accuracy.

    Returns a dict keyed by ``"{arch}/{sport}/{timestamp}"`` whose values are
    ``{accuracy, correct, judged, total}``. Empty if the user picked
    ``none`` and no judged runs were reusable.

    Raises:
        RuntimeError: no completed runs found for an arch, or run from a
            non-interactive shell.
    """
    if config is None:
        try:
            from svi_bench.core.config import load_config

            config = load_config(TASK)
        except FileNotFoundError:
            config = {}

    selected_archs = _resolve_archs(model_name, archs, config)
    root = _resolve_data_root(local_data_root)

    # Lazy imports — keep CLI startup cheap and avoid forcing openai into
    # import-time deps of this module.
    from svi_bench.tasks.t9_cross_corpus_agentic_reasoning.scripts.aggregate_results import (
        aggregate_results,
    )
    from svi_bench.tasks.t9_cross_corpus_agentic_reasoning.scripts.analyze_results import (
        score_aggregated,
    )

    out: dict[str, Any] = {}
    for arch_id, info in selected_archs:
        _check_required_env_for_arch(arch_id, info)

        runs = _discover_runs(arch_id)
        if not runs:
            from svi_bench.tasks.t9_cross_corpus_agentic_reasoning._t9_root import (
                resolve_t9_results_dir,
            )
            raise RuntimeError(
                f"No completed runs found for arch={arch_id} under "
                f"{resolve_t9_results_dir()}. Submit a batch via "
                f"scripts/submit_experiment.sh first."
            )

        pick = _picker(runs)
        if pick is None:
            return out  # user typed 'q' / 'quit'
        selected, rejudge = pick
        selected_set = {r.name for r in selected}
        rejudge_set = {r.name for r in rejudge}

        for r in runs:
            if r.name not in selected_set:
                continue  # not in user's selection → skip entirely
            if r.name in rejudge_set:
                # Force a fresh judge submission by removing the cached output.
                cached = r.dir / "results" / "batch_eval_output.jsonl"
                if cached.exists():
                    cached.unlink()
                submit = True
            elif r.judged:
                submit = False  # reuse cached output
            else:
                submit = True   # un-judged → fresh judge

            run_results_dir = str(r.dir / "results")
            # Always re-aggregate (cheap, ensures aggregated_results.json
            # reflects the current worker JSONLs).
            aggregated, judge_template = aggregate_results(run_results_dir)
            aggregated_path = os.path.join(run_results_dir, "aggregated_results.json")
            with open(aggregated_path, "w") as f:
                json.dump(aggregated, f, indent=2, ensure_ascii=False)
            if judge_template:
                from svi_bench.tasks.t9_cross_corpus_agentic_reasoning.scripts.aggregate_results import (
                    generate_batch_input,
                )
                generate_batch_input(
                    aggregated["results"],
                    judge_template,
                    os.path.join(run_results_dir, "batch_eval_input.jsonl"),
                )

            metrics = score_aggregated(
                aggregated_path,
                run_results_dir,
                submit_if_missing=submit,
            )
            if metrics is not None:
                key = f"{arch_id}/{r.sport}/{r.timestamp}"
                out[key] = metrics
                print(f"\n{key}: {metrics['accuracy']:.2f}%  "
                      f"({metrics['correct']}/{metrics['total']})")

    return out
