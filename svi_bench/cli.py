"""Unified CLI entry point.

Subcommands:
  svi-bench evaluate --task <slug|all> --model <name>
  svi-bench download --tasks <slug...>
  svi-bench list
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

from svi_bench.tasks import TASK_REGISTRY, all_tasks, resolve


def _cmd_list(_: argparse.Namespace) -> int:
    for short, full in TASK_REGISTRY.items():
        print(f"{short:>4}  {full}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    targets = all_tasks() if args.task == "all" else [resolve(args.task)]
    results: dict[str, Any] = {}
    for full in targets:
        # Lazy import — only the requested task's deps are touched.
        try:
            mod = importlib.import_module(f"svi_bench.tasks.{full}.evaluate")
        except ImportError as e:
            print(f"[{full}] skipped: {e}", file=sys.stderr)
            continue
        try:
            results[full] = mod.run(args.model)
        except NotImplementedError as e:
            print(f"[{full}] not implemented: {e}", file=sys.stderr)
            continue
    print(json.dumps(results, indent=2))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from svi_bench.core.data import load_task

    targets = all_tasks() if "all" in args.tasks else [resolve(t) for t in args.tasks]
    for full in targets:
        print(f"downloading {full}...", file=sys.stderr)
        load_task(full)  # forces a download into the HF cache
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="svi-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("evaluate", help="run evaluation for one or all tasks")
    p_eval.add_argument("--task", required=True, help="task slug (e.g. t3) or 'all'")
    p_eval.add_argument("--model", required=True, help="model name (e.g. gpt-4o)")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_dl = sub.add_parser("download", help="prefetch task data from HF")
    p_dl.add_argument("--tasks", nargs="+", required=True, help="task slugs or 'all'")
    p_dl.set_defaults(func=_cmd_download)

    p_ls = sub.add_parser("list", help="list registered tasks")
    p_ls.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
