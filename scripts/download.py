"""Standalone download helper.

Equivalent to `svi-bench download --tasks ...`, kept as a script for users who
prefer not to install the package.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prefetch SVI-Bench task data from HuggingFace.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Task slugs (e.g. t3 t7) or 'all'.",
    )
    args = parser.parse_args(argv)

    from svi_bench.core.data import load_task
    from svi_bench.tasks import all_tasks, resolve

    targets = all_tasks() if "all" in args.tasks else [resolve(t) for t in args.tasks]
    for full in targets:
        print(f"downloading {full}...", file=sys.stderr)
        load_task(full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
