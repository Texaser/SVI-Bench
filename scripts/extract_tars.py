"""Extract .tar bundles downloaded from the SVI-Bench HF repo.

Several of the larger SVI-Bench data trees (T3/clips, T9/embeds, T9/data)
are shipped as .tar bundles to stay under the Hugging Face per-repo file
limit. After downloading the repo locally, run this helper to expand the
bundles in place.

Examples:
  # extract everything under a snapshot dir, then delete the tars
  python3 extract_tars.py --root /path/to/SVI-snapshot --delete-after

  # dry-run first to see what would happen
  python3 extract_tars.py --root /path/to/SVI-snapshot --dry-run

  # only extract a subtree (e.g. just T9/embeds)
  python3 extract_tars.py --root /path/to/SVI-snapshot/T9/embeds
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
from datetime import datetime


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def find_tars(root: str):
    """Yield absolute paths to every *.tar file under root.

    Skips PyTorch checkpoint files (legacy `*.pth.tar` naming convention)
    — those are not actual tar archives.
    """
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".tar") and not f.endswith(".pth.tar"):
                yield os.path.join(dirpath, f)


def already_extracted(tar_path: str) -> bool:
    """Heuristic: a tar is 'already extracted' if every top-level member
    it would write already exists in its parent dir."""
    parent = os.path.dirname(tar_path)
    try:
        with tarfile.open(tar_path, "r") as tf:
            top_members = {m.name.split("/", 1)[0] for m in tf.getmembers() if m.name}
    except tarfile.TarError:
        return False
    if not top_members:
        return False
    return all(os.path.exists(os.path.join(parent, m)) for m in top_members)


def extract_one(tar_path: str, dry_run: bool) -> bool:
    parent = os.path.dirname(tar_path)
    if dry_run:
        log(f"  DRY-RUN would extract -> {parent}")
        return True
    try:
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(path=parent)
        return True
    except (tarfile.TarError, OSError) as e:
        log(f"  ERROR extracting: {e}")
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="dir to recursively scan for *.tar files")
    p.add_argument("--delete-after", action="store_true",
                   help="rm the .tar after successful extraction")
    p.add_argument("--skip-extracted", action="store_true", default=True,
                   help="skip tars whose top-level members already exist (default: on)")
    p.add_argument("--force", action="store_true",
                   help="extract even if --skip-extracted would skip")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would happen without extracting/deleting")
    args = p.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        log(f"FATAL: --root is not a directory: {root}")
        return 1

    tars = sorted(find_tars(root))
    log(f"found {len(tars)} .tar files under {root}")
    if not tars:
        return 0

    n_ok = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()
    for i, tar in enumerate(tars, start=1):
        rel = os.path.relpath(tar, root)
        size_mb = os.path.getsize(tar) / 1024**2
        log(f"[{i}/{len(tars)}] {rel} ({size_mb:.1f} MB)")

        if args.skip_extracted and not args.force and already_extracted(tar):
            log("  already extracted; skipping")
            n_skip += 1
            continue

        if extract_one(tar, args.dry_run):
            n_ok += 1
            if args.delete_after and not args.dry_run:
                try:
                    os.remove(tar)
                    log("  removed tar")
                except OSError as e:
                    log(f"  warn: couldn't rm tar: {e}")
        else:
            n_fail += 1

    log(f"\n=== summary: ok={n_ok} skip={n_skip} fail={n_fail} (total {len(tars)}, {(time.time()-t0)/60:.1f} min) ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
