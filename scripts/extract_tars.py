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
import concurrent.futures
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


def _extract_target(tar_path: str):
    """Return (target_dir, expected_bucket_subdir) for ``tar_path``.

    Two HF tar packing conventions coexist:
      (a) members rooted at the bucket name (``26/0019266.mp4``) — pack with
          ``tar -C parent_dir bucket``. Extract to the tar's parent dir; the
          ``<bucket>/`` subdir is created naturally.
      (b) members rooted at ``.`` (``./0083400.txt``) — pack with
          ``tar -C bucket_dir .``. Extract into ``<parent>/<bucket>/`` so the
          bucket layout is reconstructed.
    """
    parent = os.path.dirname(tar_path)
    stem = os.path.splitext(os.path.basename(tar_path))[0]
    try:
        with tarfile.open(tar_path, "r") as tf:
            for m in tf.getmembers():
                if not m.name or m.name == "." or m.name == "./":
                    continue
                head = m.name.lstrip("./").split("/", 1)[0]
                if head == stem:
                    return parent, stem        # convention (a)
                return os.path.join(parent, stem), stem   # convention (b)
    except tarfile.TarError:
        pass
    return parent, stem


def already_extracted(tar_path: str) -> bool:
    """Already extracted if the expected ``<parent>/<bucket>/`` exists and is
    non-empty."""
    target, stem = _extract_target(tar_path)
    bucket_dir = target if os.path.basename(target) == stem else os.path.join(target, stem)
    return os.path.isdir(bucket_dir) and any(os.scandir(bucket_dir))


def extract_one(tar_path: str, dry_run: bool) -> bool:
    target, _stem = _extract_target(tar_path)
    if dry_run:
        log(f"  DRY-RUN would extract -> {target}")
        return True
    try:
        os.makedirs(target, exist_ok=True)
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(path=target)
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
    p.add_argument("--workers", type=int, default=8,
                   help="parallel worker count (default: 8)")
    args = p.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        log(f"FATAL: --root is not a directory: {root}")
        return 1

    tars = sorted(find_tars(root))
    log(f"found {len(tars)} .tar files under {root}")
    if not tars:
        return 0

    def _process(idx_tar):
        i, tar = idx_tar
        rel = os.path.relpath(tar, root)
        size_mb = os.path.getsize(tar) / 1024**2
        if args.skip_extracted and not args.force and already_extracted(tar):
            return ("skip", i, rel, size_mb, None)
        if extract_one(tar, args.dry_run):
            if args.delete_after and not args.dry_run:
                try:
                    os.remove(tar)
                except OSError as e:
                    return ("ok", i, rel, size_mb, f"warn: couldn't rm tar: {e}")
            return ("ok", i, rel, size_mb, None)
        return ("fail", i, rel, size_mb, None)

    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for status, i, rel, size_mb, msg in pool.map(_process, enumerate(tars, 1)):
            tag = {"ok": "OK", "skip": "skip", "fail": "FAIL"}[status]
            log(f"[{i}/{len(tars)}] {tag:<4} {rel} ({size_mb:.1f} MB)" + (f" -- {msg}" if msg else ""))
            if status == "ok": n_ok += 1
            elif status == "skip": n_skip += 1
            else: n_fail += 1

    log(f"\n=== summary: ok={n_ok} skip={n_skip} fail={n_fail} (total {len(tars)}, {(time.time()-t0)/60:.1f} min) ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
