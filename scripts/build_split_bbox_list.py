#!/usr/bin/env python3
"""Build a full bbox-paths file from an ID-only splits file.

Given:
  - ID-only splits file: each line is a sample ID (e.g. "0000000").
  - A bbox root dir, where files live at {root}/{bucket}/{ID}.txt
    (default bucket size derives from the largest ID + 1).

Writes a list file where each line is an absolute path to a bbox txt.

Usage:
  python3 build_split_bbox_list.py \
    --ids   data/T7/basketball/splits/train.txt \
    --root  data/T7/basketball/bboxes \
    --out   data/T7/basketball/splits/train.bbox_paths.txt
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True, help="ID-only splits file (one ID per line)")
    p.add_argument("--root", required=True, help="bbox root dir with bucket subdirs")
    p.add_argument("--out", required=True, help="output bbox paths file")
    p.add_argument("--bucket-size", type=int, default=None,
                   help="bucket size (default: auto from largest ID + 1, 100 buckets)")
    args = p.parse_args()

    with open(args.ids) as f:
        ids = [l.strip() for l in f if l.strip()]
    if not ids:
        print(f"empty ids file: {args.ids}", file=sys.stderr)
        return 1

    max_id = max(int(i) for i in ids)
    bucket_size = args.bucket_size or ((max_id + 100) // 100)

    root = os.path.abspath(args.root)
    out_lines = []
    missing = 0
    for sample_id in ids:
        bkt = f"{int(sample_id) // bucket_size:02d}"
        path = f"{root}/{bkt}/{sample_id}.txt"
        if not os.path.exists(path):
            missing += 1
        out_lines.append(path)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"wrote {len(out_lines)} paths to {args.out} (missing on disk: {missing})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
