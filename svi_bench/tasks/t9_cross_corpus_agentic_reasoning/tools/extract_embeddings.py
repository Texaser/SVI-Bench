#!/usr/bin/env python3
"""Extract InternVideo2 clip embeddings for T9 games.

Reads clip paths from metadata.json, runs each clip through InternVideo2,
and writes per-clip .npy embedding files. Resumes automatically by skipping
clips whose .npy already exists.

Requires a GPU with the InternVideo2 checkpoint.

Usage:
  python3 tools/extract_embeddings.py --sport basketball
  python3 tools/extract_embeddings.py --sport hockey --batch-size 16 --limit 100
  python3 tools/extract_embeddings.py --sport soccer --game-ids 5234828 5234880
"""
import argparse
import json
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.embedding_utils import get_embedding_model
from _t9_root import require_t9_data_root


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", required=True, help="Sport name (basketball, hockey, soccer)")
    p.add_argument("--batch-size", type=int, default=8, help="Clips per batch (default: 8)")
    p.add_argument("--limit", type=int, default=None, help="Max clips to process (default: all)")
    p.add_argument("--game-ids", nargs="+", default=None, help="Only process these game IDs")
    p.add_argument("--t9-root", default=None, help="T9 data root (default: T9_ROOT env var)")
    args = p.parse_args()

    if args.t9_root:
        os.environ["T9_ROOT"] = args.t9_root
    t9_root = require_t9_data_root()

    data_path = os.path.join(t9_root, "data", args.sport)
    out_dir = os.path.join(t9_root, "embeds", "videos", args.sport)
    os.makedirs(out_dir, exist_ok=True)

    metadata_file = os.path.join(data_path, "metadata.json")
    if not os.path.exists(metadata_file):
        print(f"Error: metadata.json not found at {metadata_file}")
        return 1

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"Found {len(metadata)} games in metadata.")

    if args.game_ids:
        print(f"Filtering for games: {args.game_ids}")

    print("Initializing InternVideo2 embedding model...")
    embed_model = get_embedding_model("internvideo2")

    all_tasks = []
    seen_ids = set()
    for game_id, game_data in metadata.items():
        if args.game_ids and game_id not in args.game_ids:
            continue

        clip_paths_rel = game_data.get("clip_paths")
        if not clip_paths_rel:
            continue

        clip_paths_file = os.path.join(data_path, clip_paths_rel)
        if not os.path.exists(clip_paths_file):
            continue

        with open(clip_paths_file, 'r') as fp:
            clips_map = json.load(fp)
            clip_dir = os.path.dirname(clip_paths_file)

            for cid, cpath in clips_map.items():
                if cid in seen_ids:
                    continue
                if not os.path.isabs(cpath):
                    cpath = os.path.join(clip_dir, cpath)
                out_path = os.path.join(out_dir, f"{cid}.npy")
                if os.path.exists(out_path):
                    seen_ids.add(cid)
                    continue
                all_tasks.append((cid, cpath, out_path))
                seen_ids.add(cid)

    print(f"Found {len(all_tasks)} clips to extract.")
    if args.limit:
        all_tasks = all_tasks[:args.limit]
        print(f"Limited to first {args.limit} clips.")

    if not all_tasks:
        print("Nothing to process.")
        return 0

    total_batches = (len(all_tasks) + args.batch_size - 1) // args.batch_size
    print(f"Extracting {len(all_tasks)} clips in {total_batches} batches...")

    from tqdm import tqdm
    for i in tqdm(range(0, len(all_tasks), args.batch_size), desc="Extracting"):
        batch = all_tasks[i:i + args.batch_size]
        paths = [b[1] for b in batch]
        out_paths = [b[2] for b in batch]
        try:
            embeddings = embed_model.get_video_embeddings(paths)
            for j, emb in enumerate(embeddings):
                np.save(out_paths[j], np.array(emb))
        except Exception as e:
            cids = [b[0] for b in batch]
            print(f"Batch failed ({cids}): {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
