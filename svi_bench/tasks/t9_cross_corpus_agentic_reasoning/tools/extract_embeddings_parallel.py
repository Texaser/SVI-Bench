#!/usr/bin/env python3
"""Multi-GPU parallel extraction of InternVideo2 clip embeddings for T9.

Shards clips across available GPUs, each running an independent InternVideo2
instance. Resumes automatically by skipping clips whose .npy already exists.

Usage:
  python3 tools/extract_embeddings_parallel.py --dataset-path $T9_ROOT/data --sport basketball
  python3 tools/extract_embeddings_parallel.py --dataset-path $T9_ROOT/data --sport hockey --num-gpus 4
  python3 tools/extract_embeddings_parallel.py --dataset-path $T9_ROOT/data --sport soccer --output-path $T9_ROOT/embeds/videos
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.embedding_utils import InternVideo2Embedding


def worker_process(rank, gpu_id, tasks, output_dir, batch_size):
    try:
        print(f"[Worker {rank}] Starting on GPU {gpu_id}. Tasks: {len(tasks)}")
        device = f"cuda:{gpu_id}"
        model = InternVideo2Embedding(device=device)

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            paths = [b[1] for b in batch]
            out_paths = [b[2] for b in batch]

            if all(os.path.exists(p) for p in out_paths):
                continue

            try:
                embeddings = model.get_video_embeddings(paths)
                for j, emb in enumerate(embeddings):
                    np.save(out_paths[j], np.array(emb))
            except Exception as e:
                cids = [b[0] for b in batch]
                print(f"[Worker {rank}] Batch failed ({cids}): {e}")

            if i % (batch_size * 10) == 0 and i > 0:
                print(f"[Worker {rank}] Processed {i}/{len(tasks)} clips.")

        print(f"[Worker {rank}] Finished.")
    except Exception as e:
        print(f"[Worker {rank}] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", required=True, help="Path to T9_ROOT/data")
    p.add_argument("--sport", default="basketball", help="Sport name (default: basketball)")
    p.add_argument("--output-path", default=None,
                   help="Output directory for embeddings (default: <dataset-path>/../embeds/videos)")
    p.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs (default: 8, clamped to available)")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU (default: 8)")
    p.add_argument("--limit", type=int, default=None, help="Max clips to process")
    args = p.parse_args()

    data_path = os.path.join(args.dataset_path, args.sport)
    print(f"Dataset path: {data_path}")

    if args.output_path:
        out_dir = os.path.join(args.output_path, args.sport)
    else:
        out_dir = os.path.join(args.dataset_path, "..", "embeds", "videos", args.sport)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output path: {out_dir}")

    metadata_file = os.path.join(data_path, "metadata.json")
    if not os.path.exists(metadata_file):
        print(f"Error: metadata.json not found at {metadata_file}")
        return 1

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    all_tasks = []
    seen_ids = set()
    print(f"Scanning metadata for {len(metadata)} games...")
    for game_id, game_data in metadata.items():
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

    print(f"Found {len(all_tasks)} clips to process.")
    if args.limit:
        all_tasks = all_tasks[:args.limit]
        print(f"Limited to first {args.limit} clips.")

    if not all_tasks:
        print("Nothing to process.")
        return 0

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs.")

    chunk_size = math.ceil(len(all_tasks) / num_gpus)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]
    chunks = chunks[:num_gpus]

    mp.set_start_method('spawn', force=True)
    processes = []
    for rank, chunk in enumerate(chunks):
        proc = mp.Process(target=worker_process, args=(rank, rank, chunk, out_dir, args.batch_size))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    print("All workers finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
