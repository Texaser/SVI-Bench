import argparse
import os
import json
import numpy as np
import sys
import torch
import torch.multiprocessing as mp
import math
from concurrent.futures import ThreadPoolExecutor
import cv2
import time

# Ensure tools package is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.embedding_utils import InternVideo2Embedding

# Configuration Defaults
DEFAULT_DATA_PATH = "db"
DEFAULT_BATCH_SIZE = 8

def load_video_frames(video_path, num_frames=16, resize=224):
    """
    Load and preprocess video frames. 
    Returns: 
         - frames: list of np.arrays (BGR) or None on failure
         - video_path: str
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames:
            return None, video_path
            
        # Optimization: InternVideo2 internal preprocessor does uniform sampling.
        # But we can do it here to save memory transfer if video is long.
        # For now, let's keep it simple and just return raw frames.
        # Ideally, we should resize here to save bandwidth if possible.
        
        return frames, video_path
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None, video_path

def worker_process(rank, gpu_id, tasks, dataset_name, output_dir, batch_size):
    """
    Worker process for a specific GPU.
    """
    try:
        print(f"[Worker {rank}] Starting on GPU {gpu_id}. Tasks: {len(tasks)}")
        
        # Initialize Model
        device = f"cuda:{gpu_id}"
        model = InternVideo2Embedding(device=device)
        
        # Configure internal batching
        # The model's get_video_embeddings usually takes paths. 
        # But to optimize IO, we want to preload frames.
        # InternVideo2Embedding.get_video_embeddings takes paths and does loading internally.
        # We can either:
        # 1. Modify InternVideo2Embedding to accept pre-loaded frames.
        # 2. Or just suffer the IO in the worker thread (it's parallel across GPUs anyway).
        # Given we have 8 GPUs, 8 concurrent readers might be enough to saturate IO.
        # Let's stick to passing paths for simplicity first, as modifying the model interface is riskier.
        # If IO is still bottleneck, we can revisit.
        
        # Split into batches
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            cids = [b[0] for b in batch]
            paths = [b[1] for b in batch]
            out_paths = [b[2] for b in batch]
            
            try:
                # Check existance first (double check)
                if all(os.path.exists(p) for p in out_paths):
                    continue

                embeddings = model.get_video_embeddings(paths)
                
                for j, emb in enumerate(embeddings):
                    if j < len(out_paths):
                        np.save(out_paths[j], np.array(emb))
                
                if i % (batch_size * 10) == 0:
                     print(f"[Worker {rank}] Processed {i}/{len(tasks)} clips.")
                     
            except Exception as e:
                print(f"[Worker {rank}] Batch failed: {e}")
        
        print(f"[Worker {rank}] Finished.")
        
    except Exception as e:
        print(f"[Worker {rank}] Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Parallel Video Embedding Extraction")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset root")
    parser.add_argument("--dataset_name", default="basketball", help="Dataset name")
    parser.add_argument("--output_path", default=None, help="Root output directory for clip_embeddings")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to utilize")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--limit", type=int, default=None, help="Limit total tasks")
    
    args = parser.parse_args()
    
    # 1. Discovery
    data_path = os.path.join(args.dataset_path, args.dataset_name)
    print(f"Dataset Path: {data_path}")
    
    if args.output_path:
        out_root = args.output_path
    else:
        # Default fallback: dataset_path/clip_embeddings
        out_root = os.path.join(args.dataset_path, "clip_embeddings")
        
    out_dir = os.path.join(out_root, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output Path: {out_dir}")
    
    metadata_file = os.path.join(data_path, "metadata.json")
    if not os.path.exists(metadata_file):
        print(f"Error: metadata.json not found at {metadata_file}")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        
    all_tasks = [] #(cid, path, out_path)
    seen_ids = set()
    
    print(f"Scanning metadata for {len(metadata)} games...")
    for game_id, game_data in metadata.items():
        clip_paths_rel = game_data.get("clip_paths")
        if not clip_paths_rel: continue
        
        clip_paths_file = os.path.join(data_path, clip_paths_rel)
        if not os.path.exists(clip_paths_file): continue
        
        try:
            with open(clip_paths_file, 'r') as fp:
                clips_map = json.load(fp)
                # Values may be absolute or relative to the clips/ directory.
                clip_dir = os.path.dirname(clip_paths_file)
                for cid, cpath in clips_map.items():
                    if cid in seen_ids: continue
                    if not os.path.isabs(cpath):
                        cpath = os.path.join(clip_dir, cpath)
                    out_path = os.path.join(out_dir, f"{cid}.npy")
                    if os.path.exists(out_path):
                        seen_ids.add(cid)
                        continue
                    all_tasks.append((cid, cpath, out_path))
                    seen_ids.add(cid)
        except Exception as e:
            print(f"Error reading {clip_paths_file}: {e}")

    print(f"Found {len(all_tasks)} clips to process.")
    if args.limit:
        all_tasks = all_tasks[:args.limit]
        print(f"Limiting to first {args.limit} tasks.")
        
    if not all_tasks:
        print("No tasks found.")
        return

    # 2. Sharding
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs.")
    
    chunk_size = math.ceil(len(all_tasks) / num_gpus)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]
    
    # Ensure we don't spawn more processes than chunks
    chunks = chunks[:num_gpus] 
    
    # 3. Execution
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank, chunk in enumerate(chunks):
        p = mp.Process(target=worker_process, args=(rank, rank, chunk, args.dataset_name, out_dir, args.batch_size))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("All workers finished.")

if __name__ == "__main__":
    main()
