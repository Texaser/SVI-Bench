import argparse
import os
import json
import glob
import numpy as np
import sys

# Ensure tools package is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.embedding_utils import get_embedding_model
from _t9_root import T9_ROOT_NOT_SET, resolve_t9_data_root as _resolve_t9_root

# Configuration — defaults resolve to ${T9_ROOT}/data and ${T9_ROOT}/embeds/videos.
# T9_ROOT defaults to <repo>/data/t9/. Override via env var T9_ROOT.
# The presence-check is deferred to main() so this module imports cleanly
# even before `svi-bench download --tasks t9` has populated the data dir.

_T9_ROOT = _resolve_t9_root()
DATA_PATH = os.path.join(_T9_ROOT, "data") if _T9_ROOT != T9_ROOT_NOT_SET else None
OUTPUT_PATH = os.path.join(_T9_ROOT, "embeds", "videos") if _T9_ROOT != T9_ROOT_NOT_SET else None
DATASET_NAME = "basketball"
BATCH_SIZE = 8
LIMIT = None
GAME_IDS = None # Set to None to process all games, or list of strings ["401738"]

def main():
    if _T9_ROOT == T9_ROOT_NOT_SET:
        raise FileNotFoundError(
            "T9 data root not found. Set the T9_ROOT env var or run "
            "`svi-bench download --tasks t9` to populate <repo>/data/t9/."
        )
    # DATA_PATH is always absolute (built from _T9_ROOT above).
    data_path = f"{DATA_PATH}/{DATASET_NAME}"
    print(f"Data Path: {data_path}")
    
    # Resolve Output Path
    out_dir = f"{OUTPUT_PATH}/{DATASET_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    print("Reading metadata.json...")
    metadata_file = os.path.join(data_path, "metadata.json")
    if not os.path.exists(metadata_file):
        print(f"Error: metadata.json not found at {metadata_file}")
        return
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata.json: {e}")
        return

    all_tasks = []
    seen_ids = set()
    print(f"Found {len(metadata)} games in metadata.")
    target_games = GAME_IDS
    if target_games:
        print(f"Filtering for games: {target_games}")

    print(f"Initializing Embedding Model (InternVideo2)...")
    try:
        embed_model = get_embedding_model("internvideo2")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    for game_id, game_data in metadata.items():
        if target_games and game_id not in target_games:
            continue
            
        clip_paths_rel = game_data.get("clip_paths")
        if not clip_paths_rel:
            print(f"Warning: No clip_paths found for game {game_id}")
            continue
            
        # Resolve relative path
        clip_paths_file = os.path.join(data_path, clip_paths_rel)
        
        if not os.path.exists(clip_paths_file):
             print(f"Warning: clip_paths file not found: {clip_paths_file}")
             continue
             
        try:
            with open(clip_paths_file, 'r') as fp:
                clips_map = json.load(fp)
                # data format: {"clip_id": "abs_path" or "rel_path"}.
                # Post-2026-05-16 cleanup, values are relative to the
                # ``clips/`` dir holding clip_paths.json. Pre-cleanup values
                # are absolute and honored verbatim.
                clip_dir = os.path.dirname(clip_paths_file)

                for cid, cpath in clips_map.items():
                    if cid in seen_ids: continue

                    if not os.path.isabs(cpath):
                        cpath = os.path.join(clip_dir, cpath)

                    out_path = os.path.join(out_dir, f"{cid}.npy")

                    # Skip if already exists
                    if os.path.exists(out_path):
                         seen_ids.add(cid)
                         continue

                    all_tasks.append((cid, cpath, out_path))
                    seen_ids.add(cid)
                    
        except Exception as e:
            print(f"Error processing {clip_paths_file}: {e}")

    print(f"Found {len(all_tasks)} clips to extract from metadata.")
    if LIMIT:
        all_tasks = all_tasks[:LIMIT]
        print(f"Limiting to first {LIMIT} tasks.")

    # Batch Processing
    batch_size = BATCH_SIZE
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    
    if total_batches > 0:
        print(f"Starting extraction for {len(all_tasks)} clips in {total_batches} batches...")
        
        from tqdm import tqdm
        for i in tqdm(range(0, len(all_tasks), batch_size), desc="Extracting Embeddings"):
            batch = all_tasks[i:i+batch_size]
            
            cids = [b[0] for b in batch]
            paths = [b[1] for b in batch]
            out_paths = [b[2] for b in batch]
            
            try:
                # get_video_embeddings is a custom method on InternVideo2Embedding
                embeddings = embed_model.get_video_embeddings(paths)
                
                for j, emb in enumerate(embeddings):
                    if j < len(out_paths):
                        np.save(out_paths[j], np.array(emb))
                    else:
                        print(f"Warning: More embeddings returned than paths?")
            except Exception as e:
                print(f"Batch failed ({cids}): {e}")

    else:
        print("No tasks to process.")
        
    print("Done.")

if __name__ == "__main__":
    main()
