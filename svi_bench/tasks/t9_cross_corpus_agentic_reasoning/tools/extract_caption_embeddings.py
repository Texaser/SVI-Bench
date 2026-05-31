#!/usr/bin/env python3
"""Pre-compute BGE-M3 video caption embeddings for T9.

Loads clip metadata, builds TextNodes from event captions, distributes
across GPUs for parallel embedding, and saves the result as a pickle file
that ingest.py can load directly into Elasticsearch.

Only clips with an existing video file on disk are included, ensuring
consistency with the video (InternVideo2) embedding count.

Requires GPU(s) for BGE-M3 embedding.

Usage:
  python3 tools/extract_caption_embeddings.py --sport basketball
  python3 tools/extract_caption_embeddings.py --sport hockey --num-gpus 4
  python3 tools/extract_caption_embeddings.py --sport soccer --batch-size 128
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
import time

import torch
import torch.multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from _t9_root import require_t9_data_root
from run_agent import load_data_metadata

from llama_index.core.schema import MetadataMode, TextNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("extract_caption_embeddings")


def _parse_time_str(time_str: str) -> float:
    try:
        if not time_str or ":" not in time_str:
            return -1.0
        parts = time_str.split(":")
        return float(parts[0]) * 60 + float(parts[1])
    except Exception:
        return -1.0


def _extract_metadata(window_data, game_id, clip_id, clip_path, file_data,
                      sport, canonical_teams, canonical_players):
    m = window_data.get("metadata", {})
    teams = m.get("teams") or file_data.get("teams", [])
    players = m.get("players") or file_data.get("players", [])
    if isinstance(teams, dict): teams = list(teams.values())
    if isinstance(players, dict): players = list(players.values())
    if not isinstance(teams, list): teams = []
    if not isinstance(players, list): players = []
    for t in teams: canonical_teams.add(t)
    for p in players: canonical_players.add(p)

    clock = m.get("game_clock_window", {})
    t_start = _parse_time_str(clock.get("start_remaining"))
    t_end = _parse_time_str(clock.get("end_remaining"))
    t_start_el = _parse_time_str(clock.get("start_elapsed"))
    t_end_el = _parse_time_str(clock.get("end_elapsed"))

    return {
        "game_id": game_id, "sport": sport, "clip_id": clip_id,
        "clip_path": clip_path,
        "period": int(window_data.get("period", 0)),
        "teams": teams, "players": players,
        "time_remaining_max": max(t_start, t_end),
        "time_remaining_min": min(t_start, t_end),
        "time_elapsed_max": max(t_start_el, t_end_el),
        "time_elapsed_min": min(t_start_el, t_end_el),
        "video_window_start": window_data.get("video_window", {}).get("start", -1.0),
        "video_window_end": window_data.get("video_window", {}).get("end", -1.0),
        "events_json": json.dumps(m.get("events", {})),
    }


def build_caption_nodes(data_metadata: dict):
    nodes = []
    canonical_teams = set()
    canonical_players = set()
    skipped = 0

    LOG.info(f"Building caption nodes for {len(data_metadata)} items...")
    for game_id, game_data in data_metadata.items():
        sport = game_data.get("sport", "unknown")
        clips_metadata_path = game_data.get("clips_metadata")
        clip_paths_json = game_data.get("clip_paths")

        if not clips_metadata_path or not os.path.exists(clips_metadata_path):
            continue

        clip_paths_map = {}
        if clip_paths_json and os.path.exists(clip_paths_json):
            with open(clip_paths_json, "r") as f:
                clip_paths_map = json.load(f)

        try:
            with open(clips_metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            clip_dir = os.path.dirname(clip_paths_json) if clip_paths_json else ""
            for window_key, window_data in data.get("windows", {}).items():
                clip_id = window_data.get("window_id")
                clip_rel = clip_paths_map.get(clip_id, "")

                if clip_rel and not os.path.isabs(clip_rel):
                    clip_path = os.path.join(clip_dir, clip_rel)
                else:
                    clip_path = clip_rel

                if not clip_path or not os.path.exists(clip_path):
                    skipped += 1
                    continue

                meta = _extract_metadata(
                    window_data, game_id, clip_id, clip_path, data, sport,
                    canonical_teams, canonical_players,
                )

                events = window_data.get("metadata", {}).get("events", {})
                captions = [evt.get("caption", "") for evt in events.values() if evt.get("caption")]
                text_content = " ".join(captions)

                node = TextNode(text=text_content, id_=clip_id, metadata=meta)
                node.excluded_embed_metadata_keys.extend(
                    ["events_json", "clip_path", "teams", "players"]
                )
                node.excluded_llm_metadata_keys.extend(["events_json"])
                nodes.append(node)
        except Exception as e:
            raise Exception(f"Failed to process game {game_id}: {e}")

    LOG.info(f"Created {len(nodes)} caption nodes. Skipped {skipped} clips without video files.")
    return nodes, canonical_teams, canonical_players


def worker_process(rank, gpu_id, node_dicts, batch_size, output_dir):
    try:
        LOG.info(f"[GPU {rank}] Starting on cuda:{gpu_id} with {len(node_dicts)} nodes")
        nodes = [TextNode.from_dict(d) for d in node_dicts]

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            embed_batch_size=batch_size,
            device=f"cuda:{gpu_id}",
        )
        LOG.info(f"[GPU {rank}] Model loaded")

        start = time.time()
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in batch]
            embeddings = embed_model._get_text_embeddings(texts)
            for node, emb in zip(batch, embeddings):
                node.embedding = emb
            if (i // batch_size) % 50 == 0:
                elapsed = time.time() - start
                rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
                LOG.info(f"[GPU {rank}] {i + len(batch)}/{len(nodes)} ({rate:.0f} nodes/s)")

        out_file = os.path.join(output_dir, f"nodes_gpu_{rank}.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(nodes, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOG.info(f"[GPU {rank}] Done in {time.time() - start:.1f}s.")
    except Exception as e:
        LOG.error(f"[GPU {rank}] Fatal: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", required=True, help="Sport name (basketball, hockey, soccer)")
    p.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs (default: 8)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size per GPU (default: 256)")
    p.add_argument("--t9-root", default=None, help="T9 data root (default: T9_ROOT env var)")
    p.add_argument("--output-dir", default=None, help="Output directory (default: T9_ROOT/embeds/captions/<sport>)")
    args = p.parse_args()

    if args.t9_root:
        os.environ["T9_ROOT"] = args.t9_root
    t9_root = require_t9_data_root()

    output_dir = args.output_dir or os.path.join(t9_root, "embeds", "captions", args.sport)
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(t9_root, "data")
    datasets_metadata = load_data_metadata(data_path, [args.sport])
    all_data_metadata = {}
    for dataset, data_metadata in datasets_metadata.items():
        for item_id, item_data in data_metadata.items():
            item_data["sport"] = dataset
            all_data_metadata[item_id] = item_data
    LOG.info(f"Loaded {len(all_data_metadata)} items for {args.sport}.")

    nodes, canonical_teams, canonical_players = build_caption_nodes(all_data_metadata)

    entities = {"teams": sorted(canonical_teams), "players": sorted(canonical_players)}
    with open(os.path.join(output_dir, "entities.json"), "w") as f:
        json.dump(entities, f, indent=2)
    LOG.info(f"Saved entities: {len(entities['teams'])} teams, {len(entities['players'])} players")

    LOG.info("Serializing nodes for multiprocessing...")
    node_dicts = [n.to_dict() for n in nodes]

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus == 0:
        LOG.error("No GPUs available.")
        return 1
    LOG.info(f"Using {num_gpus} GPUs.")

    chunk_size = math.ceil(len(node_dicts) / num_gpus)
    chunks = [node_dicts[i:i + chunk_size] for i in range(0, len(node_dicts), chunk_size)][:num_gpus]

    mp.set_start_method("spawn", force=True)
    processes = []
    t0 = time.time()
    for rank, chunk in enumerate(chunks):
        proc = mp.Process(target=worker_process, args=(rank, rank, chunk, args.batch_size, output_dir))
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()

    failed = [i for i, proc in enumerate(processes) if proc.exitcode != 0]
    if failed:
        LOG.error(f"Workers failed: {failed}")
        return 1

    LOG.info(f"All workers done in {time.time() - t0:.1f}s. Merging...")
    all_nodes = []
    for rank in range(len(chunks)):
        gpu_file = os.path.join(output_dir, f"nodes_gpu_{rank}.pkl")
        with open(gpu_file, "rb") as f:
            all_nodes.extend(pickle.load(f))

    missing = sum(1 for n in all_nodes if n.embedding is None)
    if missing:
        LOG.warning(f"{missing}/{len(all_nodes)} nodes missing embeddings.")

    merged_path = os.path.join(output_dir, "all_nodes_embedded.pkl")
    with open(merged_path, "wb") as f:
        pickle.dump(all_nodes, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOG.info(f"Saved {len(all_nodes)} embedded nodes to {merged_path}")

    for rank in range(len(chunks)):
        gpu_file = os.path.join(output_dir, f"nodes_gpu_{rank}.pkl")
        if os.path.exists(gpu_file):
            os.remove(gpu_file)

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
