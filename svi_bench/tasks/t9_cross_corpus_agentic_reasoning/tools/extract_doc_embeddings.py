#!/usr/bin/env python3
"""Pre-compute BGE-M3 document embeddings for T9.

Loads documents from metadata.json, parses into sentence-window nodes,
distributes across GPUs for parallel embedding, and saves the result as
a pickle file that ingest.py can load directly into Elasticsearch.

Requires GPU(s) for BGE-M3 embedding.

Usage:
  python3 tools/extract_doc_embeddings.py --sport basketball
  python3 tools/extract_doc_embeddings.py --sport hockey --num-gpus 4
  python3 tools/extract_doc_embeddings.py --sport soccer --batch-size 128
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
from tools.document_tools import load_documents, _CANONICAL_TEAMS, _CANONICAL_PLAYERS

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import MetadataMode, TextNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("extract_doc_embeddings")

ENABLED_SOURCES = ["espn_report", "game_statistics", "season_statistics"]


def build_nodes(data_metadata: dict) -> list:
    docs = load_documents(data_metadata, ENABLED_SOURCES)
    LOG.info(f"Loaded {len(docs)} documents.")

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    nodes = node_parser.get_nodes_from_documents(docs)

    for node in nodes:
        node.excluded_embed_metadata_keys.extend(["window", "original_text"])

    LOG.info(f"Created {len(nodes)} nodes from {len(docs)} documents.")
    return nodes


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
    p.add_argument("--output-dir", default=None, help="Output directory (default: T9_ROOT/embeds/documents/<sport>)")
    args = p.parse_args()

    if args.t9_root:
        os.environ["T9_ROOT"] = args.t9_root
    t9_root = require_t9_data_root()

    output_dir = args.output_dir or os.path.join(t9_root, "embeds", "documents", args.sport)
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(t9_root, "data")
    datasets_metadata = load_data_metadata(data_path, [args.sport])
    all_data_metadata = {}
    for dataset, data_metadata in datasets_metadata.items():
        for item_id, item_data in data_metadata.items():
            item_data["sport"] = dataset
            all_data_metadata[item_id] = item_data
    LOG.info(f"Loaded {len(all_data_metadata)} items for {args.sport}.")

    nodes = build_nodes(all_data_metadata)

    entities = {"teams": sorted(_CANONICAL_TEAMS), "players": sorted(_CANONICAL_PLAYERS)}
    with open(os.path.join(output_dir, "doc_entities.json"), "w") as f:
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
