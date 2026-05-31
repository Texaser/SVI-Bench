#!/usr/bin/env python3
"""Pre-populate Elasticsearch indices for T9 before running experiments.

Loads pre-computed embedding nodes from disk and bulk-inserts them into
Elasticsearch. No GPU or embedding model needed — all embeddings are
pre-computed and shipped with the dataset.

Three index types per sport:
  - document_index_{sport}_m3_all      (document search)
  - video_index_{sport}_video_internvideo2_all  (video search by visual embedding)
  - video_index_{sport}_caption_m3_all  (video search by caption embedding)

Requires:
  - Elasticsearch running (see README for setup)
  - T9 data downloaded and extracted

Usage:
  python3 scripts/ingest.py
  python3 scripts/ingest.py --es-url http://my-es-host:9200
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")
warnings.filterwarnings("ignore", category=ResourceWarning)

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

from _t9_root import require_t9_data_root
from run_agent import load_data_metadata, load_config

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.vector_stores.elasticsearch import (
    AsyncBM25Strategy,
    AsyncDenseVectorStrategy,
    ElasticsearchStore,
)

logging.getLogger("llama_index.core.embeddings.mock_embed_model").setLevel(logging.CRITICAL)
logging.getLogger("llama_index").setLevel(logging.WARNING)


def _es_index_populated(es_url: str, index_name: str) -> bool:
    try:
        from elasticsearch import Elasticsearch
        client = Elasticsearch(es_url)
        try:
            if not client.indices.exists(index=index_name):
                return False
            return client.count(index=index_name).get("count", 0) > 0
        finally:
            try:
                client.close()
            except Exception:
                pass
    except Exception:
        return False


def ingest_precomputed_nodes(
    nodes_pkl: str,
    index_name: str,
    es_url: str,
    embed_dim: int,
    persist_dir: str,
    flag_name: str,
    entities_src: str | None = None,
    entities_dst_name: str = "doc_entities.json",
    batch_size: int = 4096,
) -> bool:
    if _es_index_populated(es_url, index_name):
        print(f"  Index '{index_name}' already populated — skipping.")
        return True

    if not os.path.exists(nodes_pkl):
        print(f"  ERROR: pre-computed nodes not found: {nodes_pkl}")
        return False

    print(f"  Loading nodes from {os.path.basename(nodes_pkl)} ...")
    with open(nodes_pkl, "rb") as f:
        nodes = pickle.load(f)
    print(f"  Loaded {len(nodes)} nodes.")

    missing = sum(1 for n in nodes if n.embedding is None)
    if missing:
        print(f"  WARNING: {missing}/{len(nodes)} nodes missing embeddings.")

    es_store = ElasticsearchStore(
        es_url=es_url,
        index_name=index_name,
        dim=embed_dim,
        retrieval_strategy=AsyncDenseVectorStrategy(hybrid=False, rrf=False),
    )
    storage_context = StorageContext.from_defaults(vector_store=es_store)
    noop_embed = MockEmbedding(embed_dim=embed_dim)

    print(f"  Ingesting into '{index_name}' (batch_size={batch_size}) ...")
    t0 = time.time()
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=noop_embed,
        show_progress=True,
        use_async=True,
        insert_batch_size=batch_size,
    )
    print(f"  Ingested in {time.time() - t0:.1f}s.")

    os.makedirs(persist_dir, exist_ok=True)
    with open(os.path.join(persist_dir, flag_name), "w") as f:
        f.write("done")

    if entities_src and os.path.exists(entities_src):
        shutil.copy2(entities_src, os.path.join(persist_dir, entities_dst_name))

    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--es-url", default=None,
                   help="Elasticsearch URL (default: from config or http://localhost:9200)")
    p.add_argument("--t9-root", default=None,
                   help="T9 data root (default: T9_ROOT env var or auto-detected)")
    args = p.parse_args()

    if args.t9_root:
        os.environ["T9_ROOT"] = args.t9_root
    t9_root = require_t9_data_root()
    print(f"T9_ROOT: {t9_root}")

    hyper_cfg = load_config(os.path.join(TASK_DIR, "configs/hyperparameters.yaml"))
    paths_cfg = load_config(os.path.join(TASK_DIR, "configs/paths.yaml"))

    for key in ("video_persist_dir", "document_persist_dir"):
        val = paths_cfg.get(key)
        if val and not os.path.isabs(val):
            paths_cfg[key] = os.path.join(t9_root, val)

    sports = hyper_cfg.get("data", {}).get("enabled_datasets", [])
    if not sports:
        print("ERROR: no sports configured in hyperparameters.yaml")
        return 1

    es_url = (args.es_url
              or os.environ.get("T9_ES_URL")
              or hyper_cfg.get("elasticsearch", {}).get("url", "http://localhost:9200"))

    print(f"Sports: {sports}")
    print(f"Elasticsearch: {es_url}")

    embeds_base = os.path.join(t9_root, "embeds")
    overall_t0 = time.time()
    failures = 0

    for sport in sports:
        print(f"\n{'='*60}")
        print(f"  {sport}")
        print(f"{'='*60}")

        split = "all"

        # 1. Documents
        print(f"\n--- Documents ({sport}) ---")
        doc_pkl = os.path.join(embeds_base, "documents", sport, "all_nodes_embedded.pkl")
        doc_entities = os.path.join(embeds_base, "documents", sport, "doc_entities.json")
        doc_persist = f"{paths_cfg['document_persist_dir']}_{sport}_m3_{split}"
        if not ingest_precomputed_nodes(
            nodes_pkl=doc_pkl,
            index_name=f"document_index_{sport}_m3_{split}",
            es_url=es_url,
            embed_dim=1024,
            persist_dir=doc_persist,
            flag_name="es_ingested.flag",
            entities_src=doc_entities,
            entities_dst_name="doc_entities.json",
        ):
            failures += 1

        # 2. Video (InternVideo2 embeddings)
        print(f"\n--- Videos ({sport}) ---")
        video_persist = f"{paths_cfg['video_persist_dir']}_{sport}"
        video_entities = os.path.join(embeds_base, "captions", sport, "entities.json")

        # Video nodes are built from .npy files at runtime (fast, no model needed)
        # Load metadata + build nodes + ingest
        data_path = os.path.join(t9_root, "data")
        datasets_metadata = load_data_metadata(data_path, [sport])
        all_data_metadata = {}
        for dataset, data_metadata in datasets_metadata.items():
            for item_id, item_data in data_metadata.items():
                item_data["sport"] = dataset
                all_data_metadata[item_id] = item_data

        clip_embeddings_base = os.path.join(embeds_base, "videos")

        if not _es_index_populated(es_url, f"video_index_{sport}_video_internvideo2_{split}"):
            print(f"  Building video nodes from .npy files ...")
            sys.path.insert(0, TASK_DIR)
            from tools.video_tools import _load_video_nodes_visual

            nodes = _load_video_nodes_visual(all_data_metadata, clip_embeddings_base)
            if nodes:
                print(f"  Loaded {len(nodes)} video nodes.")
                es_store = ElasticsearchStore(
                    es_url=es_url,
                    index_name=f"video_index_{sport}_video_internvideo2_{split}",
                    dim=512,
                    retrieval_strategy=AsyncDenseVectorStrategy(hybrid=False, rrf=False),
                )
                storage_context = StorageContext.from_defaults(vector_store=es_store)
                noop_embed = MockEmbedding(embed_dim=512)
                t0 = time.time()
                print(f"  Ingesting into 'video_index_{sport}_video_internvideo2_{split}' ...")
                VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=noop_embed,
                    show_progress=True,
                    use_async=True,
                    insert_batch_size=4096,
                )
                print(f"  Ingested in {time.time() - t0:.1f}s.")
                os.makedirs(f"{video_persist}_video_internvideo2_{split}", exist_ok=True)
                with open(os.path.join(f"{video_persist}_video_internvideo2_{split}", "ingested.flag"), "w") as f:
                    f.write("done")
                if os.path.exists(video_entities):
                    shutil.copy2(video_entities, os.path.join(f"{video_persist}_video_internvideo2_{split}", "entities.json"))
            else:
                print(f"  WARNING: no video nodes found for {sport}")
                failures += 1
        else:
            print(f"  Index already populated — skipping.")

        # 3. Video captions (pre-computed M3 embeddings)
        print(f"\n--- Video captions ({sport}) ---")
        caption_pkl = os.path.join(embeds_base, "captions", sport, "all_nodes_embedded.pkl")
        caption_entities = os.path.join(embeds_base, "captions", sport, "entities.json")
        caption_persist = f"{video_persist}_caption_m3_{split}"
        if not ingest_precomputed_nodes(
            nodes_pkl=caption_pkl,
            index_name=f"video_index_{sport}_caption_m3_{split}",
            es_url=es_url,
            embed_dim=1024,
            persist_dir=caption_persist,
            flag_name="ingested.flag",
            entities_src=caption_entities,
            entities_dst_name="entities.json",
        ):
            failures += 1

    elapsed = time.time() - overall_t0
    print(f"\n{'='*60}")
    print(f"  All done ({elapsed / 60:.1f} min)")
    if failures:
        print(f"  {failures} index(es) failed — check output above.")
    print(f"{'='*60}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
