#!/usr/bin/env python3
"""Pre-populate Elasticsearch indices for T9 before running experiments.

Ingests game documents and video clip embeddings into Elasticsearch so that
the agent's search tools can query them. Run this once after downloading
and extracting the data; subsequent run_agent.py / run_batch.py starts will
detect the populated indices and skip re-ingestion.

Requires:
  - Elasticsearch running (see README for setup)
  - T9 data downloaded and extracted (metadata.json, game dirs, embeddings)

Usage:
  # Ingest all sports (default)
  python3 scripts/ingest.py

  # Ingest one sport
  python3 scripts/ingest.py --sport basketball

  # Custom ES host
  python3 scripts/ingest.py --es-url http://my-es-host:9200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import yaml

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

from _t9_root import require_t9_data_root
from run_agent import load_data_metadata, load_config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--es-url", default=None,
                   help="Elasticsearch URL (default: from config or http://localhost:9200)")
    p.add_argument("--t9-root", default=None,
                   help="T9 data root (default: T9_ROOT env var or auto-detected)")
    args = p.parse_args()

    # Resolve T9 root
    if args.t9_root:
        os.environ["T9_ROOT"] = args.t9_root
    t9_root = require_t9_data_root()
    print(f"T9_ROOT: {t9_root}")

    # Load configs
    paths_cfg = load_config(os.path.join(TASK_DIR, "configs/paths.yaml"))
    hyper_cfg = load_config(os.path.join(TASK_DIR, "configs/hyperparameters.yaml"))
    models_cfg = load_config(os.path.join(TASK_DIR, "configs/models.yaml"))

    # Resolve paths to absolute
    for key in ("data_base_path", "clip_embeddings_base_path",
                "video_persist_dir", "document_persist_dir"):
        val = paths_cfg.get(key)
        if val and not os.path.isabs(val):
            paths_cfg[key] = os.path.join(t9_root, val)

    # Determine sports
    sports = hyper_cfg.get("data", {}).get("enabled_datasets", [])
    if not sports:
        print("ERROR: no sports configured. Pass --sport or check hyperparameters.yaml")
        return 1
    print(f"Sports: {sports}")

    # ES URL
    es_url = (args.es_url
              or os.environ.get("T9_ES_URL")
              or hyper_cfg.get("elasticsearch", {}).get("url", "http://localhost:9200"))
    print(f"Elasticsearch: {es_url}")

    # Enabled sources
    enabled_sources = hyper_cfg.get("data", {}).get("enabled_sources", [])
    print(f"Enabled sources: {enabled_sources}")

    from tools import document_tools, video_tools

    split = "all"
    overall_t0 = time.time()

    for sport in sports:
        print(f"\n{'='*60}")
        print(f"  Ingesting: {sport}")
        print(f"{'='*60}")

        # Load metadata
        datasets_metadata = load_data_metadata(paths_cfg["data_base_path"], [sport])
        all_data_metadata = {}
        for dataset, data_metadata in datasets_metadata.items():
            for item_id, item_data in data_metadata.items():
                item_data["sport"] = dataset
                all_data_metadata[item_id] = item_data
        print(f"Loaded {len(all_data_metadata)} items for {sport}")

        # Document ingestion
        doc_emb_model = "m3"
        doc_persist_dir = f"{paths_cfg['document_persist_dir']}_{sport}_{doc_emb_model}_{split}"
        print(f"\n--- Documents ({sport}) ---")
        print(f"Persist dir: {doc_persist_dir}")
        t0 = time.time()
        document_tools.init_document_database(
            doc_persist_dir,
            all_data_metadata,
            enabled_sources=enabled_sources,
            model_config={"embedding_model": doc_emb_model},
            es_url=es_url,
            split=split,
            sport=sport,
        )
        print(f"Documents done ({time.time() - t0:.1f}s)")

        # Video ingestion (only if videos in enabled_sources)
        if "videos" in enabled_sources:
            video_emb_model = "internvideo2"
            video_emb_source = "video"
            video_persist_dir = f"{paths_cfg['video_persist_dir']}_{sport}"

            tool_model_cfg = {"embedding_model": video_emb_model}
            iv2_cfg = models_cfg.get("embedding_models", {}).get("internvideo2", {})
            tool_model_cfg.update(iv2_cfg)

            print(f"\n--- Videos ({sport}) ---")
            print(f"Persist dir: {video_persist_dir}")
            print(f"Embeddings: {paths_cfg['clip_embeddings_base_path']}")
            t0 = time.time()
            video_tools.init_video_database(
                persist_dir=video_persist_dir,
                data_metadata=all_data_metadata,
                clip_embeddings_base_path=paths_cfg["clip_embeddings_base_path"],
                model_config=tool_model_cfg,
                embedding_source=video_emb_source,
                es_url=es_url,
                split=split,
                sport=sport,
            )
            print(f"Videos done ({time.time() - t0:.1f}s)")

            # Caption-based video index
            caption_persist_dir = f"{paths_cfg['video_persist_dir']}_{sport}"
            print(f"\n--- Video captions ({sport}) ---")
            t0 = time.time()
            video_tools.init_video_database(
                persist_dir=caption_persist_dir,
                data_metadata=all_data_metadata,
                clip_embeddings_base_path=paths_cfg["clip_embeddings_base_path"],
                model_config={"embedding_model": "m3"},
                embedding_source="caption",
                es_url=es_url,
                split=split,
                sport=sport,
            )
            print(f"Video captions done ({time.time() - t0:.1f}s)")

    elapsed = time.time() - overall_t0
    print(f"\n{'='*60}")
    print(f"  All done ({elapsed / 60:.1f} min)")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
