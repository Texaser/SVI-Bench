#!/usr/bin/env python3
"""Restore the T9 release Elasticsearch indices from $T9_ROOT/storage/.

The release ships each ES index as a directory under ``data/t9/storage/``
containing:
  - ``<index-name>_mapping.json`` — index mapping + settings (single JSON)
  - ``<index-name>_data.jsonl``   — one ``{"_id": ..., "_source": {...}}``
                                     per line

This script walks each subdirectory, creates the named index in the target
ES cluster from the mapping/settings, and bulk-loads the data via the
official Elasticsearch python client. No external tooling (e.g. elasticdump)
needed.

Usage:
    # Defaults: $T9_ROOT/storage  →  http://localhost:9200
    python restore_es.py
    python restore_es.py --t9-root /path/to/data/t9 --es-url http://localhost:9200
    python restore_es.py --only "document_index_basketball_m3_all"   # restore one

If an index already exists in the target cluster, it's skipped unless --force
is passed (which deletes + recreates).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def restore_index(client, name: str, dir: Path, force: bool, batch_size: int) -> int:
    """Restore one index from <dir>/<name>_mapping.json + <name>_data.jsonl."""
    mapping_path = dir / f"{name}_mapping.json"
    data_path    = dir / f"{name}_data.jsonl"
    if not mapping_path.exists():
        raise FileNotFoundError(f"missing mapping file: {mapping_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"missing data file: {data_path}")

    if client.indices.exists(index=name):
        if not force:
            print(f"  [{name}] already exists — skipping (use --force to overwrite)")
            return 0
        print(f"  [{name}] --force: deleting existing index")
        client.indices.delete(index=name)

    # The mapping file holds the full GET _mapping + GET _settings response,
    # keyed by index name (since elasticdump-style exports preserve the source
    # name).  Strip down to mappings + settings that we want to recreate.
    meta = json.loads(mapping_path.read_text())
    mappings = meta["mapping"][name]["mappings"]
    raw_settings = meta["settings"][name]["settings"]["index"]
    # When recreating, drop fields ES generates on its own.
    for k in ("creation_date", "uuid", "version", "provided_name"):
        raw_settings.pop(k, None)
    settings = {"index": raw_settings}

    print(f"  [{name}] creating index ...")
    client.indices.create(index=name, mappings=mappings, settings=settings)

    # Bulk insert. Stream to keep memory bounded.
    from elasticsearch import helpers

    total = count_lines(data_path)
    print(f"  [{name}] bulk-loading {total} docs (batch={batch_size}) ...")
    t0 = time.time()

    def actions():
        for row in iter_jsonl(data_path):
            yield {
                "_op_type": "index",
                "_index":   name,
                "_id":      row["_id"],
                "_source":  row["_source"],
            }

    success, _ = helpers.bulk(client, actions(), chunk_size=batch_size,
                              request_timeout=300, raise_on_error=True)
    print(f"  [{name}] inserted {success} docs in {time.time()-t0:.1f}s")

    # Refresh so the count is correct immediately.
    client.indices.refresh(index=name)
    final = client.count(index=name).body["count"]
    print(f"  [{name}] post-restore doc count: {final}")
    if final != total:
        print(f"  [{name}] WARNING: ingested {final} but file had {total} rows")
    return success


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t9-root", default=os.environ.get("T9_ROOT"),
                    help="Path to data/t9/. Default: $T9_ROOT env var.")
    ap.add_argument("--es-url",  default=os.environ.get("T9_ES_URL", "http://localhost:9200"))
    ap.add_argument("--only",    action="append", default=None,
                    help="Restore only this index name (can repeat). Default: all.")
    ap.add_argument("--force",   action="store_true",
                    help="If an index already exists, delete + recreate.")
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="Bulk-insert batch size (default 1000).")
    args = ap.parse_args()

    if not args.t9_root:
        print("ERROR: --t9-root not set (and $T9_ROOT empty).", file=sys.stderr)
        return 1
    storage = Path(args.t9_root) / "storage"
    if not storage.is_dir():
        print(f"ERROR: {storage} not found.", file=sys.stderr)
        return 1

    from elasticsearch import Elasticsearch
    client = Elasticsearch(args.es_url)
    if not client.ping():
        print(f"ERROR: cannot reach ES at {args.es_url}", file=sys.stderr)
        return 1

    targets = sorted(d for d in storage.iterdir() if d.is_dir())
    if args.only:
        wanted = set(args.only)
        targets = [d for d in targets if d.name in wanted]
        missing = wanted - {d.name for d in targets}
        if missing:
            print(f"WARNING: --only references missing dirs: {sorted(missing)}", file=sys.stderr)

    print(f"Restoring {len(targets)} indices from {storage} → {args.es_url}")
    total = 0
    for d in targets:
        print(f"\n--- {d.name} ---")
        total += restore_index(client, d.name, d, args.force, args.batch_size)

    print(f"\nDone. Total docs inserted: {total}")
    print("Verify with: curl -s {args.es_url}/_cat/indices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
