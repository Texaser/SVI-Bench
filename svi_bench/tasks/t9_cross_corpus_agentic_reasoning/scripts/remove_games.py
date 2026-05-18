"""
Remove specific games from Elasticsearch indices by game_id.

Deletes all nodes (document + video) associated with the given game IDs
from all pipeline Elasticsearch indices.

Usage:
    # Remove specific games (interactive confirmation)
    python remove_games.py --game_ids 400090 400102 401220

    # Remove games listed in a file (one game_id per line)
    python remove_games.py --game_ids_file invalid_games.txt

    # Target a specific split
    python remove_games.py --game_ids 400090 --split test

    # Skip confirmation
    python remove_games.py --game_ids 400090 --force

    # Dry run (show what would be deleted without deleting)
    python remove_games.py --game_ids 400090 --dry_run
"""

import argparse
import os
import sys

from elasticsearch import Elasticsearch

# Index name templates — must match pipeline conventions:
#   document_tools.py:64  -> document_index_{emb_model}_{split}
#   video_tools.py:128    -> video_index_{emb_source}_{emb_model}_{split}
INDEX_TEMPLATES = [
    "document_index_m3_{split}",
    "video_index_caption_m3_{split}",
    "video_index_video_internvideo2_{split}",
]


def count_game_docs(es, index_name, game_ids):
    """Count documents matching the given game_ids in an index."""
    body = {
        "query": {"terms": {"metadata.game_id.keyword": game_ids}},
    }
    result = es.count(index=index_name, body=body)
    return result.get("count", 0)


def delete_game_docs(es, index_name, game_ids):
    """Delete all documents matching the given game_ids from an index."""
    body = {
        "query": {"terms": {"metadata.game_id.keyword": game_ids}},
    }
    result = es.delete_by_query(index=index_name, body=body, refresh=True)
    return result.get("deleted", 0), result.get("failures", [])


def main():
    parser = argparse.ArgumentParser(
        description="Remove specific games from Elasticsearch indices by game_id."
    )
    parser.add_argument("--game_ids", nargs="+", help="Game IDs to remove")
    parser.add_argument("--game_ids_file", help="File with game IDs (one per line)")
    parser.add_argument("--split", default="test", help="Split name (default: test)")
    parser.add_argument("--url", default=os.environ.get("T9_ES_URL", "http://localhost:9200"), help="Elasticsearch URL")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()

    # Collect game IDs
    game_ids = []
    if args.game_ids:
        game_ids.extend(args.game_ids)
    if args.game_ids_file:
        with open(args.game_ids_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    game_ids.append(line)

    if not game_ids:
        print("Error: No game IDs provided. Use --game_ids or --game_ids_file.")
        sys.exit(1)

    game_ids = sorted(set(game_ids))
    print(f"Games to remove: {len(game_ids)}")
    if len(game_ids) <= 20:
        for gid in game_ids:
            print(f"  {gid}")
    else:
        for gid in game_ids[:10]:
            print(f"  {gid}")
        print(f"  ... and {len(game_ids) - 10} more")

    # Connect
    try:
        es = Elasticsearch(args.url)
        if not es.ping():
            print(f"Error: Could not connect to Elasticsearch at {args.url}")
            sys.exit(1)
        print(f"\nConnected to Elasticsearch at {args.url}")
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        sys.exit(1)

    # Resolve index names for the given split
    indices = [t.format(split=args.split) for t in INDEX_TEMPLATES]

    # Count affected documents per index
    print(f"\nSplit: {args.split}")
    print(f"{'Index':<45} {'Matching Docs':>15}")
    print("-" * 62)

    total_docs = 0
    active_indices = []
    for index_name in indices:
        if not es.indices.exists(index=index_name):
            print(f"{index_name:<45} {'(not found)':>15}")
            continue
        count = count_game_docs(es, index_name, game_ids)
        total_docs += count
        if count > 0:
            active_indices.append((index_name, count))
        print(f"{index_name:<45} {count:>15}")

    print("-" * 62)
    print(f"{'Total':<45} {total_docs:>15}")

    if total_docs == 0:
        print("\nNo matching documents found. Nothing to delete.")
        return

    if args.dry_run:
        print("\n[DRY RUN] No documents were deleted.")
        return

    # Confirm
    if not args.force:
        response = input(f"\nDelete {total_docs} documents across {len(active_indices)} indices? (y/N): ").strip().lower()
        if response != "y":
            print("Operation cancelled.")
            return

    # Delete
    print("\nDeleting...")
    for index_name, expected in active_indices:
        deleted, failures = delete_game_docs(es, index_name, game_ids)
        if failures:
            print(f"  [WARN] {index_name}: deleted {deleted}, failures: {len(failures)}")
        else:
            print(f"  [OK]   {index_name}: deleted {deleted} documents")

    print("\nDone.")


if __name__ == "__main__":
    main()
