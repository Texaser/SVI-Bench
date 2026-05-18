"""Remove the T9 Elasticsearch indices for one or more sports.

The 9 shipping indices (per services.md §1.bis) follow this naming:

    {kind}_index_{sport}_{source_or_emb}_all

Concretely:
    document_index_{sport}_m3_all
    video_index_{sport}_caption_m3_all
    video_index_{sport}_video_internvideo2_all

This script enumerates the indices for the requested sports and deletes
them.

Examples:
    # Dry-run on a single sport
    python scripts/remove_indices.py --sport basketball --list

    # Delete all 9 (interactive confirmation)
    python scripts/remove_indices.py --sport all

    # Skip confirmation
    python scripts/remove_indices.py --sport hockey --force
"""

import argparse
import os
import sys
from elasticsearch import Elasticsearch


SPORTS = ("basketball", "hockey", "soccer")


def index_names_for(sport: str) -> list[str]:
    return [
        f"document_index_{sport}_m3_all",
        f"video_index_{sport}_caption_m3_all",
        f"video_index_{sport}_video_internvideo2_all",
    ]


def main():
    parser = argparse.ArgumentParser(description="Remove T9 Elasticsearch indices for one or more sports.")
    parser.add_argument("--url", default=os.environ.get("T9_ES_URL", "http://localhost:9200"),
                        help="Elasticsearch URL (default: $T9_ES_URL or http://localhost:9200)")
    parser.add_argument("--sport", required=True, choices=list(SPORTS) + ["all"],
                        help="Sport(s) whose indices to remove. 'all' selects basketball+hockey+soccer.")
    parser.add_argument("--force", action="store_true", help="Skip the interactive confirmation prompt.")
    parser.add_argument("--list", action="store_true", help="List indices that would be removed and exit.")
    args = parser.parse_args()

    sports = list(SPORTS) if args.sport == "all" else [args.sport]
    target_indices = [name for s in sports for name in index_names_for(s)]

    try:
        es = Elasticsearch(args.url)
        if not es.ping():
            print(f"Error: Could not connect to Elasticsearch at {args.url}")
            sys.exit(1)
        print(f"Connected to Elasticsearch at {args.url}")

        indices_to_delete = [idx for idx in target_indices if es.indices.exists(index=idx)]

        if not indices_to_delete:
            print("No matching T9 indices found to delete.")
            return

        print("\nThe following indices will be REMOVED:")
        for idx in indices_to_delete:
            print(f"  - {idx}")

        if args.list:
            return

        if not args.force:
            response = input("\nAre you sure you want to delete these indices? (y/N): ").strip().lower()
            if response != 'y':
                print("Operation cancelled.")
                return

        print("\nDeleting indices...")
        for idx in indices_to_delete:
            try:
                response = es.indices.delete(index=idx)
                if response.get('acknowledged', False):
                    print(f"[OK] Deleted {idx}")
                else:
                    print(f"[FAILED] Could not delete {idx}: {response}")
            except Exception as e:
                print(f"[ERROR] Failed to delete {idx}: {e}")

        print("\nDone.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
