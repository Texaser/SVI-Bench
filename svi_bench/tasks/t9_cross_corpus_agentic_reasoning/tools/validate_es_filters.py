#!/usr/bin/env python
"""
Validate ES metadata filters for document and video indices.
Checks actual field mappings and tests each filter type against live data.

Usage:
    python tools/validate_es_filters.py [--es-url URL] [--doc-index INDEX] [--video-index INDEX]
"""

import argparse
import json
import os
import sys
from elasticsearch import Elasticsearch


def get_mapping(es, index):
    """Get the mapping for metadata fields."""
    mapping = es.indices.get_mapping(index=index)
    props = mapping[index]["mappings"]["properties"].get("metadata", {}).get("properties", {})
    return props


def test_query(es, index, label, query_body, expect_hits=True):
    """Run a query and report results."""
    try:
        result = es.search(index=index, body=query_body, size=1)
        hits = result["hits"]["total"]["value"]
        status = "PASS" if (hits > 0) == expect_hits else "FAIL"
        sample = ""
        if hits > 0:
            meta = result["hits"]["hits"][0].get("_source", {}).get("metadata", {})
            # Show a compact sample
            compact = {}
            for k in ["game_id", "clip_id", "doc_type", "date", "season", "season_type",
                       "period", "teams", "players", "video_window_start", "video_window_end",
                       "time_remaining_min", "time_remaining_max"]:
                v = meta.get(k)
                if v is not None:
                    # Truncate long lists
                    if isinstance(v, list) and len(v) > 2:
                        compact[k] = v[:2] + [f"...+{len(v)-2}"]
                    else:
                        compact[k] = v
            sample = f"  sample: {json.dumps(compact, default=str)}"
        print(f"  [{status}] {label}: {hits} hits (expected {'>=1' if expect_hits else '0'}){sample}")
        return status == "PASS"
    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e)[:200]
        print(f"  [ERROR] {label}: {err_type} - {err_msg}")
        return False


def print_mappings(es, index, fields):
    """Print ES field mappings for given fields."""
    props = get_mapping(es, index)
    for field in fields:
        if field in props:
            ftype = props[field].get("type", "?")
            has_keyword = "keyword" in props[field].get("fields", {})
            kw_str = " (.keyword)" if has_keyword else ""
            print(f"  metadata.{field}: {ftype}{kw_str}")
        else:
            print(f"  metadata.{field}: NOT IN MAPPING")
    return props


# =============================================================================
# Document Index Tests
# =============================================================================
def validate_document_index(es, index):
    """Validate all filters for the document index."""
    print(f"\n{'#' * 60}")
    print(f"# DOCUMENT INDEX: {index}")
    print(f"{'#' * 60}")

    if not es.indices.exists(index=index):
        print(f"  Index '{index}' does not exist — SKIPPING")
        return True

    total = es.count(index=index)["count"]
    print(f"  Total documents: {total}\n")

    # -- Mappings --
    print("  FIELD MAPPINGS:")
    doc_fields = ["game_id", "doc_type", "doc_id", "sport", "teams", "players",
                  "date", "season", "season_type", "home_team", "away_team"]
    props = print_mappings(es, index, doc_fields)
    print()

    # -- Sample data --
    print("  SAMPLE DATA:")
    for dt in ["espn_report", "game_stat_player", "game_stat_team",
               "season_stat_player", "season_stat_team"]:
        r = es.search(index=index, body={"query": {"term": {"metadata.doc_type.keyword": dt}}, "size": 1})
        n = r["hits"]["total"]["value"]
        if n > 0:
            meta = r["hits"]["hits"][0]["_source"].get("metadata", {})
            show = {k: meta.get(k) for k in doc_fields if meta.get(k) is not None}
            # Truncate long player lists for readability
            for lk in ["teams", "players"]:
                if lk in show and isinstance(show[lk], list) and len(show[lk]) > 3:
                    show[lk] = show[lk][:3] + [f"...+{len(show[lk])-3}"]
            print(f"    {dt} ({n} docs): {json.dumps(show, default=str)}")
        else:
            print(f"    {dt}: NO DOCUMENTS")
    print()

    # -- Get sample values --
    sample_r = es.search(index=index, body={
        "query": {"term": {"metadata.doc_type.keyword": "game_stat_player"}}, "size": 1})
    if sample_r["hits"]["total"]["value"] == 0:
        print("  No game_stat_player docs — cannot test filters")
        return False

    sm = sample_r["hits"]["hits"][0]["_source"]["metadata"]
    game_id = sm.get("game_id")
    date = sm.get("date")
    season = sm.get("season")
    season_type = sm.get("season_type")
    teams = sm.get("teams", [])
    players = sm.get("players", [])

    print(f"  Test sample: game_id={game_id}, date={date}, season={season}, season_type={season_type}")
    print(f"    teams={teams[:2]}, players={players[:2]}\n")

    all_ok = True

    # -- doc_type --
    print("  -- doc_type (text+.keyword) --")
    all_ok &= test_query(es, index, "term .keyword",
        {"query": {"term": {"metadata.doc_type.keyword": "game_stat_player"}}})
    all_ok &= test_query(es, index, "terms .keyword (multiple)",
        {"query": {"terms": {"metadata.doc_type.keyword": ["game_stat_player", "game_stat_team"]}}})
    print()

    # -- game_id --
    print("  -- game_id (text+.keyword) --")
    if game_id:
        all_ok &= test_query(es, index, f"term .keyword ({game_id})",
            {"query": {"term": {"metadata.game_id.keyword": game_id}}})
    print()

    # -- teams --
    print("  -- teams (text+.keyword) --")
    if teams:
        t = teams[0]
        all_ok &= test_query(es, index, f"terms .keyword ('{t}')",
            {"query": {"terms": {"metadata.teams.keyword": [t]}}})
    print()

    # -- players --
    print("  -- players (text+.keyword) --")
    if players:
        p = players[0]
        all_ok &= test_query(es, index, f"terms .keyword ('{p}')",
            {"query": {"terms": {"metadata.players.keyword": [p]}}})
    print()

    # -- season (long, no .keyword) --
    print("  -- season (long) --")
    if season is not None:
        all_ok &= test_query(es, index, f"term int ({season})",
            {"query": {"term": {"metadata.season": int(season)}}})
        all_ok &= test_query(es, index, f"term .keyword (should be 0)",
            {"query": {"term": {"metadata.season.keyword": str(season)}}}, expect_hits=False)
    print()

    # -- date (date type, no .keyword) --
    print("  -- date (date type) --")
    if date:
        all_ok &= test_query(es, index, f"term no .keyword ('{date}')",
            {"query": {"term": {"metadata.date": date}}})
        all_ok &= test_query(es, index, f"term .keyword (should be 0)",
            {"query": {"term": {"metadata.date.keyword": date}}}, expect_hits=False)
        all_ok &= test_query(es, index, f"range no .keyword",
            {"query": {"range": {"metadata.date": {"gte": date, "lte": date}}}})
        all_ok &= test_query(es, index, f"range .keyword (should be 0)",
            {"query": {"range": {"metadata.date.keyword": {"gte": date, "lte": date}}}}, expect_hits=False)
        if len(str(date)) == 10:
            year = str(date)[:4]
            all_ok &= test_query(es, index, f"range full year ({year})",
                {"query": {"range": {"metadata.date": {"gte": f"{year}-01-01", "lte": f"{year}-12-31"}}}})
    print()

    # -- season_type (text+.keyword) --
    print("  -- season_type (text+.keyword) --")
    if season_type:
        all_ok &= test_query(es, index, f"term .keyword ('{season_type}')",
            {"query": {"term": {"metadata.season_type.keyword": season_type}}})
    print()

    # -- Combined --
    print("  -- Combined (realistic) --")
    if game_id and date:
        all_ok &= test_query(es, index, "game_stat_player + game_id + date",
            {"query": {"bool": {"must": [
                {"term": {"metadata.doc_type.keyword": "game_stat_player"}},
                {"term": {"metadata.game_id.keyword": game_id}},
                {"term": {"metadata.date": date}},
            ]}}})
    if season and teams:
        all_ok &= test_query(es, index, "season_stat_team + season + team",
            {"query": {"bool": {"must": [
                {"term": {"metadata.doc_type.keyword": "season_stat_team"}},
                {"term": {"metadata.season": int(season)}},
                {"terms": {"metadata.teams.keyword": [teams[0]]}},
            ]}}})
    if players and game_id:
        all_ok &= test_query(es, index, "game_stat_player + player + game_id",
            {"query": {"bool": {"must": [
                {"term": {"metadata.doc_type.keyword": "game_stat_player"}},
                {"terms": {"metadata.players.keyword": [players[0]]}},
                {"term": {"metadata.game_id.keyword": game_id}},
            ]}}})
    print()

    return all_ok


# =============================================================================
# Video Index Tests
# =============================================================================
def validate_video_index(es, index):
    """Validate all filters for the video index."""
    print(f"\n{'#' * 60}")
    print(f"# VIDEO INDEX: {index}")
    print(f"{'#' * 60}")

    if not es.indices.exists(index=index):
        print(f"  Index '{index}' does not exist — SKIPPING")
        return True

    total = es.count(index=index)["count"]
    print(f"  Total documents: {total}\n")

    # -- Mappings --
    print("  FIELD MAPPINGS:")
    video_fields = ["game_id", "clip_id", "sport", "period", "teams", "players",
                    "video_window_start", "video_window_end",
                    "time_remaining_min", "time_remaining_max",
                    "time_elapsed_min", "time_elapsed_max"]
    props = print_mappings(es, index, video_fields)
    print()

    # -- Sample data --
    print("  SAMPLE DATA:")
    r = es.search(index=index, body={"query": {"match_all": {}}, "size": 1})
    if r["hits"]["total"]["value"] > 0:
        meta = r["hits"]["hits"][0]["_source"].get("metadata", {})
        show = {k: meta.get(k) for k in video_fields if meta.get(k) is not None}
        for lk in ["teams", "players"]:
            if lk in show and isinstance(show[lk], list) and len(show[lk]) > 3:
                show[lk] = show[lk][:3] + [f"...+{len(show[lk])-3}"]
        print(f"    {json.dumps(show, default=str)}")
    else:
        print("    NO DOCUMENTS")
        return False
    print()

    sm = meta
    game_id = sm.get("game_id")
    clip_id = sm.get("clip_id")
    period = sm.get("period")
    teams = sm.get("teams", [])
    players = sm.get("players", [])
    vw_start = sm.get("video_window_start")
    vw_end = sm.get("video_window_end")
    tr_min = sm.get("time_remaining_min")
    tr_max = sm.get("time_remaining_max")

    print(f"  Test sample: game_id={game_id}, clip_id={clip_id}, period={period}")
    print(f"    vw=[{vw_start}, {vw_end}], tr=[{tr_min}, {tr_max}]")
    print(f"    teams={teams[:2]}, players={players[:2]}\n")

    all_ok = True

    # -- game_id --
    print("  -- game_id --")
    if game_id:
        all_ok &= test_query(es, index, f"term .keyword ({game_id})",
            {"query": {"term": {"metadata.game_id.keyword": game_id}}})
        all_ok &= test_query(es, index, f"terms .keyword",
            {"query": {"terms": {"metadata.game_id.keyword": [game_id]}}})
    print()

    # -- clip_id (video_ids filter maps to this) --
    print("  -- clip_id (video_ids filter) --")
    if clip_id:
        all_ok &= test_query(es, index, f"term .keyword ({clip_id})",
            {"query": {"term": {"metadata.clip_id.keyword": clip_id}}})
        all_ok &= test_query(es, index, f"terms .keyword",
            {"query": {"terms": {"metadata.clip_id.keyword": [clip_id]}}})
    else:
        print("  [SKIP] clip_id not in sample")
    print()

    # -- teams --
    print("  -- teams --")
    if teams:
        t = teams[0]
        all_ok &= test_query(es, index, f"terms .keyword ('{t}')",
            {"query": {"terms": {"metadata.teams.keyword": [t]}}})
    print()

    # -- players --
    print("  -- players --")
    if players:
        p = players[0]
        all_ok &= test_query(es, index, f"terms .keyword ('{p}')",
            {"query": {"terms": {"metadata.players.keyword": [p]}}})
    print()

    # -- period (integer, no .keyword) --
    print("  -- period (quarter filter) --")
    if period is not None:
        all_ok &= test_query(es, index, f"term int ({period})",
            {"query": {"term": {"metadata.period": int(period)}}})
        all_ok &= test_query(es, index, f"term .keyword (should be 0)",
            {"query": {"term": {"metadata.period.keyword": str(period)}}}, expect_hits=False)
    print()

    # -- temporal_boundary (video_window_start/end range) --
    print("  -- temporal_boundary (video_window_start/end) --")
    if vw_start is not None and vw_end is not None and vw_start >= 0 and vw_end >= 0:
        mid = (vw_start + vw_end) / 2
        # The filter logic: clip_start < query_end AND clip_end > query_start
        all_ok &= test_query(es, index, f"range overlap (mid={mid:.1f})",
            {"query": {"bool": {"must": [
                {"range": {"metadata.video_window_start": {"lt": mid + 1}}},
                {"range": {"metadata.video_window_end": {"gt": mid - 1}}},
            ]}}})
        # Exact clip boundaries should match itself
        all_ok &= test_query(es, index, f"range exact clip bounds [{vw_start:.1f}, {vw_end:.1f}]",
            {"query": {"bool": {"must": [
                {"range": {"metadata.video_window_start": {"lt": vw_end}}},
                {"range": {"metadata.video_window_end": {"gt": vw_start}}},
            ]}}})
        # Outside range should not match this clip (but might match others)
    print()

    # -- time_remaining (countdown range) --
    print("  -- time_remaining (countdown range) --")
    if tr_min is not None and tr_max is not None and tr_min >= 0 and tr_max >= 0:
        # Overlap: clip_min < query_high AND clip_max > query_low
        all_ok &= test_query(es, index, f"range overlap [{tr_min:.1f}, {tr_max:.1f}]",
            {"query": {"bool": {"must": [
                {"range": {"metadata.time_remaining_min": {"lt": tr_max}}},
                {"range": {"metadata.time_remaining_max": {"gt": tr_min}}},
            ]}}})
    else:
        print("  [SKIP] time_remaining values not valid")
    print()

    # -- Combined --
    print("  -- Combined (realistic) --")
    if game_id and teams:
        all_ok &= test_query(es, index, "game_id + team",
            {"query": {"bool": {"must": [
                {"term": {"metadata.game_id.keyword": game_id}},
                {"terms": {"metadata.teams.keyword": [teams[0]]}},
            ]}}})
    if game_id and period is not None:
        all_ok &= test_query(es, index, f"game_id + period ({period})",
            {"query": {"bool": {"must": [
                {"term": {"metadata.game_id.keyword": game_id}},
                {"term": {"metadata.period": int(period)}},
            ]}}})
    if clip_id and game_id:
        all_ok &= test_query(es, index, f"game_id + clip_id",
            {"query": {"bool": {"must": [
                {"term": {"metadata.game_id.keyword": game_id}},
                {"term": {"metadata.clip_id.keyword": clip_id}},
            ]}}})
    print()

    return all_ok


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Validate ES metadata filters for the T9 document and video indices.")
    parser.add_argument("--es-url", default=os.environ.get("T9_ES_URL", "http://localhost:9200"))
    parser.add_argument("--sport", default="basketball", choices=["basketball", "hockey", "soccer"],
                        help="Sport whose indices to validate. Default basketball.")
    parser.add_argument("--doc-index", default=None,
                        help="Document index name. Default: document_index_<sport>_m3_all.")
    parser.add_argument("--video-index", default=None,
                        help="Video index name. Default: video_index_<sport>_video_internvideo2_all.")
    args = parser.parse_args()

    # Build per-sport defaults if not overridden.
    if args.doc_index is None:
        args.doc_index = f"document_index_{args.sport}_m3_all"
    if args.video_index is None:
        args.video_index = f"video_index_{args.sport}_video_internvideo2_all"

    es = Elasticsearch(args.es_url, request_timeout=30)

    if not es.ping():
        print(f"Cannot connect to ES at {args.es_url}")
        sys.exit(1)
    print(f"Connected to ES at {args.es_url}\n")

    doc_ok = validate_document_index(es, args.doc_index)
    video_ok = validate_video_index(es, args.video_index)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Document index ({args.doc_index}): {'ALL PASS' if doc_ok else 'FAILURES'}")
    print(f"  Video index ({args.video_index}):    {'ALL PASS' if video_ok else 'FAILURES'}")
    print("=" * 60)

    sys.exit(0 if (doc_ok and video_ok) else 1)


if __name__ == "__main__":
    main()
