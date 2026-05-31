import os
import asyncio
import concurrent.futures
import json
import glob
import sys
import numpy as np
import torch
import datetime
import base64
import logging
from typing import List, Dict, Optional, Any, Union


def _run_coro_sync(coro):
    """Run an async coroutine synchronously, regardless of an outer event loop.

    asyncio.run() refuses to nest inside a running loop. If we're already
    inside one, delegate to a one-shot worker thread that creates its own loop.
    Otherwise, plain asyncio.run.
    """
    try:
        asyncio.get_running_loop()
        outer_loop_active = True
    except RuntimeError:
        outer_loop_active = False
    if outer_loop_active:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings, Document
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.elasticsearch import ElasticsearchStore, AsyncDenseVectorStrategy
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from openai import AsyncOpenAI

try:
    from decord import VideoReader, cpu
    import cv2
except ImportError:
    pass

from tools import entity_utils
from tools.embedding_utils import get_embedding_model

logger = logging.getLogger(__name__)


async def _es_search_with_retry(es_client, index_name, body, max_retries=3, base_delay=1.0):
    """Execute ES search with retry on transient connection failures."""
    for attempt in range(max_retries + 1):
        try:
            return await es_client.search(index=index_name, body=body)
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"ES search failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)


_VIDEO_INDEX = None
_VIDEO_INDEX_BM25 = None
_DATA_METADATA = None
_CANONICAL_TEAMS = set()
_CANONICAL_PLAYERS = set()


def parse_time_str(time_str: str) -> float:
    """Converts 'MM:SS' string to seconds (float). Returns -1.0 on failure/empty."""
    try:
        if not time_str or ":" not in time_str:
            return -1.0
        parts = time_str.split(":")
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    except Exception as e:
        return -1.0


def get_clip_path(clip_id: str, data_metadata: Dict = None) -> str:
    """
    Returns the absolute path for a given clip_id.
    """
    if os.path.exists(clip_id):
        return clip_id
    
    # Use global metadata if not provided
    if data_metadata is None:
        data_metadata = _DATA_METADATA
    
    if not data_metadata:
        raise ValueError("No data metadata available")
    
    # clip_id format is {game_id}_{window_idx}
    parts = clip_id.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid clip_id format: {clip_id}")
    
    game_id = parts[0]
    
    if game_id not in data_metadata:
        raise ValueError(f"Game {game_id} not found in metadata")
    
    clip_paths_json = data_metadata[game_id].get("clip_paths")
    if not clip_paths_json or not os.path.exists(clip_paths_json):
        raise ValueError(f"Clip paths file not found for game: {game_id}")
    
    with open(clip_paths_json, 'r') as f:
        data = json.load(f)
        path = data.get(clip_id, "")
        if not path:
            return ""
        # Paths may be absolute or relative to the clips/ directory; handle both.
        if os.path.isabs(path):
            return path
        return os.path.join(os.path.dirname(clip_paths_json), path)


def _es_index_populated(es_url: str, index_name: str) -> bool:
    """Return True if the ES index exists and has at least one document.

    Used to avoid re-ingesting when the cluster already has data, even if
    the per-persist-dir flag file is missing. The flag-file path remains
    the secondary signal (kept for offline / no-ES rebuild scenarios).
    """
    try:
        from elasticsearch import Elasticsearch
        client = Elasticsearch(es_url)
        try:
            if not client.indices.exists(index=index_name):
                return False
            return client.count(index=index_name).get('count', 0) > 0
        finally:
            try: client.close()
            except Exception: pass
    except Exception as e:
        print(f"Warning: could not probe ES index '{index_name}': {e}. Falling back to flag-file check.")
        return False


def init_video_database(persist_dir: str, data_metadata: Dict,
                        clip_embeddings_base_path: str = None, model_config: Dict = None,
                        embedding_source: str = "video", es_url: str = "http://localhost:9200",
                        split: str = "all", sport: str = None):
    """
    Initializes the video database using either:
    1. 'video': Custom pre-computed visual embeddings (from .npy files).
    2. 'caption': Text embeddings (M3) generated from metadata captions.

    The persist_dir is automatically suffixed with the source to prevent collisions.
    """
    global _VIDEO_INDEX, _VIDEO_INDEX_BM25, _DATA_METADATA, _CANONICAL_TEAMS, _CANONICAL_PLAYERS

    # Store metadata globally
    _DATA_METADATA = data_metadata

    # 1. Embedding Model Setup
    assert model_config is not None and "embedding_model" in model_config, "model_config['embedding_model'] must be provided."
    embedding_model_name = model_config["embedding_model"]

    # Adjust persist path based on source, model, and split
    persist_dir = f"{persist_dir}_{embedding_source}_{embedding_model_name}_{split}"

    print(f"Initializing Video DB (Source: {embedding_source}) with model: {embedding_model_name} (split={split})")

    embed_model = get_embedding_model(embedding_model_name, model_config)

    # Detect Dimension
    try:
        dummy_emb = embed_model.get_text_embedding("test")
        embed_dim = len(dummy_emb)
    except Exception:
        embed_dim = 1024

    # 2. ES Store Setup
    print(f"Initializing Elasticsearch Store for Video at {es_url}...")
    if sport:
        index_name = f"video_index_{sport}_{embedding_source}_{embedding_model_name}_{split}"
    else:
        index_name = f"video_index_{embedding_source}_{embedding_model_name}_{split}"
    
    try:
        # A. Dense Vector Store
        from llama_index.vector_stores.elasticsearch import AsyncDenseVectorStrategy, AsyncBM25Strategy
        
        # Define longer timeout for ES operations
        es_client_kwargs = {"request_timeout": 300}

        es_store = ElasticsearchStore(
            es_url=es_url, 
            index_name=index_name,
            dim=embed_dim,
            retrieval_strategy=AsyncDenseVectorStrategy(hybrid=False, rrf=False),
            **es_client_kwargs
        )
        storage_context = StorageContext.from_defaults(vector_store=es_store)
        _VIDEO_INDEX = VectorStoreIndex.from_vector_store(es_store, storage_context=storage_context,
                                                          embed_model=embed_model)
                                                          
        # B. BM25 Store (Same Index, Different Strategy)
        es_store_bm25 = ElasticsearchStore(
            es_url=es_url, 
            index_name=index_name,
            dim=embed_dim,
            retrieval_strategy=AsyncBM25Strategy(),
            **es_client_kwargs
        )
        storage_context_bm25 = StorageContext.from_defaults(vector_store=es_store_bm25)
        _VIDEO_INDEX_BM25 = VectorStoreIndex.from_vector_store(es_store_bm25, storage_context=storage_context_bm25,
                                                               embed_model=embed_model)
        
        if not _es_index_populated(es_url, index_name):
            raise RuntimeError(
                f"Elasticsearch index '{index_name}' is empty. "
                f"Run 'python3 scripts/ingest.py' to populate it before starting the agent."
            )
        print(f"Using existing Elasticsearch index: {index_name}")
        _load_entities(persist_dir)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise Exception(f"Error initializing Video DB: {e}")


def _load_video_nodes_visual(data_metadata: Dict, clip_embeddings_base_path: str):
    """Loads nodes for Visual Semantic Search (using pre-computed embeddings)."""
    nodes = []
    
    print(f"Loading visual nodes for {len(data_metadata)} games...")
    
    # Pre-collect tasks to parallelize loading
    tasks = []
    
    for game_id, game_data in data_metadata.items():
        sport = game_data.get('sport', 'unknown')
        clips_metadata_path = game_data.get("clips_metadata")
        clip_paths_json = game_data.get("clip_paths")
        
        if not clips_metadata_path or not os.path.exists(clips_metadata_path):
            continue
        
        # Clip embeddings are in dataset-specific subdirectory
        clip_embeddings_path = os.path.join(clip_embeddings_base_path, sport)

        # Load clip paths mapping (fast enough to do main thread or per game)
        clip_paths_map = {}
        if clip_paths_json and os.path.exists(clip_paths_json):
            with open(clip_paths_json, 'r') as f:
                clip_paths_map = json.load(f)

        # Load embed-bucket mapping for sharded layout
        # ({clip_embeddings_path}/_bucket_mapping.json maps "<clip_id>.npy" → "00".."99"
        # so we can resolve the bucket directory at runtime). Returns None for a
        # flat (unsharded) layout — caller falls back to the flat lookup.
        bucket_map = None
        bm_path = os.path.join(clip_embeddings_path, "_bucket_mapping.json")
        if os.path.exists(bm_path):
            with open(bm_path, 'r') as f:
                bucket_map = json.load(f)

        if not os.path.exists(clips_metadata_path):
            continue

        tasks.append((game_id, sport, clips_metadata_path, clip_paths_map, clip_embeddings_path, bucket_map))

    # Worker function
    def process_game(task):
        game_id, sport, clips_metadata_path, clip_paths_map, clip_embeddings_path, bucket_map = task
        game_nodes = []
        try:
            with open(clips_metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            windows = data.get("windows", {})

            for window_key, window_data in windows.items():
                clip_id = window_data.get("window_id")

                emb_filename = f"{clip_id}.npy"
                if bucket_map is not None:
                    bucket = bucket_map.get(emb_filename)
                    if bucket is None:
                        continue
                    emb_path = os.path.join(clip_embeddings_path, bucket, emb_filename)
                else:
                    emb_path = os.path.join(clip_embeddings_path, emb_filename)

                if not os.path.exists(emb_path):
                    continue

                embedding = np.load(emb_path).tolist()
                clip_path = clip_paths_map.get(clip_id, "")


                meta = _extract_metadata(window_data, game_id, clip_id, 
                                        clip_path, data, sport)
                
                events = window_data.get("metadata", {}).get("events", {})
                captions = [evt.get("caption", "") for evt in events.values() if evt.get("caption")]
                text_content = " ".join(captions) if captions else "video clip"
                
                node = TextNode(
                    text=text_content,
                    id_=clip_id,
                    embedding=embedding,
                    metadata=meta
                )
                node.excluded_embed_metadata_keys.extend(meta.keys())
                node.excluded_llm_metadata_keys.extend(["events_json"])
                game_nodes.append(node)
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            
        return game_nodes

    # Execute in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_game, t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Clips"):
            nodes.extend(future.result())
            
    return nodes


def _load_video_nodes_caption(data_metadata: Dict):
    """Loads nodes for Caption Search."""
    nodes = []
    
    print(f"Loading caption nodes for {len(data_metadata)} games...")
    
    for game_id, game_data in data_metadata.items():
        sport = game_data.get('sport', 'unknown')
        clips_metadata_path = game_data.get("clips_metadata")
        clip_paths_json = game_data.get("clip_paths")
        
        if not clips_metadata_path or not os.path.exists(clips_metadata_path):
            continue
        
        # Load clip paths mapping
        clip_paths_map = {}
        if clip_paths_json and os.path.exists(clip_paths_json):
            with open(clip_paths_json, 'r') as f:
                clip_paths_map = json.load(f)
        
        try:
            with open(clips_metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            windows = data.get("windows", {})
            
            for window_key, window_data in windows.items():
                clip_id = window_data.get("window_id")
                clip_path = clip_paths_map.get(clip_id, "")
                
                meta = _extract_metadata(window_data, game_id, clip_id,
                                        clip_path, data, sport)
                
                events = window_data.get("metadata", {}).get("events", {})
                captions = [evt.get("caption", "") for evt in events.values() if evt.get("caption")]
                
                if not captions:
                    captions = []
                
                text_content = " ".join(captions)
                node = TextNode(
                    text=text_content,
                    id_=clip_id,
                    metadata=meta
                )
                node.excluded_embed_metadata_keys.extend(["events_json", "clip_path", "teams", "players"])
                node.excluded_llm_metadata_keys.extend(["events_json"])
                nodes.append(node)
        except Exception as e:
            raise Exception(f"Failed to process clip {clip_id}: {str(e)}")             

    return nodes


def _extract_metadata(window_data, game_id, clip_id, clip_path, file_data, sport):
    """Helper to extract and normalize metadata."""
    m = window_data.get("metadata", {})
    
    teams = m.get("teams") or file_data.get("teams", [])
    players = m.get("players") or file_data.get("players", [])
    
    if isinstance(teams, dict): teams = list(teams.values())
    if isinstance(players, dict): players = list(players.values())
    if not isinstance(teams, list): teams = []
    if not isinstance(players, list): players = []
    
    for t in teams: _CANONICAL_TEAMS.add(t)
    for p in players: _CANONICAL_PLAYERS.add(p)
    
    clock = m.get("game_clock_window", {})
    # Use -1.0 as default for missing/invalid times
    t_start = parse_time_str(clock.get("start_remaining"))
    t_end = parse_time_str(clock.get("end_remaining"))
    t_start_el = parse_time_str(clock.get("start_elapsed"))
    t_end_el = parse_time_str(clock.get("end_elapsed"))
    
    return {
        "game_id": game_id,
        "sport": sport,
        "clip_id": clip_id,
        "clip_path": clip_path,
        "period": int(window_data.get("period", 0)),
        "teams": teams,
        "players": players,
        "time_remaining_max": max(t_start, t_end),
        "time_remaining_min": min(t_start, t_end),
        "time_elapsed_max": max(t_start_el, t_end_el),
        "time_elapsed_min": min(t_start_el, t_end_el),
        "video_window_start": window_data.get("video_window", {}).get("start", -1.0),
        "video_window_end": window_data.get("video_window", {}).get("end", -1.0),
        "events_json": json.dumps(m.get("events", {}))
    }


def _save_entities(persist_dir):
    p = os.path.join(persist_dir, "entities.json")
    with open(p, 'w') as f:
        json.dump({"teams": list(_CANONICAL_TEAMS), "players": list(_CANONICAL_PLAYERS)}, f)


def _load_entities(persist_dir):
    p = os.path.join(persist_dir, "entities.json")
    if os.path.exists(p):
        with open(p, 'r') as f:
            d = json.load(f)
            _CANONICAL_TEAMS.update(d.get("teams", []))
            _CANONICAL_PLAYERS.update(d.get("players", []))


def _execute_search(query: str, metadata_filters: Dict, top_k: int = 5,
                    score_threshold: float = 0.0, embedding_source: str = "video",
                    sport: str = None, data_metadata: Dict = None):
    """
    Unified execution for video search.
    """
    assert _VIDEO_INDEX is not None, "Video Index not initialized."

    # Build ES filters
    es_filter_list = []

    # Sport filter (injected externally, not from agent)
    if sport:
        es_filter_list.append({"term": {"metadata.sport.keyword": sport}})

    if metadata_filters:
        key_map = {
            "quarter": "period",
            "half": "period",
            "game_ids": "game_id",
            "video_ids": "clip_id",
            "game_id": "game_id",
            "video_id": "clip_id"
        }
        for key, val in metadata_filters.items():
            target_key = key_map.get(key, key)
            
            if key in ["teams", "players"]:
                 vals = val if isinstance(val, list) else [val]
                 if key == "teams":
                     vals = [entity_utils.normalize_team_name(t, sport=sport) for t in vals]
                 global_set = _CANONICAL_TEAMS if key == "teams" else _CANONICAL_PLAYERS
                 resolved = entity_utils.resolve_entities(vals, global_set)

                 if resolved:
                     es_filter_list.append({"terms": {f"metadata.{key}.keyword": resolved}})
                 else:
                     # Add to query if no match?
                     if query: query += " " + " ".join(vals)
            
            elif key in ["period", "quarter", "half"]:
                if isinstance(val, list):
                     es_filter_list.append({"terms": {f"metadata.{target_key}": val}})
                else:
                     es_filter_list.append({"term": {f"metadata.{target_key}": val}})
                     
            elif key in ["game_ids", "game_id", "video_ids", "video_id"]:
                 if isinstance(val, list):
                     es_filter_list.append({"terms": {f"metadata.{target_key}.keyword": val}})
                 else:
                     es_filter_list.append({"term": {f"metadata.{target_key}.keyword": val}})

            elif key == "temporal_boundary":
                # Expect "SS.SS-SS.SS" (Video Timestamps in seconds)
                if val and "-" in str(val):
                    parts = str(val).split("-")
                    try:
                        t1 = float(parts[0].strip())
                        t2 = float(parts[1].strip())
                    except ValueError:
                        t1 = -1.0
                        t2 = -1.0
                    
                    if t1 >= 0 and t2 >= 0:
                        # Check for overlap: clip_start < query_end AND clip_end > query_start
                        es_filter_list.append({"range": {"metadata.video_window_start": {"lt": t2}}})
                        es_filter_list.append({"range": {"metadata.video_window_end": {"gt": t1}}})

            elif key == "time_remaining":
                # Expect "MM:SS-MM:SS" (Countdown Time)
                if val and "-" in str(val):
                    parts = str(val).split("-")
                    t1_raw = parts[0].strip()
                    t2_raw = parts[1].strip()
                    
                    # Parse and determine low/high (since 10:00 > 08:00)
                    v1 = parse_time_str(t1_raw)
                    v2 = parse_time_str(t2_raw)
                    
                    if v1 >= 0 and v2 >= 0:
                        q_low = min(v1, v2)
                        q_high = max(v1, v2)
                        
                        # Overlap: clip_min < query_high AND clip_max > query_low
                        es_filter_list.append({"range": {"metadata.time_remaining_min": {"lt": q_high}}})
                        es_filter_list.append({"range": {"metadata.time_remaining_max": {"gt": q_low}}})


    vector_store_kwargs = {}
    if es_filter_list:
        vector_store_kwargs["es_filter"] = es_filter_list
        
    # Retrieval
    # If query is empty, use BM25 index which handles text-based matches better (or allows match all if supported)
    # Otherwise use Dense index.
    
    if not query:
        target_index = _VIDEO_INDEX_BM25
        # Text search mode often helps with sparse or empty queries in some stores
        mode = "text_search" 
    else:
        target_index = _VIDEO_INDEX
        mode = "default" 

    # Get raw Elasticsearch scores by querying directly instead of using retriever
    # The retriever normalizes scores so the top result always gets 1.0
    # We bypass this by accessing the vector store client directly
    
    search_query = query if query else ""
    
    # Get the vector store and ES client
    vector_store = target_index.vector_store
    es_client = vector_store.client
    index_name = vector_store.index_name
    
    # Get embedding for the query (only for dense mode)
    query_embedding = None
    if mode != "text_search":
        query_embedding = target_index._embed_model.get_query_embedding(search_query)
    
    # Build Elasticsearch query body directly to bypass llama_index normalization
    import asyncio
    
    async def get_raw_es_results():
        if query_embedding is not None:
            # Exact Dot Product search using script_score (Brute force across filtered set)
            # This ensures 100% recall for filtered queries.
            # We use constant_score to ensure the filter itself doesn't affect the ranking.
            body = {
                "query": {
                    "script_score": {
                        "query": {
                            "constant_score": {
                                "filter": {"bool": {"must": es_filter_list}} if es_filter_list else {"match_all": {}}
                            }
                        },
                        "script": {
                            "source": "dotProduct(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "_source": False,
                "size": top_k
            }
            
            """
            # PREVIOUS APPROXIMATE KNN (HNSW)
            # Dense vector search using kNN
            body = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": top_k
                },
                "fields": ["_id"],
                "_source": False
            }
            
            # Add filters if present
            if es_filter_list:
                body["knn"]["filter"] = {"bool": {"must": es_filter_list}}
            """
                
        else:
            # BM25 text search
            if search_query and search_query.strip():
                # Query provided - use match query
                body = {
                    "query": {
                        "bool": {
                            "must": [{"match": {"content": search_query}}]
                        }
                    },
                    "size": top_k
                }
            else:
                # No query - use match_all for filter-only search
                body = {
                    "query": {
                        "bool": {
                            "must": [{"match_all": {}}]
                        }
                    },
                    "size": top_k
                }
            
            if es_filter_list:
                body["query"]["bool"]["filter"] = es_filter_list
        
        # Execute the search (with retry for transient ES failures)
        result = await _es_search_with_retry(es_client, index_name, body)
        return result
    
    es_result = _run_coro_sync(get_raw_es_results())
    
    # Extract node IDs and RAW scores from Elasticsearch
    node_ids = []
    raw_scores = []
    for hit in es_result['hits']['hits']:
        node_ids.append(hit['_id'])
        raw_scores.append(hit['_score'])
    
    # Get the actual nodes from the vector store
    nodes = []
    if node_ids:
        from llama_index.core.schema import NodeWithScore
        fetched_nodes = vector_store.get_nodes(node_ids)
        
        # Build a map for fast lookup and re-alignment
        # LlamaIndex nodes have an id_ (or node_id) attribute
        node_map = {n.id_: n for n in fetched_nodes if n}
        
        for node_id, score in zip(node_ids, raw_scores):
            if node_id in node_map:
                nodes.append(NodeWithScore(node=node_map[node_id], score=score))
    
    # Process Results
    # Normalize scores: dense (dotProduct+1) → /2.0 for [0,1]; filter-only → null
    is_filter_only = query_embedding is None and (not search_query or not search_query.strip())

    results = []
    for n in nodes:
        if is_filter_only:
            score = None
        else:
            score = round(n.score / 2.0, 4)
            if score < score_threshold:
                continue

        meta = n.metadata
        period_key = {"basketball": "quarter", "hockey": "period", "soccer": "half"}.get(sport, "period")
        row = {
            "clip_id": meta.get("clip_id"),
            "sport": meta.get("sport"),
            "score": score,
            "game_metadata": {
                "game_id": meta.get("game_id"),
                period_key: meta.get("period"),
                "teams": meta.get("teams"),
                "players": meta.get("players")
            },
            "temporal_context": {
                "video_timestamps": f"{meta.get('video_window_start')} - {meta.get('video_window_end')}"
            },
        }
        results.append(row)

    final_results = results[:top_k]

    # Check for missing data when no results found with specific filters
    if not final_results and metadata_filters and data_metadata:
        game_ids = metadata_filters.get('game_ids', metadata_filters.get('game_id', []))
        if isinstance(game_ids, str):
            game_ids = [game_ids]

        for gid in game_ids:
            game_meta = data_metadata.get(str(gid), {})
            if not game_meta:
                final_results.append({"message": f"Game {gid} does not exist in the database."})
            elif 'clips_metadata' not in game_meta or 'clip_paths' not in game_meta:
                final_results.append({
                    "message": f"No video clips available for game {gid}."
                })

    return final_results


def search_videos(query: str, metadata_filters: Dict = None, top_k: int = 10,
                  embedding_source: str = "video", sport: str = None,
                  data_metadata: Dict = None) -> List[Dict]:
    """
    Performs semantic search for video clips.
    """
    return _execute_search(query, metadata_filters or {}, top_k=top_k,
                           embedding_source=embedding_source,
                           sport=sport, data_metadata=data_metadata)


def get_event_ids_for_clip(clip_id: str) -> List[str]:
    """
    Retrieves the list of event_ids associated with a specific clip_id.
    Reads directly from the clips_metadata.json for the corresponding game.
    """
    global _DATA_METADATA
    if not _DATA_METADATA:
        # If metadata is not loaded, we can't look it up easily without game_id
        # Assuming clip_id is {game_id}_{window_idx}
        return []

    try:
        parts = clip_id.split('_')
        if len(parts) < 2:
            return []
        
        game_id = parts[0]
        
        if game_id not in _DATA_METADATA:
            return []

        game_data = _DATA_METADATA[game_id]
        
        # We need to load the actual metadata json file
        # To avoid re-reading for every single clip, we could cache, but for now we read on demand
        # or rely on os cache. 
        # Ideally, this should be optimized if called frequently in a tight loop.
        # But search_videos returns top_k (small), so it's acceptable.
        
        clips_metadata_path = game_data.get("clips_metadata")
        if not clips_metadata_path or not os.path.exists(clips_metadata_path):
            return []

        with open(clips_metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        windows = data.get("windows", {})
        window_data = windows.get(clip_id)
        
        if not window_data:
            return []
            
        # event_ids are at the top level of the window object in the provided JSON schema
        # but the JSON example shows them as a list of integers.
        # We convert to string to match ground_truth format.
        event_ids = window_data.get("event_ids", [])
        return [str(eid) for eid in event_ids]

    except Exception as e:
        print(f"Error extracting event_ids for {clip_id}: {e}")
        return []


async def _get_oracle_text(clip_id: str) -> Optional[str]:
    """Retrieves caption text for a simplified clip_id using direct ES query."""
    if _VIDEO_INDEX is None: return None
    
    try:
        vs = _VIDEO_INDEX.storage_context.vector_store
        es_client = vs.client
        index_name = vs.index_name
        
        body = {
            "query": {"term": {"metadata.clip_id.keyword": clip_id}},
            "size": 1,
            "_source": ["content"]
        }
        
        es_res = await _es_search_with_retry(es_client, index_name, body)
        hits = es_res.get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"]["content"]
    except Exception as e:
        print(f"[Oracle Lookup] ES Error for {clip_id}: {e}")
        
    return None


def _log_prompt(experiment_path: str, tool_name: str, identifier: str, prompt: Union[str, List]):
    """Logs the full prompt to a file in the experiment directory."""
    if not experiment_path:
        return
    
    log_dir = os.path.join(experiment_path, "prompts")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Sanitize identifier: use basename to get unique part, sanitize chars, keep last 64 chars
    base_ident = os.path.basename(identifier)
    safe_ident = "".join([c if c.isalnum() else "_" for c in base_ident])[-64:]
    filename = f"qa_{timestamp}_{tool_name}_{safe_ident}.txt"
    filepath = os.path.join(log_dir, filename)
    
    try:
        with open(filepath, "w") as f:
            if isinstance(prompt, str):
                f.write(prompt)
            elif isinstance(prompt, list):
                # Formatted logging for chat messages
                for msg in prompt:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        f.write(f"[{msg['role'].upper()}]\n")
                        content = msg["content"]
                        if isinstance(content, str):
                            f.write(f"{content}\n\n")
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    if "text" in block:
                                        f.write(f"{block.get('text', '')}\n")
                                    elif "image_url" in block:
                                        f.write("[Image Content]\n")
                        f.write("-" * 40 + "\n")
                    else:
                         f.write(json.dumps(msg, indent=2) + "\n")
    except Exception as e:
        print(f"Error logging prompt to {filepath}: {e}")

def _encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def _extract_frames_decord(video_path: str, num_frames: int, resize: int = 336) -> List[str]:
    """
    Extracts `num_frames` uniformly from a video using decord.
    Resizes frames so that the long side is `resize` pixels (preserving aspect ratio).
    Returns list of base64 encoded images (JPEGs).
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            return []
            
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        
        # Get frames (RGB)
        frames = vr.get_batch(indices).asnumpy()
        
        encoded_frames = []
        for frame in frames: 
            # Decord returns RGB, but cv2.imencode expects BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if resize:
                h, w = frame_bgr.shape[:2]
                scale = resize / max(h, w)
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame_bgr = cv2.resize(frame_bgr, (new_w, new_h))
            
            # Encode to JPEG
            success, buffer = cv2.imencode('.jpg', frame_bgr)
            if success:
                b64_str = base64.b64encode(buffer).decode('utf-8')
                encoded_frames.append(b64_str)
                
        return encoded_frames

    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []


async def _process_single_video_qa(client: AsyncOpenAI, model: str, clip_id: str, query: str, prompt_template: str, **kwargs) -> Dict:
    """Processes a single video QA task (Oracle or VLM) asynchronously."""
    try:
        messages = []
        
        # 1. Oracle Text Injection
        if "{{oracle_text}}" in prompt_template:
             oracle_text = await _get_oracle_text(clip_id)
             if not oracle_text:
                 oracle_text = "No text data found for this clip."
                 
             max_content_tokens = kwargs.get("max_content_tokens", 32000)
             prompt_content = prompt_template.replace("{{oracle_text}}", oracle_text[:max_content_tokens]).replace("{{query}}", query)
             messages = [{"role": "user", "content": prompt_content}]
             
        # 2. Direct Video/Image Path Handling
        else:
             target_path = get_clip_path(clip_id, kwargs.get('video_data_path'))

             if not target_path or not os.path.exists(target_path):
                  return {"clip_id": clip_id, "error": f"Clip {clip_id} not found."}
             prompt_content = prompt_template.replace("{{query}}", query)
             content_blocks = [{"type": "text", "text": prompt_content}]
             
             ext = os.path.splitext(target_path)[1].lower()
             
             # Image Handling
             if ext in [".jpg", ".png", ".jpeg"]:
                  base64_image = _encode_image(target_path)
                  content_blocks.append({
                       "type": "image_url",
                       "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                  })
                  
             # Video Handling
             elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                  num_frames = kwargs.get("video_num_frames")
                  assert num_frames is not None, "video_num_frames must be provided in config"
                  
                  resize_dim = kwargs.get("video_resize")
                  assert resize_dim is not None, "video_resize must be provided in config"

                  frames_b64 = _extract_frames_decord(target_path, num_frames, resize=resize_dim)
                  
                  if not frames_b64:
                      return {"clip_id": clip_id, "error": f"Failed to extract frames from {target_path}"}
                      
                  for f_b64 in frames_b64:
                       content_blocks.append({
                           "type": "image_url",
                           "image_url": {"url": f"data:image/jpeg;base64,{f_b64}"}
                       })
             
             messages = [{"role": "user", "content": content_blocks}]
        

        # 3. Request
        _log_prompt(kwargs.get("experiment_path"), "video_qa", clip_id, messages)
        
        llm_config = kwargs.get("llm_config", {})
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=llm_config.get("temperature", 1.0),
            max_tokens=llm_config.get("max_tokens", llm_config.get("max_output_tokens", 4096))
        )
        
        answer_raw = response.choices[0].message.content

        try:
            import json
            import re

            # 1. Strip <think>...</think> blocks (handles tagged reasoning)
            answer_clean = re.sub(r'<think>.*?</think>', '', answer_raw, flags=re.DOTALL).strip()
            # Handle unclosed <think> tag (everything from <think> to end)
            answer_clean = re.sub(r'<think>.*', '', answer_clean, flags=re.DOTALL).strip()

            # 2. Extract content from <answer> tags if present
            if "<answer>" in answer_clean:
                answer_clean = answer_clean.split("<answer>")[1].split("</answer>")[0].strip()

            # 3. Extract JSON code block
            if "```json" in answer_clean:
                answer_clean = answer_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in answer_clean:
                answer_clean = answer_clean.split("```")[1].strip()

            # 4. Find the LAST complete JSON object via brace matching
            #    (reasoning text with braces often precedes the actual answer JSON)
            last_brace = answer_clean.rfind("}")
            if last_brace != -1:
                depth = 0
                start = last_brace
                for i in range(last_brace, -1, -1):
                    if answer_clean[i] == '}':
                        depth += 1
                    elif answer_clean[i] == '{':
                        depth -= 1
                    if depth == 0:
                        start = i
                        break
                candidate = answer_clean[start:last_brace + 1]
                res = json.loads(candidate)
                res["clip_id"] = clip_id
                return res

            res = json.loads(answer_clean)
            res["clip_id"] = clip_id
            return res
        except (ValueError, TypeError):
            import re
            fallback = re.sub(r'<think>.*?</think>', '', answer_raw, flags=re.DOTALL).strip()
            fallback = re.sub(r'<think>.*', '', fallback, flags=re.DOTALL).strip()
            if "<answer>" in fallback:
                fallback = fallback.split("<answer>")[1].split("</answer>")[0].strip()
            return {
                "clip_id": clip_id,
                "answer": fallback,
                "confidence": 0.5,
                "evidence": "Model returned raw text."
            }

    except Exception as e:
        return {"clip_id": clip_id, "error": f"QA Failed: {str(e)}"}

def video_qa(video_ids: List[str], query: str, **kwargs) -> List[Dict]:
    """
    Analyzes list of video clips using Pixel/VLM.
    """
    llm_cfg = kwargs.get("llm_config", {})
    return _run_qa_batch(video_ids, query, llm_cfg, "video", **kwargs)

def video_qa_oracle(video_ids: List[str], query: str, **kwargs) -> List[Dict]:
    """
    Analyzes list of video clips using Text Oracle.
    """
    llm_cfg = kwargs.get("llm_config", {})
    return _run_qa_batch(video_ids, query, llm_cfg, "oracle", **kwargs)

def _run_qa_batch(video_ids, query, llm_cfg, mode, **kwargs):
    
    # Extract LLM configuration
    llm_api_base = llm_cfg.get("model_server")
    llm_api_key = llm_cfg.get("api_key", "EMPTY")
    model_name = llm_cfg.get("model")
    if not llm_api_base or not model_name:
        raise ValueError(
            "_run_qa_batch: llm_config is missing required keys "
            f"(model_server={llm_api_base!r}, model={model_name!r}). "
            "This is set by init_tools at agent startup — ensure the arch "
            "YAML defines a 'video_qa' / 'video_qa_oracle' tool block."
        )

    # Hyperparameters override
    # Pass 'qa' hyperparams down to process_single via kwargs

    
    # Load Prompt from config
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Try explicit 'prompt' from top-level tool config
    rel_path = llm_cfg.get('prompt')
    if not rel_path:
        # 2. Fallback to legacy dictionary or default paths
        config_prompts = llm_cfg.get('prompts', {})
        if mode == "oracle":
            rel_path = config_prompts.get("oracle", "prompts/tools/video_qa_oracle.txt")
        else:
            rel_path = config_prompts.get("video", "prompts/tools/video_qa.txt")

    p_path = os.path.join(base_dir, rel_path)
    if os.path.exists(p_path):
        with open(p_path, 'r') as f: template = f.read()
    else:
        # Hardcoded defaults if file missing
        if mode == "oracle":
            template = "Events: {{oracle_text}}\nQuestion: {{query}}\nAnswer in JSON."
        else:
            template = "Question: {{query}}\nAnswer in JSON."

    async def run_batch():
        client = AsyncOpenAI(api_key=llm_api_key, base_url=llm_api_base)
        tasks = []
        for video_id in video_ids:
            tasks.append(_process_single_video_qa(client, model_name, video_id, query, template, **kwargs))
        
        results = await asyncio.gather(*tasks)
        await client.close()
        return results
    
    try:
        return _run_coro_sync(run_batch())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [{"error": f"Batch execution failed: {str(e)}"}]
