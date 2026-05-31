import os
import glob
import json
import asyncio
import concurrent.futures
import logging
from typing import List, Dict, Optional, Any, Union


def _run_coro_sync(coro):
    """Run an async coroutine synchronously, regardless of an outer event loop.

    asyncio.run() refuses to nest inside a running loop. If we're already in
    one, delegate to a one-shot worker thread that creates its own loop.
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
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
# from llama_index.core.retrievers import QueryFusionRetriever # Removed library usage
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import Document, TextNode, NodeWithScore, IndexNode, QueryBundle
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from tools import entity_utils
from tools.embedding_utils import get_embedding_model



from llama_index.vector_stores.elasticsearch import ElasticsearchStore

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


# Global Instances (Singleton Pattern for efficiently reusing index across tool calls)
_INDEX = None
_INDEX_BM25 = None
# Global canonical sets for entities
_CANONICAL_TEAMS = set()
_CANONICAL_PLAYERS = set()

# Mapping user-defined data source keys to actual file basenames (without .txt)
SOURCE_TO_FILE_MAP = {
    "espn_report": ["espn_report"],
    "game_statistics": ["game_stat_player", "game_stat_team"],
    "statistics": ["game_stat_player", "game_stat_team"],
    "season_statistics": ["season_stat_player", "season_stat_team"]
}

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


def init_document_database(persist_dir: str, data_metadata: Dict, enabled_sources: List[str] = None, model_config: Dict = None, es_url: str = "http://localhost:9200", split: str = "all", sport: str = None):
    """Initializes the document database and global retriever instances."""
    global _INDEX, _INDEX_BM25

    # 1. Embedding Configuration
    embedding_model_name = "m3" # Default
    if model_config and "embedding_model" in model_config:
        embedding_model_name = model_config["embedding_model"]

    print(f"Using embedding model: {embedding_model_name} for Document Search.")

    # Reuse factory from embedding_utils
    embed_model = get_embedding_model(embedding_model_name, model_config)

    # Detect Dim
    try:
        dummy_emb = embed_model.get_text_embedding("test")
        embed_dim = len(dummy_emb)
        print(f"Detected embedding dimension: {embed_dim}")
    except Exception:
        print(f"Could not detect dim, using default 1024 for {embedding_model_name}")
        embed_dim = 1024
        if "internvideo" in embedding_model_name.lower(): embed_dim = 512


    # 2. Load or Build Index
    # Use Elasticsearch
    print(f"Initializing Elasticsearch Store at {es_url}...")
    try:
        from llama_index.vector_stores.elasticsearch import AsyncDenseVectorStrategy, AsyncBM25Strategy

        if sport:
            index_name = f"document_index_{sport}_{embedding_model_name}_{split}"
        else:
            index_name = f"document_index_{embedding_model_name}_{split}"
        print(f"Using Elasticsearch Index: {index_name}")
        
        es_store = ElasticsearchStore(
            es_url=es_url, 
            index_name=index_name,
            dim=embed_dim,
            retrieval_strategy=AsyncDenseVectorStrategy(hybrid=False, rrf=False)
        )
        storage_context = StorageContext.from_defaults(vector_store=es_store)
        
        # Check if index is empty by trying to load. 
        # VectorStoreIndex.from_vector_store will always succeed but might be empty.
        # But crucially, we must assign it to the global _INDEX
        _INDEX = VectorStoreIndex.from_vector_store(es_store, storage_context=storage_context, embed_model=embed_model)
        
        # Initialize BM25 Store (Same Index)
        print("Initializing Elasticsearch BM25 Store...")
        es_store_bm25 = ElasticsearchStore(
            es_url=es_url, 
            index_name=index_name,
            dim=embed_dim,
            retrieval_strategy=AsyncBM25Strategy()
        )
        storage_context_bm25 = StorageContext.from_defaults(vector_store=es_store_bm25)
        _INDEX_BM25 = VectorStoreIndex.from_vector_store(es_store_bm25, storage_context=storage_context_bm25, embed_model=embed_model)
        
        if not _es_index_populated(es_url, index_name):
            raise RuntimeError(
                f"Elasticsearch index '{index_name}' is empty. "
                f"Run 'python3 scripts/ingest.py' to populate it before starting the agent."
            )
        print(f"Using existing Elasticsearch index: {index_name}")

        # Load entities from persistence
        entities_path = os.path.join(persist_dir, "doc_entities.json")
        if os.path.exists(entities_path):
            try:
                with open(entities_path, 'r') as f:
                    data = json.load(f)
                    _CANONICAL_TEAMS.update(data.get("teams", []))
                    _CANONICAL_PLAYERS.update(data.get("players", []))
                print(f"Loaded {len(_CANONICAL_TEAMS)} teams and {len(_CANONICAL_PLAYERS)} players.")
            except Exception as e:
                print(f"Failed to load entities: {e}")

    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        # Make sure we don't leave it in partially failed state if essential
        _INDEX = None
        return



def _extract_players_from_stat_text(text: str) -> List[str]:
    """Parse player names from a game_stat_player.txt body.

    The file format is stable across all 3 sports: under each ``### Team`` header,
    every non-blank non-header line containing ``:`` is a player entry of the
    form ``Player Name: <stats>``. Sampled across 190 random files: zero noise
    lines, 48-146 players per file.

    Used as a fallback for documents-only games (no clips_metadata.json) so that
    ``metadata.players`` is populated for ES filter queries.
    """
    seen = set()
    out: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        name = s.split(":", 1)[0].strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _season_code_to_year(season_code: str) -> int:
    """Convert season code like '21-22' to season year (2022)."""
    parts = season_code.split("-")
    if len(parts) == 2:
        end_year = int(parts[1])
        return 2000 + end_year if end_year < 100 else end_year
    return int(season_code)


def load_documents(data_metadata: Dict, enabled_sources: List[str] = None) -> List[Document]:
    """Ingest documents with metadata from centralized data metadata."""
    documents = []

    # Flatten enabled sources if using high-level keys
    allowed_doc_types = []
    if enabled_sources:
        for src in enabled_sources:
            if src in SOURCE_TO_FILE_MAP:
                allowed_doc_types.extend(SOURCE_TO_FILE_MAP[src])
            else:
                allowed_doc_types.append(src)

    # Iterate through each item in metadata
    for item_id, item_data in data_metadata.items():
        sport = item_data.get('sport', 'unknown')

        # --- Handle per-entity season stat entries ---
        entity_type = item_data.get('entity_type')
        if entity_type is not None:
            entity_id = item_data.get('entity_id')
            season_code = item_data.get('season_code')

            # Load entity info
            entity_info = {}
            entity_info_path = item_data.get("entity_info")
            if entity_info_path and os.path.exists(entity_info_path):
                try:
                    with open(entity_info_path, 'r') as f:
                        entity_info = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load entity info for {item_id}: {e}")

            entity_name = entity_info.get("entity_name", f"{entity_type} {entity_id}")
            team_name = entity_info.get("team_name")

            # Determine which doc types to load.
            if entity_type == "player":
                entity_doc_map = {"season_stat_player": "season_stat_player"}
            else:  # team
                entity_doc_map = {"season_stat_team": "season_stat_team"}

            for key, doc_type in entity_doc_map.items():
                if allowed_doc_types and doc_type not in allowed_doc_types:
                    continue

                file_path = item_data.get(key)
                if not file_path or not os.path.exists(file_path):
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().replace('\n', '. \n')

                doc_id = f"{entity_type}_{entity_id}_{season_code}_{doc_type}"

                metadata = {
                    "doc_id": doc_id,
                    "sport": sport,
                    "doc_type": doc_type,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "season": _season_code_to_year(season_code),
                }

                if entity_type == "player":
                    metadata["teams"] = [team_name] if team_name else []
                    metadata["players"] = [entity_name]
                    _CANONICAL_PLAYERS.add(entity_name)
                    if team_name:
                        _CANONICAL_TEAMS.add(team_name)
                else:  # team
                    metadata["teams"] = [entity_name]
                    _CANONICAL_TEAMS.add(entity_name)

                documents.append(Document(text=text, doc_id=doc_id, metadata=metadata))

            continue  # Skip game-keyed logic for entity entries

        # --- Existing game-keyed logic ---
        game_id = item_id

        # Extract team/player info from clips_metadata
        teams = []
        players = []

        clips_metadata_path = item_data.get("clips_metadata")
        if clips_metadata_path and os.path.exists(clips_metadata_path):
            try:
                with open(clips_metadata_path, 'r') as f:
                    clip_data = json.load(f)

                windows = clip_data.get("windows", {})
                teams_set = set()
                players_set = set()
                for win in windows.values():
                    win_meta = win.get("metadata", {})
                    teams_set.update(win_meta.get("teams", {}).values())
                    players_set.update(win_meta.get("players", {}).values())
                teams = list(teams_set)
                players = list(players_set)

                for t in teams: _CANONICAL_TEAMS.add(t)
                for p in players: _CANONICAL_PLAYERS.add(p)
            except Exception as e:
                print(f"Warning: Failed to load clip metadata for game {game_id}: {e}")

        # Extract game info metadata
        game_info_path = item_data.get("game_info")
        game_info = {}
        if game_info_path and os.path.exists(game_info_path):
            try:
                with open(game_info_path, 'r') as f:
                    game_info = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load game info for game {game_id}: {e}")

        # Documents-only games (e.g. ~70% of soccer games) have no clips_metadata,
        # so the loop above leaves teams=[]. Fall back to game_info.{home,away}_team
        # so the resulting docs still match team-filter queries in ES.
        if not teams and game_info:
            teams = [t for t in (game_info.get("home_team"),
                                 game_info.get("away_team")) if t]
            for t in teams:
                _CANONICAL_TEAMS.add(t)

        # Per-game stat documents.
        doc_type_map = {
            "espn_report": "espn_report",
            "game_stat_player": "game_stat_player",
            "game_stat_team":   "game_stat_team",
        }

        for key, doc_type in doc_type_map.items():
            # Filter by enabled sources
            if allowed_doc_types and doc_type not in allowed_doc_types:
                continue

            file_path = item_data.get(key)
            if not file_path or not os.path.exists(file_path):
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            text = raw_text.replace('\n', '. \n')

            # Documents-only games (no clips_metadata.json) have players=[]
            # from the per-game loop. Parse the player roster out of the
            # game_stat_player.txt body so ES filter queries match.
            doc_players = players
            if doc_type == "game_stat_player" and not players:
                doc_players = _extract_players_from_stat_text(raw_text)
                for p in doc_players:
                    _CANONICAL_PLAYERS.add(p)

            doc_id = f"{game_id}_{doc_type}"
            metadata = {
                "game_id": game_id,
                "sport": sport,
                "doc_type": doc_type,
                "doc_id": doc_id,
                "teams": teams,
            }
            if doc_type not in ("game_stat_team",):
                metadata["players"] = doc_players

            # Add game info metadata based on doc type
            if doc_type in ("game_stat_player", "game_stat_team"):
                metadata["date"] = game_info.get("date")
                metadata["season"] = game_info.get("season")
                metadata["season_type"] = game_info.get("season_type")
                metadata["home_team"] = game_info.get("home_team")
                metadata["away_team"] = game_info.get("away_team")

            documents.append(Document(text=text, doc_id=doc_id, metadata=metadata))

    return documents

# ==============================================================================
# Helper for RRF
# ==============================================================================
def fuse_results(results_dict: Dict[str, List[NodeWithScore]], similarity_top_k: int = 5):
    """
    Fuse results using Reciprocal Rank Fusion (RRF).
    Adapted from user reference to use node_id for uniqueness.
    """
    k = 60.0  # constant for RRF
    fused_scores = {}
    id_to_node = {}

    # compute reciprocal rank scores
    # Note: nodes_with_scores now contains raw Elasticsearch scores (not normalized)
    # RRF ranks are computed based on these raw scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
        ):
            # Use node_id for uniqueness instead of content
            nid = node_with_score.node.node_id
            id_to_node[nid] = node_with_score
            
            if nid not in fused_scores:
                fused_scores[nid] = 0.0
            fused_scores[nid] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for nid, score in reranked_results.items():
        # Create a copy or modify in place? Modifying in place is fine as these are transient
        node = id_to_node[nid]
        node.score = score
        reranked_nodes.append(node)

    return reranked_nodes[:similarity_top_k]

# ==============================================================================
# Filter support per doc type
# ==============================================================================
FILTER_SUPPORTED_DOC_TYPES = {
    "players": ["game_stat_player", "season_stat_player"],
    "season": ["game_stat_player", "game_stat_team", "season_stat_player", "season_stat_team"],
    "date": ["game_stat_player", "game_stat_team"],
}

# ==============================================================================
# Tools
# ==============================================================================
def search_documents(query: str, metadata_filters: Dict = None, top_k: int = 5, score_threshold: float = 0.0, highlight_max_length: int = 500, semantic_weight_multiplier: int = 100, lexical_weight_multiplier: int = 100, fusion_rank_multiplier: int = 5, sport: str = None, data_metadata: Dict = None) -> List[Dict]:
    """
    Performs hybrid search (Semantic + Lexical) and returns metadata-rich summaries.
    """

    if _INDEX is None or _INDEX_BM25 is None:
        return [{"error": "Retriever not initialized. Please call init_document_database first."}], []

    # 1. Normalize Query
    if query is None or query.lower() in ["none", "null", ""]:
        query = ""

    # Build ES Filters
    es_filter_list = []
    ignored_filters = []

    # Sport filter (injected externally, not from agent)
    if sport:
        es_filter_list.append({"term": {"metadata.sport.keyword": sport}})

    if metadata_filters:
        # Map plural filter keys to singular metadata fields
        key_mapping = {
            "game_ids": "game_id",
            "doc_ids": "doc_id",
            "game_id": "game_id", # Support both for safety
            "doc_id": "doc_id"
        }

        for k, v in metadata_filters.items():
            # Use mapped key if available, else original
            meta_key = key_mapping.get(k, k)

            if k == "doc_type":
                # Expand high-level source keys (e.g. game_statistics -> [game_stat_player, ...])
                if isinstance(v, str) and v in SOURCE_TO_FILE_MAP:
                    v = SOURCE_TO_FILE_MAP[v]

                if isinstance(v, list):
                    es_filter_list.append({"terms": {f"metadata.{meta_key}.keyword": v}})
                else:
                    es_filter_list.append({"term": {f"metadata.{meta_key}.keyword": v}})

            elif k == "teams":
                input_teams = v if isinstance(v, list) else [v]
                input_teams = [entity_utils.normalize_team_name(t, sport=sport) for t in input_teams]
                resolved_teams = entity_utils.resolve_entities(input_teams, _CANONICAL_TEAMS)
                print(f"[Fuzzy Filter Doc] Teams: {input_teams} -> {resolved_teams}")
                if resolved_teams:
                    # Use .keyword for exact term matching on analyzed fields if possible
                    es_filter_list.append({"terms": {"metadata.teams.keyword": resolved_teams}})
                else:
                    query += " " + " ".join(input_teams)
                    ignored_filters.append(f"teams (could not resolve {input_teams}, appended back to query)")

            elif k in FILTER_SUPPORTED_DOC_TYPES:
                # Filters with doc_type restrictions
                supported = FILTER_SUPPORTED_DOC_TYPES[k]
                current_doc_type = metadata_filters.get("doc_type")

                # Check if doc_type is compatible
                if current_doc_type is not None:
                    # Expand high-level source keys for comparison
                    check_types = current_doc_type
                    if isinstance(current_doc_type, str) and current_doc_type in SOURCE_TO_FILE_MAP:
                        check_types = SOURCE_TO_FILE_MAP[current_doc_type]

                    if isinstance(check_types, str) and check_types not in supported:
                        ignored_filters.append(k)
                        print(f"[Filter Doc] Ignoring '{k}' filter: doc_type '{current_doc_type}' not supported.")
                        continue
                    elif isinstance(check_types, list) and not any(dt in supported for dt in check_types):
                        ignored_filters.append(k)
                        print(f"[Filter Doc] Ignoring '{k}' filter: doc_type {current_doc_type} not supported.")
                        continue
                else:
                    # No doc_type specified — auto-restrict to supported types
                    es_filter_list.append({"terms": {"metadata.doc_type.keyword": supported}})

                # Apply the actual filter
                if k == "players":
                    input_players = v if isinstance(v, list) else [v]
                    resolved_players = entity_utils.resolve_entities(input_players, _CANONICAL_PLAYERS)
                    print(f"[Fuzzy Filter Doc] Players: {input_players} -> {resolved_players}")
                    if resolved_players:
                        es_filter_list.append({"terms": {"metadata.players.keyword": resolved_players}})
                    else:
                        query += " " + " ".join(input_players)
                        ignored_filters.append(f"players (could not resolve {input_players}, appended back to query)")

                elif k == "date":
                    # Parse date format: "YYYY-MM-DD", "YYYY-MM-DD..", "..YYYY-MM-DD", "YYYY-MM-DD..YYYY-MM-DD"
                    v_str = str(v).strip()
                    if ".." in v_str:
                        parts = v_str.split("..", 1)
                        range_filter = {}
                        if parts[0]:
                            range_filter["gte"] = parts[0]
                        if parts[1]:
                            range_filter["lte"] = parts[1]
                        if range_filter:
                            es_filter_list.append({"range": {"metadata.date": range_filter}})
                    else:
                        es_filter_list.append({"term": {"metadata.date": v_str}})

                elif k == "season":
                    # season is stored as integer — no .keyword sub-field
                    es_filter_list.append({"term": {f"metadata.{k}": int(v)}})

            elif isinstance(v, list):
                es_filter_list.append({"terms": {f"metadata.{meta_key}.keyword": v}})
            else:
                es_filter_list.append({"term": {f"metadata.{meta_key}.keyword": v}})


    # 3. Direct Elasticsearch Retrieval (bypassing llama_index normalization)
    vector_store = _INDEX.vector_store
    es_client = vector_store.client
    index_name = vector_store.index_name
    
    # Get embedding for semantic search
    query_embedding = None
    if query:  # Only generate embedding if query exists
        query_embedding = _INDEX._embed_model.get_query_embedding(query)
    
    async def get_hybrid_results():
        semantic_nodes = []
        lexical_nodes = []
        
        # Semantic Search (Dense Vector)
        if query_embedding is not None:
            body_semantic = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k * semantic_weight_multiplier,
                    "num_candidates": max(top_k * semantic_weight_multiplier * 10, 100)
                },
                "fields": ["_id"],
                "_source": False
            }
            
            if es_filter_list:
                body_semantic["knn"]["filter"] = {"bool": {"must": es_filter_list}}
            
            result_semantic = await _es_search_with_retry(es_client, index_name, body_semantic)
            
            # Extract results
            semantic_ids = []
            semantic_scores = []
            for hit in result_semantic['hits']['hits']:
                semantic_ids.append(hit['_id'])
                semantic_scores.append(hit['_score'])
            
            # Get nodes
            if semantic_ids:
                fetched_nodes = vector_store.get_nodes(semantic_ids)
                for node, score in zip(fetched_nodes, semantic_scores):
                    if node:
                        semantic_nodes.append(NodeWithScore(node=node, score=score))
        
        # Lexical Search (BM25)
        if query:
            body_lexical = {
                "query": {
                    "bool": {
                        "must": [{"match": {"content": query}}]
                    }
                },
                "size": top_k * lexical_weight_multiplier
            }
            
            if es_filter_list:
                body_lexical["query"]["bool"]["filter"] = es_filter_list
            
            result_lexical = await _es_search_with_retry(es_client, index_name, body_lexical)
            
            # Extract results
            lexical_ids = []
            lexical_scores = []
            for hit in result_lexical['hits']['hits']:
                lexical_ids.append(hit['_id'])
                lexical_scores.append(hit['_score'])
            
            # Get nodes
            if lexical_ids:
                fetched_nodes = vector_store.get_nodes(lexical_ids)
                for node, score in zip(fetched_nodes, lexical_scores):
                    if node:
                        lexical_nodes.append(NodeWithScore(node=node, score=score))

        # Fallback: Filter-only search if no query but filters exist
        if not query and es_filter_list and not semantic_nodes and not lexical_nodes:
             body_filter = {
                "query": {
                    "bool": {
                        "filter": es_filter_list
                    }
                },
                "size": top_k * semantic_weight_multiplier
            }

             result_filter = await _es_search_with_retry(es_client, index_name, body_filter)
            
             filter_ids = []
             filter_scores = []
             for hit in result_filter['hits']['hits']:
                filter_ids.append(hit['_id'])
                filter_scores.append(1.0) 
            
             if filter_ids:
                fetched_nodes = vector_store.get_nodes(filter_ids)
                for node, score in zip(fetched_nodes, filter_scores):
                    if node:
                        # Append to semantic_nodes as a container
                        semantic_nodes.append(NodeWithScore(node=node, score=score))
        
        return semantic_nodes, lexical_nodes
    
    # Run async queries
    semantic_nodes, lexical_nodes = _run_coro_sync(get_hybrid_results())
    
    # Build raw score maps before case logic (RRF fusion overwrites node.score)
    semantic_score_map = {n.node.node_id: n.score for n in semantic_nodes}
    bm25_score_map = {n.node.node_id: n.score for n in lexical_nodes}
    max_bm25 = max(bm25_score_map.values()) if bm25_score_map else 0

    # Apply scoring logic based on query and filter presence
    # Both query cases use RRF for sorting; filter-only assigns null scores
    has_query = bool(query)
    has_filter = bool(es_filter_list)

    if not has_query and has_filter:
        # Filter-only: no retrieval scores
        nodes = semantic_nodes
        for node in nodes:
            node.score = 1.0
        is_filter_only = True
    elif has_query and not has_filter:
        # Query only: hybrid search (semantic + BM25), sort by RRF
        nodes = fuse_results(
            results_dict={"semantic": semantic_nodes, "lexical": lexical_nodes},
            similarity_top_k=top_k * fusion_rank_multiplier
        )
        is_filter_only = False
    elif has_query and has_filter:
        # Query + filter: hybrid search with filters, sort by RRF
        nodes = fuse_results(
            results_dict={"semantic": semantic_nodes, "lexical": lexical_nodes},
            similarity_top_k=top_k * fusion_rank_multiplier
        )
        is_filter_only = False
    else:
        nodes = []
        is_filter_only = False

    # Capture RRF scores (node.score is RRF after fuse_results)
    if not is_filter_only:
        rrf_score_map = {n.node.node_id: n.score for n in nodes}
    else:
        rrf_score_map = {}

    # Restructure Results
    doc_scores = {}
    for n in nodes:
        nid = n.node.node_id
        if is_filter_only:
            score_dict = {"rrf": None, "semantic": None, "bm25": None}
            sort_score = 1.0
        else:
            rrf_raw = rrf_score_map.get(nid)
            rrf_norm = min(round(rrf_raw * 30, 4), 1.0) if rrf_raw is not None else None

            sem = semantic_score_map.get(nid)
            sem = round(sem, 4) if sem is not None else None

            bm25_raw = bm25_score_map.get(nid)
            bm25_norm = round(bm25_raw / max_bm25, 4) if bm25_raw is not None and max_bm25 > 0 else None

            score_dict = {"rrf": rrf_norm, "semantic": sem, "bm25": bm25_norm}
            sort_score = rrf_norm if rrf_norm is not None else 0.0

        doc_scores[nid] = {"node": n, "score_dict": score_dict, "_sort_score": sort_score}

    grouped_results = {}
    sorted_nodes = sorted(doc_scores.values(), key=lambda x: x["_sort_score"], reverse=True)

    for item in sorted_nodes:
        node = item["node"]
        doc_id = node.node.ref_doc_id or node.node.metadata.get("doc_id")

        snippet = node.node.get_content(metadata_mode="none")

        if doc_id not in grouped_results:
             grouped_results[doc_id] = {
                "doc_id": doc_id,
                "_sort_score": item["_sort_score"],
                "score": item["score_dict"],
                "metadata": {k:v for k,v in node.node.metadata.items() if k not in ["window", "original_text"]},
                "highlights": [snippet]
             }
        else:
             if snippet not in grouped_results[doc_id]["highlights"]:
                 grouped_results[doc_id]["highlights"].append(snippet)

    final_docs = [d for d in grouped_results.values() if d["_sort_score"] >= score_threshold]
    final_docs = sorted(final_docs, key=lambda x: x["_sort_score"], reverse=True)[:top_k]

    for doc in final_docs:
        del doc["_sort_score"]

        full_high = "\n... ".join(doc["highlights"])
        if len(full_high) > highlight_max_length:
            full_high = full_high[:highlight_max_length] + " ... (truncated)"

        doc["highlights"] = full_high

        # Exclude specific keys from metadata
        exclude_keys = ["window", "original_text", "players", "doc_id"]
        for k in exclude_keys:
            if k in doc.get("metadata", {}):
                del doc["metadata"][k]

    # Check for missing data when no results found with specific filters
    if not final_docs and metadata_filters and data_metadata:
        game_ids = metadata_filters.get('game_ids', metadata_filters.get('game_id', []))
        if isinstance(game_ids, str):
            game_ids = [game_ids]
        doc_type = metadata_filters.get('doc_type')

        for gid in game_ids:
            game_meta = data_metadata.get(str(gid), {})
            if not game_meta:
                final_docs.append({"message": f"Game {gid} does not exist in the database."})
            elif doc_type:
                # Check if the specific document type exists for this game
                type_to_key = {
                    'espn_report': ['espn_report'],
                    'game_statistics': ['game_stat_player', 'game_stat_team'],
                    'game_stat_player': ['game_stat_player'],
                    'game_stat_team': ['game_stat_team'],
                    'season_statistics': ['season_stat_player', 'season_stat_team'],
                    'season_stat_player': ['season_stat_player'],
                    'season_stat_team': ['season_stat_team'],
                }
                keys = type_to_key.get(doc_type, [doc_type])
                missing = [k for k in keys if k not in game_meta]
                if missing:
                    final_docs.append({
                        "message": f"{doc_type} does not exist for game {gid}."
                    })

    return final_docs, ignored_filters

from openai import AsyncOpenAI
import datetime

def _log_prompt(experiment_path: str, tool_name: str, identifier: str, prompt: str):
    """Logs the full prompt to a file in the experiment directory."""
    if not experiment_path:
        return
    
    log_dir = os.path.join(experiment_path, "prompts")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_ident = os.path.basename(identifier)
    safe_ident = "".join([c if c.isalnum() else "_" for c in base_ident])[-64:]
    filename = f"qa_{timestamp}_{tool_name}_{safe_ident}.txt"
    filepath = os.path.join(log_dir, filename)
    
    try:
        with open(filepath, "w") as f:
            f.write(prompt)
    except Exception as e:
        print(f"Error logging prompt to {filepath}: {e}")

async def _process_single_doc_qa(client: AsyncOpenAI, model: str, doc_id: str, query: str, prompt_template: str, **kwargs) -> Dict:
    """Processes a single document QA task (Direct ElasticSearch lookup)."""
    try:
        content = None
        
        if _INDEX is not None:
            try:
                # Direct Native ES Query for better reliability
                vs = _INDEX.storage_context.vector_store
                es_client = vs.client
                index_name = vs.index_name
                
                body = {
                    "query": {"term": {"metadata.doc_id": doc_id}},
                    "size": 500,
                    "_source": ["content"]
                }
                
                es_res = await _es_search_with_retry(es_client, index_name, body)
                hits = es_res.get("hits", {}).get("hits", [])
                
                if hits:
                    # Deduplicate contents while preserving order
                    unique_contents = []
                    seen = set()
                    for hit in hits:
                        c = hit["_source"]["content"]
                        if c not in seen:
                            seen.add(c)
                            unique_contents.append(c)
                    content = "\n".join(unique_contents)
                else:
                    # Fallback to docstore
                    node = _INDEX.docstore.get_document(doc_id)
                    if node: content = node.get_content(metadata_mode="none")
            except Exception as e:
                print(f"[QA Lookup] ES Error for {doc_id}: {e}")

        if content is None:
            return {"doc_id": doc_id, "error": f"Document {doc_id} not found."}
            
        max_content_tokens = kwargs.get("max_content_tokens")
        assert max_content_tokens is not None, "max_content_tokens must be provided in config"

        full_prompt = prompt_template.replace("{{document_content}}", content[:max_content_tokens]).replace("{{query}}", query)
        
        # Log the prompt if experiment_path is provided
        _log_prompt(kwargs.get("experiment_path"), "document_qa", doc_id, full_prompt)
        
        llm_config = kwargs.get("llm_config", {})
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
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

            # 2. Extract content from <answer> tags if present (prompt requests this format)
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
                res["doc_id"] = doc_id
                return res

            res = json.loads(answer_clean)
            res["doc_id"] = doc_id
            return res
        except (ValueError, TypeError):
            # Best-effort cleanup for fallback
            import re
            fallback = re.sub(r'<think>.*?</think>', '', answer_raw, flags=re.DOTALL).strip()
            fallback = re.sub(r'<think>.*', '', fallback, flags=re.DOTALL).strip()
            if "<answer>" in fallback:
                fallback = fallback.split("<answer>")[1].split("</answer>")[0].strip()
            return {
                "doc_id": doc_id,
                "answer": fallback,
                "confidence": 0.5,
                "evidence": "Model returned raw text."
            }

    except Exception as e:
        return {"doc_id": doc_id, "error": f"QA Failed: {str(e)}"}

def document_qa(doc_ids: List[str], query: str, **kwargs) -> List[Dict]:
    """
    Analyzes specific documents concurrently using an LLM (VLLM Async Batching).
    """
    
    llm_cfg = kwargs.get("llm_config", {})
    llm_api_base = llm_cfg.get("model_server")
    llm_api_key = llm_cfg.get("api_key", "EMPTY")
    model_name = llm_cfg.get("model")
    if not llm_api_base or not model_name:
        raise ValueError(
            "document_qa: llm_config is missing required keys "
            f"(model_server={llm_api_base!r}, model={model_name!r}). "
            "This is set by init_tools at agent startup — ensure the arch "
            "YAML defines a 'document_qa' tool block."
        )
    
    # Load Prompt
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Use path from config or fallback
    rel_path = llm_cfg.get('prompt', "prompts/tools/document_qa.txt")
    prompt_path = os.path.join(base_dir, rel_path)
    
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()
    else:
        # Fallback
        prompt_template = "Document: {{document_content}}\nQuestion: {{query}}\nAnswer in JSON."

    async def run_batch():
        client = AsyncOpenAI(api_key=llm_api_key, base_url=llm_api_base)
        tasks = []
        for doc_id in doc_ids:
            tasks.append(_process_single_doc_qa(client, model_name, doc_id, query, prompt_template, **kwargs))
        
        results = await asyncio.gather(*tasks)
        await client.close()
        return results

    try:
        return _run_coro_sync(run_batch())

    except Exception as e:
        return [{"error": f"Batch Execution Failed: {str(e)}"}]
