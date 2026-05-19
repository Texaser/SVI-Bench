import os
import argparse
import yaml
import json
import json5
import sys
import glob
import logging
from typing import Union, Dict, List


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['QWEN_AGENT_DEFAULT_WORKSPACE'] = os.path.join(BASE_DIR, 'etc', 'qwen_agent_workspace')

# Ensure same-dir imports resolve when this file is run as a script.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _t9_root import (
    T9_ROOT_NOT_SET,
    resolve_t9_data_root as _resolve_t9_data_root,
    require_t9_data_root as _require_t9_data_root,
    resolve_t9_results_dir as _resolve_t9_results_dir,
)


T9_DATA_ROOT = _resolve_t9_data_root()

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message
from qwen_agent.tools.base import BaseTool, register_tool
from agent.core import CustomQwenMultiTurnReactAgent, CustomGPT5MultiTurnReactAgent, CustomGPT5ResponsesReactAgent

from tools import document_tools, video_tools

logging.getLogger('qwen_agent').setLevel(logging.INFO)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def apply_host_rewrite(url: str, role: str = 'tool') -> str:
    """Rewrite a localhost service URL to point at a remote host.

    Reads T9-prefixed env vars so the agent service and tool services can
    live on different nodes:

      role='agent'  → T9_AGENT_SERVER_HOST
      role='tool'   → T9_TOOL_SERVER_HOST

    T9_PORT_OFFSET shifts the port in either case. Non-localhost URLs are
    returned unchanged (external APIs are never touched). If neither host
    nor offset is set for the requested role, the URL passes through.
    """
    offset = int(os.environ.get('T9_PORT_OFFSET', '0'))
    if role == 'agent':
        service_host = os.environ.get('T9_AGENT_SERVER_HOST', '')
    elif role == 'tool':
        service_host = os.environ.get('T9_TOOL_SERVER_HOST', '')
    else:
        raise ValueError(f"role must be 'agent' or 'tool', got {role!r}")
    if offset == 0 and not service_host:
        return url
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    if not parsed.hostname or parsed.hostname not in ('localhost', '127.0.0.1'):
        return url
    new_host = service_host if service_host else parsed.hostname
    new_port = (parsed.port + offset) if parsed.port else parsed.port
    new_netloc = f"{new_host}:{new_port}" if new_port else new_host
    return urlunparse(parsed._replace(netloc=new_netloc))

def load_data_metadata(data_base_path: str, enabled_datasets: List[str]) -> Dict[str, Dict]:
    """Load metadata for all enabled datasets.

    Args:
        data_base_path: Base directory containing dataset subdirectories.
        enabled_datasets: List of dataset names (e.g., ["basketball"]).

    Returns:
        Dict mapping dataset -> {item_id -> item_data}
    """
    datasets_metadata = {}

    for dataset in enabled_datasets:
        dataset_dir = os.path.join(data_base_path, dataset)
        # Make dataset_dir absolute
        dataset_dir = os.path.abspath(dataset_dir)

        metadata_file = os.path.join(dataset_dir, "metadata.json")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found for dataset: {dataset} at {metadata_file}")

        with open(metadata_file, 'r') as f:
            data_metadata = json.load(f)

        # Convert relative paths to absolute by joining with absolute dataset_dir
        for item_id, item_data in data_metadata.items():
            # Add sport field
            item_data['sport'] = dataset

            for key, rel_path in item_data.items():
                # Skip sport field - it's not a path
                if key == 'sport':
                    continue
                if isinstance(rel_path, str) and rel_path:
                    # Join absolute dataset_dir with relative path
                    item_data[key] = os.path.join(dataset_dir, rel_path)

        datasets_metadata[dataset] = data_metadata
        print(f"Loaded {len(data_metadata)} items for {dataset}")

        # Load per-entity season metadata (player and team)
        entity_meta_files = {
            "metadata_season_player.json": "player",
            "metadata_season_team.json": "team",
        }

        for entity_meta_filename, entity_type in entity_meta_files.items():
            entity_meta_path = os.path.join(dataset_dir, entity_meta_filename)
            if not os.path.exists(entity_meta_path):
                continue

            with open(entity_meta_path, 'r') as f:
                entity_metadata = json.load(f)

            count = 0
            for entity_id, seasons_data in entity_metadata.items():
                for season_code, paths_dict in seasons_data.items():
                    synthetic_key = f"{entity_type}_{entity_id}_{season_code}"

                    item_data = {
                        "sport": dataset,
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "season_code": season_code,
                    }

                    # Convert relative paths to absolute
                    for key, rel_path in paths_dict.items():
                        if isinstance(rel_path, str) and rel_path:
                            item_data[key] = os.path.join(dataset_dir, rel_path)

                    data_metadata[synthetic_key] = item_data
                    count += 1

            print(f"  Loaded {count} {entity_type} entity-season entries from {entity_meta_filename}")

    return datasets_metadata



def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, 'r') as f:
        return f.read()

def generate_env_desc(config: Dict) -> str:
    enabled_sources = config.get('hyperparams', {}).get('data', {}).get('enabled_sources', [])
    if not enabled_sources:
        return "No specific data sources configured."
    env_prompts_dir = os.path.join(BASE_DIR, 'prompts', 'env')
    desc_lines = []
    for source in enabled_sources:
        # Prefer _w_example.txt if it exists
        desc_file_w_example = os.path.join(env_prompts_dir, f"{source}_w_example.txt")
        desc_file = os.path.join(env_prompts_dir, f"{source}.txt")
        if os.path.exists(desc_file_w_example):
            with open(desc_file_w_example, 'r') as f:
                desc = f.read().strip()
            desc_lines.append(f"- [{source}]:\n{desc}")
        elif os.path.exists(desc_file):
            with open(desc_file, 'r') as f:
                desc = f.read().strip()
            desc_lines.append(f"- [{source}]: {desc}")
        else:
            print(f"Warning: Description file for {source} not found at {desc_file}")
            desc_lines.append(f"- [{source}]: Data source enabled but no description file found.")
    desc_text = "\n\n".join(desc_lines)
    # Inject canonical teams found in metadata
    if video_tools._CANONICAL_TEAMS:
        teams_str = ", ".join(sorted(list(video_tools._CANONICAL_TEAMS)))
        desc_text += f"\n\n[Entity Reference]: The following teams are present in the current dataset: {teams_str}. Please use these names for filtering."
    return desc_text


def load_tools_config() -> Dict:
    """Load tool definitions from configs/tools.yaml."""
    tools_config_path = os.path.join(BASE_DIR, 'configs', 'tools.yaml')
    if not os.path.exists(tools_config_path):
        print(f"Warning: tools.yaml not found at {tools_config_path}, using hardcoded definitions")
        return {}
    
    with open(tools_config_path, 'r') as f:
        tools_config = yaml.safe_load(f)
    
    return tools_config


# ==============================================================================
# Tool Wrappers (Linking Config to Implementation)
# ==============================================================================
@register_tool("search_documents")
class SearchDocumentsTool(BaseTool):
    description = ""
    description += "Performs hybrid search (Semantic + Lexical) on document database."
    description += "Note that the results show only the search highlights in the documents, not the full documents."
    parameters = [
        {'name': 'query', 'type': 'string', 'description': 'The search query. For filter options, use the other parameters. Empty string with filters are supported.', 'required': True},
        {'name': 'game_ids', 'type': 'array', 'items': {'type': 'string'}, 'description': 'Optional list of game IDs (e.g. ["401738"]).', 'required': False},
        {'name': 'doc_type', 'type': 'string', 'description': 'Filter by type: espn_report, game_stat_player, game_stat_team, season_stat_player, season_stat_team.', 'required': False},
        {'name': 'doc_ids', 'type': 'array', 'items': {'type': 'string'}, 'description': 'Optional list of IDs in {game_id}_{doc_type} format (e.g. ["401738_espn_report"]).', 'required': False},
        {'name': 'teams', 'type': 'array', 'items': {'type': 'string'}, 'description': 'List of canonical team names (e.g. ["Portland Trail Blazers"]). Matches documents containing ANY of the listed teams (OR logic).', 'required': False},
        {'name': 'players', 'type': 'array', 'items': {'type': 'string'}, 'description': 'List of full player names (e.g. ["Stephen Curry"]). Matches documents containing ANY of the listed players (OR logic). Supported doc types: game_stat_player, season_stat_player.', 'required': False},
        {'name': 'season', 'type': 'integer', 'description': 'Filter by season year (e.g. 2019). Supported doc types: game_stat_player, game_stat_team, season_stat_player, season_stat_team.', 'required': False},
        {'name': 'date', 'type': 'string', 'description': 'Filter by game date. Exact: "2019-04-28", range: "2019-04-01..2019-04-30", after: "2019-04-01..", before: "..2019-04-30". Supported doc types: game_stat_player, game_stat_team.', 'required': False},
    ]

    def _get_repacked_filters(self, params):
        keys = ["game_ids", "doc_type", "doc_ids", "teams", "players", "season", "date"]
        metadata_filters = {}
        for k in keys:
            v = params.get(k)
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            metadata_filters[k] = v
        return metadata_filters if metadata_filters else None

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            try: params = json5.loads(params)
            except: params = {"query": params}

        search_cfg = self.cfg.get('hyperparams', {}).get('document_search', {})
        top_k = search_cfg.get('top_k', 5)
        score_threshold = search_cfg.get('score_threshold', 0.0)
        highlight_max = search_cfg.get('highlight_max_length', 500)

        results, ignored = document_tools.search_documents(
            query=params.get("query"),
            metadata_filters=self._get_repacked_filters(params),
            top_k=top_k,
            score_threshold=score_threshold,
            highlight_max_length=highlight_max,
            semantic_weight_multiplier=search_cfg.get('semantic_multiplier', 10),
            lexical_weight_multiplier=search_cfg.get('lexical_multiplier', 10),
            fusion_rank_multiplier=search_cfg.get('fusion_rank_multiplier', 5),
            sport=self.cfg.get('sport'),
            data_metadata=self.cfg.get('data_metadata')
        )

        if ignored:
            doc_type_str = params.get("doc_type", "unspecified")
            warning = {
                "warning": f"Filters {ignored} were ignored for doc_type '{doc_type_str}'. "
                           f"Refer to the tool description for supported doc types."
            }
            results = [warning] + results

        return json.dumps(results, indent=2)

@register_tool("document_qa")
class DocumentQATool(BaseTool):
    description = ""
    description += "Takes a list of documents and analyzes the given documents one by one by reading the full content of each document.\n"
    description += "Note that the QA model processes each document independently.\n"
    description += "So, it does not know about the other documents. Be careful of this when formulating your question.\n\n"
    parameters = [
        {'name': 'doc_ids', 'type': 'array', 'items': {'type': 'string'}, 'description': 'List of document IDs. Use the format {game_id}_{doc_type}, e.g., 401738_espn_report.', 'required': True},
        {'name': 'query', 'type': 'string', 'description': 'The specific question.', 'required': True}
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            try: params = json5.loads(params)
            except: return "Error: invalid params"

        qa_cfg = self.cfg.get('hyperparams', {}).get('document_qa', {})
        max_content_tokens = qa_cfg.get('max_content_tokens')

        return json.dumps(document_tools.document_qa(
            doc_ids=params.get("doc_ids", []),
            query=params.get("query", ""),
            llm_config=self.cfg.get('llm_config', {}),
            max_content_tokens=max_content_tokens,
            experiment_path=self.cfg.get('experiment_path')
        ), indent=2)

@register_tool("search_videos")
class SearchVideosTool(BaseTool):
    description = ""
    description += "Performs semantic search for video clips. Returns IDs and metadata."
    parameters = [
        {'name': 'query', 'type': 'string', 'description': "The search query. For filter options, use the other parameters. Empty string with filters are supported.", 'required': True},
        {'name': 'game_ids', 'type': 'array', 'items': {'type': 'string'}, 'description': 'Optional list of game IDs (e.g. ["401738"]).', 'required': False},
        {'name': 'period', 'type': 'integer', 'description': 'Optional period/quarter/half number (e.g., 1-4 for basketball quarters, 1-3+ for hockey periods, 1-2 for soccer halves).', 'required': False},
        {'name': 'teams', 'type': 'array', 'items': {'type': 'string'}, 'description': 'List of canonical team names (e.g. ["Portland Trail Blazers"]). Matches clips containing ANY of the listed teams (OR logic).', 'required': False},
        {'name': 'players', 'type': 'array', 'items': {'type': 'string'}, 'description': 'List of full player names (e.g. ["Maurice Harkless"]). Matches clips containing ANY of the listed players (OR logic).', 'required': False},
        {'name': 'video_ids', 'type': 'array', 'items': {'type': 'string'}, 'description': 'Optional list of IDs in {game_id}_{window_id} format (e.g. ["401738_5"]).', 'required': False},
        # {'name': 'time_remaining', 'type': 'string', 'description': 'Countdown time range "MM:SS-MM:SS" (e.g. "02:00-00:00").', 'required': False},
        {'name': 'temporal_boundary', 'type': 'string', 'description': 'Temporal boundary in actual VIDEO TIMESTAMPS in seconds (e.g. "100.5-120.0").', 'required': False}
    ]

    def _get_repacked_filters(self, params):
        keys = ["game_ids", "period", "teams", "players", "video_ids", "time_remaining", "temporal_boundary"]
        metadata_filters = {}
        for k in keys:
            v = params.get(k)
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            metadata_filters[k] = v
        return metadata_filters if metadata_filters else None

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            try: params = json5.loads(params)
            except: params = {"query": params}

            
        # Read embedding_source from arch config instead of hyperparameters
        search_videos_cfg = self.cfg.get('search_videos_config', {})
        emb_source = search_videos_cfg.get("embedding_source", "video")
        
        video_search_cfg = self.cfg.get('hyperparams', {}).get('video_search', {})
        top_k = video_search_cfg.get('top_k', 10)

        return json.dumps(video_tools.search_videos(
            query=params.get("query"),
            metadata_filters=self._get_repacked_filters(params),
            embedding_source=emb_source,
            top_k=top_k,
            sport=self.cfg.get('sport'),
            data_metadata=self.cfg.get('data_metadata')
        ), indent=2)

@register_tool("video_qa")
class VideoQATool(BaseTool):
    description = ""
    description += "Takes a list of videos and analyzes the given videos one by one.\n"
    description += "Note that the QA model processes each video clip independently.\n"
    description += "So, it does not know about the other clips. Be careful of this when formulating your question.\n\n"
    parameters = [
        {'name': 'video_ids', 'type': 'array', 'items': {'type': 'string'}, 'description': 'List of Video IDs. Use the format [{game_id}_{window_id}, ...] e.g., [401738_5].', 'required': True},
        {'name': 'query', 'type': 'string', 'description': 'Visual question.', 'required': True}
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            try: params = json5.loads(params)
            except: return "Error: invalid params"

        qa_cfg = self.cfg.get('hyperparams', {}).get('video_qa', {})
        max_content_tokens = qa_cfg.get('max_content_tokens')
        num_frames = qa_cfg.get('video_num_frames')
        video_resize = qa_cfg.get('video_resize')

        return json.dumps(video_tools.video_qa(
            video_ids=params.get("video_ids", []),
            query=params.get("query", ""),
            llm_config=self.cfg.get('llm_config', {}),
            max_content_tokens=max_content_tokens,
            video_num_frames=num_frames,
            video_resize=video_resize,
            experiment_path=self.cfg.get('experiment_path'),
            video_data_path=self.cfg.get('data_metadata')
        ), indent=2)

@register_tool("video_qa_oracle")
class VideoQAOracleTool(BaseTool):
    description = ""
    description += "Takes a list of videos and analyzes the given videos one by one.\n"
    description += "Note that the QA model processes each video clip independently.\n"
    description += "So, it does not know about the other clips. Be careful of this when formulating your question.\n\n"
    parameters = [
        {'name': 'video_ids', 'type': 'array', 'items': {'type': 'string'}, 'description': 'List of Video IDs. Use the format [{game_id}_{window_id}, ...] e.g., [401738_5].', 'required': True},
        {'name': 'query', 'type': 'string', 'description': 'Question about events in the clip.', 'required': True}
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            try: params = json5.loads(params)
            except: return "Error: invalid params"

        qa_cfg = self.cfg.get('hyperparams', {}).get('video_qa', {})
        max_content_tokens = qa_cfg.get('max_content_tokens', 64000)
        return json.dumps(video_tools.video_qa_oracle(
            video_ids=params.get("video_ids", []),
            query=params.get("query", ""),
            llm_config=self.cfg.get('llm_config', {}),
            max_content_tokens=max_content_tokens,
            experiment_path=self.cfg.get('experiment_path')
        ), indent=2)

# ==============================================================================
# Initialization
# ==============================================================================
def get_model_config(key, all_models):
    """Searches for a model key across all categories in models.yaml."""
    for category in ['agent_models', 'tool_models']:
        section = all_models.get(category, {})
        if key in section:
            return section[key]
    return None

def build_llm_cfg(m_cfg, all_models=None, role: str = 'tool'):
    """Helper to build LLM config from a model dict or model_key.

    ``role`` ('agent' or 'tool') decides which T9_*_SERVER_HOST env var
    applies when rewriting a localhost server URL — see apply_host_rewrite.
    """
    if 'model_key' in m_cfg and all_models:
        key = m_cfg['model_key']
        base_cfg = get_model_config(key, all_models)
        if not base_cfg:
             raise ValueError(f"Model key '{key}' not found in any section of models.yaml")
        # Overlays arch-specific overrides (like 'prompt') onto base model config
        merged = base_cfg.copy()
        merged.update({k:v for k,v in m_cfg.items() if k != 'model_key'})
        m_cfg = merged

    api_key = m_cfg.get('api_key', "EMPTY")
    if api_key == "ENV":
        api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)

    # Filter out vLLM-specific parameters from generate_cfg as a safety measure
    vllm_params = ['max_model_len', 'gpu_memory_utilization', 'tensor_parallel_size']
    generate_cfg = m_cfg.get('generate_cfg', {}).copy()
    for param in vllm_params:
        if param in generate_cfg:
            del generate_cfg[param]

    return {
        "model": m_cfg.get('name', ''),
        "model_server": apply_host_rewrite(m_cfg.get('server', ''), role=role),
        "model_type": m_cfg.get('type', 'llm'),
        "generate_cfg": generate_cfg,
        "api_key": api_key,
        "prompt": m_cfg.get('prompt'),
        "prompts": m_cfg.get('prompts')
    }

def init_tools(config: Dict, all_data_metadata: Dict) -> List[BaseTool]:
    """Initializes tools based on configuration."""
    
    all_models = config.get('models', {})
    
    tool_runtime_cfg = config['hyperparams'].get('tools', {})
    
    # Load tool definitions from YAML
    tools_config = load_tools_config()

    def get_tool_cfg(tool_name):
        cfg = tool_runtime_cfg.copy()
        cfg['hyperparams'] = config['hyperparams']
        cfg['document_search_config'] = config['hyperparams'].get('document_search', {})
        cfg['data_metadata'] = all_data_metadata
        if 'sport' in config:
            cfg['sport'] = config['sport']
        
        # Add search_videos config from arch (contains embedding_model and embedding_source)
        cfg['search_videos_config'] = config['arch'].get('search_videos', {})
        
        tool_llm_cfg_dict = config['arch'].get(tool_name)
        if tool_llm_cfg_dict:
             cfg['llm_config'] = build_llm_cfg(tool_llm_cfg_dict, all_models, role='tool')
        else:
             if 'llm_config' not in cfg:
                 cfg['llm_config'] = {}
        return cfg

    tool_cls_map = {
        "search_documents": SearchDocumentsTool,
        "document_qa": DocumentQATool,
        "search_videos": SearchVideosTool,
        "video_qa": VideoQATool,
        "video_qa_oracle": VideoQAOracleTool
    }
    
    active_tools_names = config['arch'].get('tools', [])
    function_list = []
    
    for tool_name in active_tools_names:
        if tool_name in tool_cls_map:
             tool_cls = tool_cls_map[tool_name]
             tool_instance = tool_cls(get_tool_cfg(tool_name))
             
             # Apply YAML config if available (Option A: override from YAML)
             if tool_name in tools_config:
                 yaml_cfg = tools_config[tool_name]
                 # Override description from YAML
                 if 'description' in yaml_cfg:
                     tool_instance.description = yaml_cfg['description'].strip()
                 # Override parameters from YAML
                 if 'parameters' in yaml_cfg:
                     tool_instance.parameters = yaml_cfg['parameters']
                 # Append example to description if available
                 if 'example' in yaml_cfg:
                     tool_instance.description += f"\n\nExample:\n{yaml_cfg['example']}\n"
             
             # Substitute top-K placeholders in description with actual hyperparameter values
             doc_search_top_k = config['hyperparams'].get('document_search', {}).get('top_k', 20)
             video_search_top_k = config['hyperparams'].get('video_search', {}).get('top_k', 10)
             tool_instance.description = tool_instance.description.replace(
                 '{document_search_top_k}', str(doc_search_top_k)
             ).replace(
                 '{video_search_top_k}', str(video_search_top_k)
             )
             
             function_list.append(tool_instance)
        else:
             print(f"Warning: Tool '{tool_name}' defined in architecture but not implemented.")
             
    return function_list


def init_agent(config: Dict, prompt: str, agent_model_key: str, function_list: List[BaseTool]):
    model_cfg = config['models']['agent_models'].get(agent_model_key)
    if not model_cfg:
        raise ValueError(f"Agent model '{agent_model_key}' not found in configs.")

    print(f"Initializing Agent with Model: {model_cfg['name']} ({model_cfg['type']})")
    llm_cfg = build_llm_cfg(model_cfg, role='agent')

    system_prompt = prompt
    agent_class = None
    
    # Check for api_version to select Responses API agent
    api_version = model_cfg.get('api_version', 'completions')  # Default to completions
    
    if 'qwen' in model_cfg['type'].lower() or 'qwen' in model_cfg['name'].lower():
        agent_class = CustomQwenMultiTurnReactAgent
    elif 'gpt-5' in model_cfg['name'].lower() or 'o1' in model_cfg['name'].lower() or 'o3' in model_cfg['name'].lower():
        # Use Responses API agent if api_version is "responses"
        if api_version == 'responses':
            agent_class = CustomGPT5ResponsesReactAgent
            print(f"Using Responses API agent (api_version: {api_version})")
        else:
            agent_class = CustomGPT5MultiTurnReactAgent
    elif 'minimax' in model_cfg['name'].lower():
        # MiniMax-M2.5 uses the same XML <tool_call> protocol as Qwen via vLLM's
        # OpenAI-compat endpoint. Its arch YAMLs already point to
        # prompts/qwen_system_prompt_react.txt, so the Qwen agent class is the
        # correct routing target.
        agent_class = CustomQwenMultiTurnReactAgent

    if agent_class is None:
        raise ValueError(f"No suitable agent class found for model '{model_cfg['name']}' (type: {model_cfg['type']}). Expected 'qwen', 'gpt-5', or 'minimax' in name/type.")

    agent = agent_class(
        llm_config=llm_cfg,
        system_message=system_prompt,
        function_list=function_list,
        max_llm_calls=config['hyperparams']['react'].get('max_llm_calls', 20),
        max_retries=config['hyperparams']['react'].get('max_retries', 5),
        retry_base_sleep=config['hyperparams']['react'].get('retry_base_sleep', 2),
        max_context_tokens=config['hyperparams']['react'].get('max_context_tokens', 110000),
        pruning_token_limit=config['hyperparams']['react'].get('pruning_token_limit', 100000),
        verbosity_level=config['hyperparams']['react'].get('verbosity_level', 1),
        prompts_dir=os.path.join(BASE_DIR, 'prompts')
    )
    return agent, system_prompt




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, help="Architecture config name (one of KNOWN_ARCHS in evaluate.py)")
    parser.add_argument("--sport", required=True, choices=["basketball", "hockey", "soccer"],
                        help="Sport that drives ES index selection. The retrieval tools query "
                             "{document,video}_index_<sport>_<emb>_all, so this must match the "
                             "sport whose data was indexed.")
    args = parser.parse_args()

    CONFIG_MODELS_PATH = os.path.join(BASE_DIR, "configs/models.yaml")
    CONFIG_HYPER_PATH = os.path.join(BASE_DIR, "configs/hyperparameters.yaml")
    CONFIG_PATHS_PATH = os.path.join(BASE_DIR, "configs/paths.yaml")
    ARCH_PATH = os.path.join(BASE_DIR, f"archs/{args.arch}.yaml")

    if not os.path.exists(ARCH_PATH):
        raise ValueError(f"Architecture config not found: {ARCH_PATH}")

    # Experiment Logging Setup
    def setup_experiment_logging(arch_name, config_map, prompt_path, final_prompt_content, function_list):
        import datetime, shutil
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_root = _resolve_t9_results_dir()

        exp_dir_name = f"{timestamp}_{arch_name}"
        exp_path = os.path.join(log_root, exp_dir_name)
        os.makedirs(exp_path, exist_ok=True)
        print(f"Created experiment log: {exp_path}")
        
        # Copy Configs
        for name, path in [
            ('models.yaml', CONFIG_MODELS_PATH), ('hyperparameters.yaml', CONFIG_HYPER_PATH),
            ('paths.yaml', CONFIG_PATHS_PATH),
            (f'{arch_name}.yaml', ARCH_PATH)
        ]:
            if os.path.exists(path): shutil.copy(path, os.path.join(exp_path, name))
        
        # Generate tools.yaml dynamically from function_list
        tools_yaml_content = "# Tool Definitions (Generated from Python classes)\n\n"
        for tool in function_list:
            tools_yaml_content += f"{tool.name}:\n"
            tools_yaml_content += f"  description: \"{tool.description}\"\n"
            tools_yaml_content += f"  parameters:\n"
            for param in tool.parameters:
                tools_yaml_content += f"    - name: \"{param['name']}\"\n"
                tools_yaml_content += f"      type: \"{param['type']}\"\n"
                tools_yaml_content += f"      description: \"{param.get('description', '')}\"\n"
                tools_yaml_content += f"      required: {param.get('required', False)}\n"
                if param.get('items'):
                    tools_yaml_content += f"      items:\n"
                    tools_yaml_content += f"        type: \"{param['items'].get('type', 'string')}\"\n"
            tools_yaml_content += "\n"
        
        with open(os.path.join(exp_path, 'tools.yaml'), 'w') as f:
            f.write(tools_yaml_content)
        
        # Save Prompt
        prompt_filename = os.path.basename(prompt_path)
        with open(os.path.join(exp_path, prompt_filename), "w") as f:
            f.write(final_prompt_content)
        return exp_path

    print("Loading configs...")
    config = {
        "models": load_config(CONFIG_MODELS_PATH),
        "hyperparams": load_config(CONFIG_HYPER_PATH),
        "paths": load_config(CONFIG_PATHS_PATH),
        "arch": load_config(ARCH_PATH),
        "sport": args.sport,
    }

    _t9_root_runtime = _require_t9_data_root()

    def resolve_path(cli_arg, config_key):
        path = cli_arg or config['paths'].get(config_key)
        return path if os.path.isabs(path) else os.path.join(_t9_root_runtime, path)

    path_keys = [
        'data_base_path',
        'clip_embeddings_base_path',
        'video_persist_dir',
        'video_oracle_persist_dir',
        'document_persist_dir',
    ]

    for key in path_keys:
        val = config['paths'].get(key)
        if val and not os.path.isabs(val):
            val = os.path.join(_t9_root_runtime, val)
        # Update config so downstream tools get absolute path
        config['paths'][key] = val

    DATA_BASE_PATH = config['paths']['data_base_path']
    CLIP_EMBEDDINGS_BASE_PATH = config['paths']['clip_embeddings_base_path']
    VIDEO_PERSIST_DIR = config['paths']['video_persist_dir']
    VIDEO_ORACLE_PERSIST_DIR = config['paths']['video_oracle_persist_dir']
    DOCUMENT_PERSIST_DIR = config['paths']['document_persist_dir']

    PROMPT_PATH = os.path.join(BASE_DIR, config['arch']['agent'].get('prompt'))
    system_prompt = load_prompt(PROMPT_PATH)
    
    # Load Model Config to check type
    model_key = config['arch']['agent']['model_key']
    model_cfg = get_model_config(model_key, config['models'])
    if not model_cfg:
        raise ValueError(f"Agent model_key '{model_key}' not found in models.yaml")

    env_desc = generate_env_desc(config)
    if "{{env_desc}}" in system_prompt:
        system_prompt = system_prompt.replace("{{env_desc}}", env_desc)
    # Load enabled sources and datasets
    enabled_datasets = config.get('hyperparams', {}).get('data', {}).get('enabled_datasets', [])
    if not enabled_datasets:
        raise ValueError("No datasets enabled in hyperparameters.yaml")
    
    enabled_sources = config.get('hyperparams', {}).get('data', {}).get('enabled_sources', [])
    
    # Load all datasets metadata
    print(f"Loading metadata for datasets: {enabled_datasets}")
    datasets_metadata = load_data_metadata(DATA_BASE_PATH, enabled_datasets)
    
    # Flatten to single dict with dataset info in metadata
    all_data_metadata = {}
    for dataset, data_metadata in datasets_metadata.items():
        for item_id, item_data in data_metadata.items():
            # Add dataset info to item data
            item_data['sport'] = dataset
            all_data_metadata[item_id] = item_data
    
    print(f"Total items loaded across all datasets: {len(all_data_metadata)}")
    
    active_tools = config['arch']['tools']

    # Set EMBEDDING_GPUS from arch config for bge-m3 GPU distribution
    embedding_gpus = config['arch'].get('services', {}).get('m3', {}).get('gpus', [])
    if embedding_gpus:
        os.environ.setdefault('EMBEDDING_GPUS', ','.join(map(str, embedding_gpus)))

    doc_search_cfg = config['arch'].get('search_documents', {})
    es_url = os.environ.get('T9_ES_URL') or config.get('hyperparams', {}).get('elasticsearch', {}).get('url', "http://localhost:9200")

    # 2. Initialize Document DB
    if 'search_documents' in active_tools:
         emb_model = doc_search_cfg.get('embedding_model', 'm3')
         doc_persist_dir_with_suffix = f"{DOCUMENT_PERSIST_DIR}_{args.sport}_{emb_model}_all"
         document_tools.init_document_database(
             doc_persist_dir_with_suffix,
             all_data_metadata,
             enabled_sources=enabled_sources,
             model_config=doc_search_cfg,
             es_url=es_url,
             sport=args.sport,
         )

    # 3. Initialize Video DB
    video_db_enabled = "videos" in enabled_sources
    if video_db_enabled:
        # Get search_videos config from arch (contains both embedding_model and embedding_source)
        search_videos_cfg = config['arch'].get('search_videos', {})
        video_emb_model = search_videos_cfg.get('embedding_model', 'm3')
        video_emb_source = search_videos_cfg.get('embedding_source', 'video')

        # Sport-aware persist dir so flag files / caches don't cross-contaminate sports.
        video_persist = f"{VIDEO_PERSIST_DIR}_{args.sport}"

        # Build model config for video database initialization
        tool_model_cfg = {'embedding_model': video_emb_model}

        # If using internvideo2, add its specific paths
        if video_emb_model == 'internvideo2':
            internvideo2_cfg = config['models'].get('embedding_models', {}).get('internvideo2', {})
            tool_model_cfg.update(internvideo2_cfg)
            if 'server' in tool_model_cfg:
                tool_model_cfg['server'] = apply_host_rewrite(tool_model_cfg['server'], role='tool')

        video_tools.init_video_database(
            persist_dir=video_persist,
            data_metadata=all_data_metadata,
            clip_embeddings_base_path=CLIP_EMBEDDINGS_BASE_PATH,
            model_config=tool_model_cfg,
            embedding_source=video_emb_source,
            es_url=es_url,
            sport=args.sport,
        )


    print("Initializing Tool List for Core Agent...")
    function_list = init_tools(config, all_data_metadata)
    print(f"Tools correctly initialized: {[f.name if hasattr(f, 'name') else f.__class__.__name__ for f in function_list]}")

    agent, final_system_prompt = init_agent(config, system_prompt, model_key, function_list)
    
    exp_path = setup_experiment_logging(
        arch_name=config['arch']['name'],
        config_map=config,
        prompt_path=PROMPT_PATH,
        final_prompt_content=final_system_prompt,
        function_list=agent.function_map.values()
    )
    config['experiment_path'] = exp_path
    for tool in function_list:
        if hasattr(tool, 'cfg'):
            tool.cfg['experiment_path'] = exp_path
    
    print("\n" + "="*50)
    print("Agent ready. Ask a sports question to start.")
    print()
    print("Type 'clear' to reset the conversation.")
    print("Type 'exit' (or 'quit') when you're done.")
    print("="*50)

    messages = [] # system prompt is added automatically by agent
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['exit()', 'quit()', 'exit', 'quit']: break
            if user_input.lower() in ['clear', 'clear()']:
                messages = []
                print("\n[SYSTEM]: Conversation history cleared.")
                continue
            if not user_input.strip(): continue

            messages.append(Message(role="user", content=user_input))
            response_generator = agent.run(messages=messages)
             
            turn_messages = []
            printed_chars = {} 
            
            for trn_msgs in response_generator:

                for i, msg in enumerate(trn_msgs):


                    if i not in printed_chars:
                        turn_id = msg.extra.get('turn') if getattr(msg, 'extra', None) else None
                        turn_str = f" (Turn {turn_id})" if turn_id else ""
                        print(f"\n\n[{msg.role.upper()}{turn_str}]:", flush=True)
                        content = msg.content or ""
                        if isinstance(content, list):
                            content = "\n".join([str(item) for item in content])
                        print(content, end="", flush=True)
                        printed_chars[i] = len(content)
                        
                        if msg.function_call:
                            tool_id = msg.extra.get('tool_call_id') if getattr(msg, 'extra', None) else None
                            id_str = f" [ID: {tool_id}]" if tool_id else ""
                            print(f"\n[Tool Call{id_str}]: {msg.function_call.name}({msg.function_call.arguments})", flush=True)

                        token_usage = msg.extra.get('token_usage') if getattr(msg, 'extra', None) else None
                        if token_usage:
                             prompt = token_usage.get('prompt_tokens', 0)
                             completion = token_usage.get('completion_tokens', 0)
                             total = token_usage.get('total_tokens', 0)
                             context_tokens = msg.extra.get('context_tokens', '') if getattr(msg, 'extra', None) else ''
                             ctx_str = f", Context Estimate {context_tokens}" if context_tokens else ""
                             print(f"\n[Usage]: Input {prompt}, Output {completion}, Total {total}{ctx_str}", flush=True)

                             # Print raw usage details if any
                             raw_keys = {k: v for k, v in token_usage.items() if k not in ['prompt_tokens', 'completion_tokens', 'total_tokens']}
                             if raw_keys:
                                 import json
                                 print(f"[Raw Usage Details]: {json.dumps(raw_keys)}", flush=True)
                    else:
                        new_content = msg.content or ""
                        if isinstance(new_content, list):
                            new_content = "\n".join([str(item) for item in new_content])
                            
                        old_len = printed_chars[i]
                        if len(new_content) > old_len:
                            print(new_content[old_len:], end="", flush=True)
                            printed_chars[i] = len(new_content)
                
                turn_messages = trn_msgs

            print()
            
            # Post-process: Check for pruning metadata (Logging only or other side effects if needed)
            # The newly generated messages might contain 'newly_pruned_indices' in extra
            # User requested removal of back-patching 'pruned_turn' logic.
            pass
                                
            messages.extend(turn_messages)
            
            if exp_path:
                with open(os.path.join(exp_path, 'chat_log.json'), 'w') as f:
                    def msg_to_dict(m):
                        if isinstance(m, dict):
                            return m
                        d = m.model_dump()
                        # Always try to copy extra if it exists
                        if hasattr(m, 'extra'):
                             d['extra'] = m.extra
                        return d

                    # Preferred: Get full history from ContextManager
                    if hasattr(agent, 'context_manager'):
                        full_history = agent.context_manager.get_full_history()
                        
                        # Parse tool outputs for structured logging (in-place modification)
                        
                        json_messages = [msg_to_dict(m) for m in full_history]
                    else:
                        # Parse tool outputs (No longer needed, handled at execution time)
                        json_messages = [msg_to_dict(m) for m in messages]
                        
                    json.dump(json_messages, f, indent=2, ensure_ascii=False)
                
                with open(os.path.join(exp_path, 'chat_log.txt'), 'w') as f:
                    for m in messages:
                        extra = getattr(m, 'extra', {}) or {}
                        turn_id = extra.get('turn')
                        turn_str = f" (Turn {turn_id})" if turn_id else ""
                        
                        pruned_at = extra.get('pruned_turn')
                        pruned_str = f" *Pruned_at_turn_{pruned_at}" if pruned_at else ""
                        
                        f.write(f"[{m.role.upper()}{turn_str}{pruned_str}]\n")
                        if m.content:
                            content_str = m.content
                            if isinstance(content_str, list):
                                content_str = "\n".join([str(item) for item in content_str])
                            f.write(content_str + "\n")
                        if m.function_call:
                            f.write(f"[Tool Call]: {m.function_call.name}\nArgs: {m.function_call.arguments}\n")
                        f.write("-" * 40 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
