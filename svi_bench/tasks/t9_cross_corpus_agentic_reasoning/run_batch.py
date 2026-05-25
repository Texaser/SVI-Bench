#!/usr/bin/env python
"""
Batch processing entry point for benchmark evaluation.
Reuses initialization from run_agent.py but replaces interactive loop with batch processing.

Results are saved to: $T9_ROOT/results/{experiment_name}/
"""

import os
import sys
import argparse
import json
import json5
import time
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Add module path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from _t9_root import (
    T9_ROOT_NOT_SET,
    resolve_t9_data_root as _resolve_t9_data_root,
    require_t9_data_root as _require_t9_data_root,
    resolve_t9_results_dir as _resolve_t9_results_dir,
)


T9_DATA_ROOT = _resolve_t9_data_root()

# Experiment results land next to the task code by default
# (``<task_dir>/results/``); override with the ``T9_RESULTS`` env var.
EXPERIMENTS_DIR = _resolve_t9_results_dir()

# Import initialization functions from run_agent.py
from run_agent import (
    load_config,
    load_prompt,
    load_data_metadata,
    generate_env_desc,
    init_tools,
    init_agent,
    build_llm_cfg,
    get_model_config,
)
from qwen_agent.llm.schema import Message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_questions(questions_path: str, start_idx: int, end_idx: Optional[int]) -> List[Dict]:
    """Load questions from JSON file and return subset for this worker."""
    with open(questions_path, 'r') as f:
        all_questions = json.load(f)
    
    # Handle different formats
    if isinstance(all_questions, dict):
        # If dict with 'questions' key
        all_questions = all_questions.get('questions', list(all_questions.values()))
    
    if end_idx is None:
        end_idx = len(all_questions)
    
    end_idx = min(end_idx, len(all_questions))
    
    logger.info(f"Loading questions {start_idx} to {end_idx} (total: {len(all_questions)})")
    return all_questions[start_idx:end_idx]

def get_checkpoint_path(output_dir: str, worker_id: int) -> Path:
    """Get path to checkpoint file for this worker."""
    return Path(output_dir) / f"checkpoint_worker_{worker_id}.json"

def load_checkpoint(output_dir: str, worker_id: int) -> set:
    """Load set of already processed question IDs."""
    checkpoint_path = get_checkpoint_path(output_dir, worker_id)
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return set(json.load(f))
    return set()

def save_checkpoint(output_dir: str, worker_id: int, processed_ids: set):
    """Save set of processed question IDs."""
    checkpoint_path = get_checkpoint_path(output_dir, worker_id)
    with open(checkpoint_path, 'w') as f:
        json.dump(list(processed_ids), f)

def get_question_id(question: Dict, idx: int) -> str:
    """Extract or generate a unique ID for the question."""
    # Try common ID fields
    for key in ['id', 'question_id', 'qid', 'index']:
        if key in question:
            return str(question[key])
    # Fallback to index
    return str(idx)

def run_single_question(agent, question: Dict, config: Dict) -> Dict:
    """
    Run the agent on a single question and return the result.
    
    Returns:
        Dict with keys: question, answer, messages, metadata
    """
    # Extract question text
    if isinstance(question, str):
        query = question
    else:
        # Try common keys for question text
        query = question.get('question') or question.get('query') or question.get('text') or str(question)
    
    # Prepare messages
    messages = [Message(role="user", content=query)]
    
    # Run agent
    start_time = time.time()
    
    try:
        response_generator = agent.run(messages=messages)
        
        # Collect all messages from the generator
        final_messages = []
        for turn_messages in response_generator:
            final_messages = turn_messages
        
        # Extract final answer from assistant messages
        answer = ""
        for msg in reversed(final_messages):
            if msg.role == "assistant" and msg.content:
                content = msg.content
                if isinstance(content, list):
                    content = "\n".join([str(item) for item in content])
                answer = content
                break
        
        elapsed_time = time.time() - start_time
        
        # Capture full history for logs
        dumped_messages = []
        if hasattr(agent, 'context_manager'):
             full_history = agent.context_manager.get_full_history()
             for m in full_history:
                 if isinstance(m, dict):
                     dumped_messages.append(m)
                 else:
                     d = m.model_dump()
                     # specific for Qwen Message which might store extra in .extra
                     if getattr(m, 'extra', None):
                         d['extra'] = m.extra
                     dumped_messages.append(d)
        else:
             # Fallback to response accumulator (missing user query context in log, but better than nothing)
             dumped_messages = [msg.model_dump() for msg in final_messages]

        return {
            "question": question,
            "answer": answer,
            "messages": dumped_messages,
            "metadata": {
                "elapsed_time": elapsed_time,
                "num_turns": len(final_messages),
                "success": True,
                "error": None
            }
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error processing question: {e}")
        
        return {
            "question": question,
            "answer": None,
            "messages": [],
            "metadata": {
                "elapsed_time": elapsed_time,
                "num_turns": 0,
                "success": False,
                "error": str(e)
            }
        }


def save_question_logs(chats_dir: str, question_id: str, messages: List[Dict], 
                       system_prompt: str = None, tool_prompt: str = None):
    """Save detailed chat logs for a single question."""
    # Ensure directory exists (redundant if done in main, but safe)
    os.makedirs(chats_dir, exist_ok=True)
    
    filename_base = f"chat_{question_id}"
    
    # Save JSON log (just the messages list, as system prompt is included in messages)
    def msg_to_dict_local(m):
        if isinstance(m, dict):
            return m
        d = m.model_dump()
        if hasattr(m, 'extra'):
             d['extra'] = m.extra
        return d

    json_messages = [msg_to_dict_local(m) for m in messages]
    with open(os.path.join(chats_dir, f'{filename_base}.json'), 'w') as f:
        json_str = json.dumps(json_messages, indent=2, ensure_ascii=False)
        # Handle surrogate characters from API responses that can't be encoded in UTF-8
        f.write(json_str.encode('utf-8', errors='replace').decode('utf-8'))
    
    # Save human-readable TXT log
    def _safe(s):
        """Strip surrogate characters that can't be encoded in UTF-8."""
        return str(s).encode('utf-8', errors='replace').decode('utf-8')

    with open(os.path.join(chats_dir, f'{filename_base}.txt'), 'w') as f:
        # Write conversation
        for m in messages:
            role = m.get('role', 'unknown').upper()
            extra = m.get('extra', {}) or {}
            turn_id = extra.get('turn')
            turn_str = f" (Turn {turn_id})" if turn_id else ""
            f.write(f"[{role}{turn_str}]\n")

            # Show truncation notice if this is a forced final answer
            forced_final = extra.get('forced_final')
            if forced_final:
                label = 'Hard Limit Exceeded' if forced_final == 'hard_limit' else 'Max Turns Reached'
                f.write(f"[FORCED FINAL: {label}]\n")
                notice = extra.get('truncation_notice', '')
                if notice:
                    f.write(_safe(notice) + "\n")
                f.write("=" * 40 + "\n")

            content = m.get('content', '')
            if content:
                if isinstance(content, list):
                    content = "\n".join([str(item) for item in content])
                f.write(_safe(content) + "\n")

            # Check for tool/function calls
            func_call = m.get('function_call')
            if func_call:
                name = func_call.get('name', 'unknown')
                args = func_call.get('arguments', '{}')
                f.write(f"[Tool Call]: {name}\nArgs: {_safe(args)}\n")

            f.write("-" * 40 + "\n")

            # Log Token Usage if available
            token_usage = extra.get('token_usage')
            if token_usage:
                prompt = token_usage.get('prompt_tokens', 0)
                completion = token_usage.get('completion_tokens', 0)
                total = token_usage.get('total_tokens', 0)
                f.write(f"[Usage]: Input {prompt}, Output {completion}, Total {total}\n")

                # Print raw usage details if any
                raw_keys = {k: v for k, v in token_usage.items() if k not in ['prompt_tokens', 'completion_tokens', 'total_tokens']}
                if raw_keys:
                    f.write(f"[Raw Usage Details]: {json.dumps(raw_keys)}\n")

                f.write("-" * 40 + "\n")


def append_result(result: Dict, output_dir: str, worker_id: int):
    """Append a single result to the worker's output file (JSONL format)."""
    output_path = Path(output_dir) / f"results_worker_{worker_id}.jsonl"
    with open(output_path, 'a') as f:
        json_str = json.dumps(result, ensure_ascii=False)
        # Handle surrogate characters from API responses that can't be encoded in UTF-8
        f.write(json_str.encode('utf-8', errors='replace').decode('utf-8') + '\n')



def setup_experiment(experiment_name: Optional[str], arch: str, questions_file: str) -> str:
    """
    Create experiment directory with config snapshots.

    Args:
        experiment_name: Optional name, defaults to timestamp
        arch: Architecture name
        questions_file: Path to questions file

    Returns:
        Path to experiment directory
    """
    # Generate experiment name if not provided
    if not experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"batch_{arch}_{timestamp}"
    
    # Create experiment directory
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "configs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "chats"), exist_ok=True)
    
    # Copy config files for reproducibility
    config_files = [
        f"archs/{arch}.yaml",
        "configs/models.yaml",
        "configs/hyperparameters.yaml",
        "configs/paths.yaml",
    ]
    for cfg_file in config_files:
        src = os.path.join(BASE_DIR, cfg_file)
        if os.path.exists(src):
            dst = os.path.join(exp_dir, "configs", os.path.basename(cfg_file))
            shutil.copy2(src, dst)
    
    # Save experiment metadata
    metadata = {
        "experiment_name": experiment_name,
        "architecture": arch,
        "questions_file": questions_file,
        "created_at": datetime.now().isoformat(),
        "base_dir": BASE_DIR,
    }
    with open(os.path.join(exp_dir, "experiment_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir


def initialize_pipeline(arch: str, sport: str = None) -> tuple:
    """
    Initialize the agent pipeline (reuses run_agent.py logic).

    Args:
        arch: Architecture config name (e.g., "gpt5").
        sport: Sport name for filtering searches (e.g., "hockey", "soccer").

    Returns:
        tuple: (agent, config, function_list)
    """
    split = "all"  # Public release ships split=all only; legacy param removed.
    CONFIG_MODELS_PATH = os.path.join(BASE_DIR, "configs/models.yaml")
    CONFIG_HYPER_PATH = os.path.join(BASE_DIR, "configs/hyperparameters.yaml")
    CONFIG_PATHS_PATH = os.path.join(BASE_DIR, "configs/paths.yaml")
    ARCH_PATH = os.path.join(BASE_DIR, f"archs/{arch}.yaml")

    if not os.path.exists(ARCH_PATH):
        raise ValueError(f"Architecture config not found: {ARCH_PATH}")

    logger.info("Loading configs...")
    config = {
        "models": load_config(CONFIG_MODELS_PATH),
        "hyperparams": load_config(CONFIG_HYPER_PATH),
        "paths": load_config(CONFIG_PATHS_PATH),
        "arch": load_config(ARCH_PATH)
    }
    if sport:
        config['sport'] = sport

    # Override server URLs for multi-node setups
    tool_server_host = os.environ.get('T9_TOOL_SERVER_HOST')
    if tool_server_host:
        logger.info(f"T9_TOOL_SERVER_HOST={tool_server_host} — replacing localhost in tool/embedding model servers")
        for section in ['tool_models', 'embedding_models']:
            for key, model_cfg in config['models'].get(section, {}).items():
                if 'server' in model_cfg and 'localhost' in model_cfg['server']:
                    model_cfg['server'] = model_cfg['server'].replace('localhost', tool_server_host)

    agent_server_host = os.environ.get('T9_AGENT_SERVER_HOST')
    if agent_server_host:
        logger.info(f"T9_AGENT_SERVER_HOST={agent_server_host} — replacing localhost in agent model server")
        agent_key = config['arch'].get('agent', {}).get('model_key')
        if agent_key and agent_key in config['models'].get('agent_models', {}):
            model_cfg = config['models']['agent_models'][agent_key]
            if 'server' in model_cfg and 'localhost' in model_cfg['server']:
                model_cfg['server'] = model_cfg['server'].replace('localhost', agent_server_host)

    # Resolve paths to absolute
    path_keys = [
        'data_base_path', 'clip_embeddings_base_path', 'video_persist_dir',
        'document_persist_dir',
    ]
    _t9_root_runtime = _require_t9_data_root()
    for key in path_keys:
        val = config['paths'].get(key)
        if val and not os.path.isabs(val):
            config['paths'][key] = os.path.join(_t9_root_runtime, val)

    # Load prompt
    PROMPT_PATH = os.path.join(BASE_DIR, config['arch']['agent'].get('prompt'))
    system_prompt = load_prompt(PROMPT_PATH)

    # Load model config
    model_key = config['arch']['agent']['model_key']

    # Load datasets
    enabled_datasets = config.get('hyperparams', {}).get('data', {}).get('enabled_datasets', [])
    if not enabled_datasets:
        raise ValueError("No datasets enabled in hyperparameters.yaml")

    DATA_BASE_PATH = config['paths']['data_base_path']
    datasets_metadata = load_data_metadata(DATA_BASE_PATH, enabled_datasets)

    # Flatten metadata
    all_data_metadata = {}
    for dataset, data_metadata in datasets_metadata.items():
        for item_id, item_data in data_metadata.items():
            item_data['sport'] = dataset
            all_data_metadata[item_id] = item_data

    logger.info(f"Total items loaded: {len(all_data_metadata)} (split={split})")

    # Generate environment description
    env_desc = generate_env_desc(config)
    if "{{env_desc}}" in system_prompt:
        system_prompt = system_prompt.replace("{{env_desc}}", env_desc)

    # Set EMBEDDING_GPUS from arch config for bge-m3 GPU distribution
    embedding_gpus = config['arch'].get('services', {}).get('m3', {}).get('gpus', [])
    if embedding_gpus:
        os.environ.setdefault('EMBEDDING_GPUS', ','.join(map(str, embedding_gpus)))

    # Initialize databases (documents and videos)
    from tools import document_tools, video_tools

    active_tools = config['arch']['tools']
    enabled_sources = config.get('hyperparams', {}).get('data', {}).get('enabled_sources', [])
    es_url = os.environ.get('T9_ES_URL') or config.get('hyperparams', {}).get('elasticsearch', {}).get('url', "http://localhost:9200")

    # Document DB
    if 'search_documents' in active_tools:
        doc_search_cfg = config['arch'].get('search_documents', {})
        emb_model = doc_search_cfg.get('embedding_model', 'm3')
        if sport:
            doc_persist_dir = f"{config['paths']['document_persist_dir']}_{sport}_{emb_model}_{split}"
        else:
            doc_persist_dir = f"{config['paths']['document_persist_dir']}_{emb_model}_{split}"
        document_tools.init_document_database(
            doc_persist_dir, all_data_metadata,
            enabled_sources=enabled_sources,
            model_config=doc_search_cfg,
            es_url=es_url,
            split=split,
            sport=sport
        )

    # Video DB
    if "videos" in enabled_sources:
        search_videos_cfg = config['arch'].get('search_videos', {})
        video_emb_model = search_videos_cfg.get('embedding_model', 'm3')
        video_emb_source = search_videos_cfg.get('embedding_source', 'video')

        tool_model_cfg = {'embedding_model': video_emb_model}
        if video_emb_model == 'internvideo2':
            internvideo2_cfg = config['models'].get('embedding_models', {}).get('internvideo2', {})
            tool_model_cfg.update(internvideo2_cfg)
            if 'server' in tool_model_cfg:
                from run_agent import apply_host_rewrite
                tool_model_cfg['server'] = apply_host_rewrite(tool_model_cfg['server'], role='tool')

        if sport:
            video_persist = f"{config['paths']['video_persist_dir']}_{sport}"
        else:
            video_persist = config['paths']['video_persist_dir']
        video_tools.init_video_database(
            persist_dir=video_persist,
            data_metadata=all_data_metadata,
            clip_embeddings_base_path=config['paths']['clip_embeddings_base_path'],
            model_config=tool_model_cfg,
            embedding_source=video_emb_source,
            es_url=es_url,
            split=split,
            sport=sport
        )

    # Initialize tools and agent
    logger.info("Initializing tools...")
    function_list = init_tools(config, all_data_metadata)

    logger.info("Initializing agent...")
    agent, _ = init_agent(config, system_prompt, model_key, function_list)

    # Get tool prompt from agent (generated during init)
    tool_prompt = getattr(agent, 'tool_prompt', None)

    return agent, config, function_list, system_prompt, tool_prompt


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch processing for benchmark evaluation")
    parser.add_argument("--questions", required=True, help="Path to questions JSON file")
    parser.add_argument("--experiment-name", default=None,
                        help="Experiment name (default: auto-generated timestamp)")
    parser.add_argument("--experiment-dir", default=None,
                        help="Full path to experiment dir (overrides --experiment-name)")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end-idx", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--arch", default="gpt5", help="Architecture config name")
    parser.add_argument("--sport", type=str, default=None,
                        choices=["basketball", "hockey", "soccer"],
                        help="Sport for this evaluation run (auto-filters all searches)")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for this process")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser


def main_as_function(args: argparse.Namespace) -> Dict:
    """Importable batch entry. Drives the agent loop over a question slice and
    writes per-question outputs under the experiment dir.

    Two invocation paths share this function:
      1. Interactive Python — ``from ...run_batch import main_as_function``.
      2. SLURM wrapper — ``python -m ...run_batch ...`` -> __main__ -> here.

    The LLM judge / aggregation step is handled separately by
    ``evaluate.run()`` (and the underlying ``scripts/aggregate_results.py`` +
    ``scripts/analyze_results.py``).
    """

    # Determine experiment directory
    if args.experiment_dir:
        # Use provided full path
        exp_dir = args.experiment_dir
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "chats"), exist_ok=True)
    else:
        # Setup experiment directory (only worker 0 creates full structure)
        if args.worker_id == 0:
            exp_dir = setup_experiment(args.experiment_name, args.arch, args.questions)
        else:
            # Other workers use the same experiment directory
            if args.experiment_name:
                exp_dir = os.path.join(EXPERIMENTS_DIR, args.experiment_name)
            else:
                # Workers need experiment name when not worker 0
                raise ValueError("Non-zero workers need --experiment-name or --experiment-dir")
    
    # Results go in results/ subdirectory
    results_dir = os.path.join(exp_dir, "results")
    logs_dir = os.path.join(exp_dir, "logs")
    chats_dir = os.path.join(exp_dir, "chats")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(chats_dir, exist_ok=True)
    
    # Setup logging to file
    log_file = os.path.join(logs_dir, f"worker_{args.worker_id}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"=== Batch Worker {args.worker_id} Starting ===")
    logger.info(f"Experiment: {exp_dir}")
    logger.info(f"Questions: {args.questions}")
    logger.info(f"Range: {args.start_idx} to {args.end_idx}")
    logger.info(f"Architecture: {args.arch}")
    logger.info(f"Sport: {args.sport}")
    logger.info(f"Resume: {args.resume}")

    # Initialize pipeline
    agent, config, function_list, system_prompt, tool_prompt = initialize_pipeline(args.arch, sport=args.sport)
    
    # Add experiment path to tool configs
    config['experiment_path'] = exp_dir
    for tool in function_list:
        if hasattr(tool, 'cfg'):
            tool.cfg['experiment_path'] = exp_dir
    
    # Load questions
    questions = load_questions(args.questions, args.start_idx, args.end_idx)
    
    # Load checkpoint if resuming
    processed_ids = set()
    if args.resume:
        processed_ids = load_checkpoint(results_dir, args.worker_id)
        if processed_ids:
            logger.info(f"Resuming from saved progress: {len(processed_ids)} questions already processed")
        else:
            logger.info("No saved progress found. Starting from scratch.")
    
    # Process questions
    total_questions = len(questions)
    start_time = time.time()
    
    for idx, question in enumerate(questions):
        global_idx = args.start_idx + idx
        question_id = get_question_id(question, global_idx)

        # Skip if already processed
        if question_id in processed_ids:
            logger.info(f"[{idx+1}/{total_questions}] Skipping {question_id} (already processed)")
            continue

        logger.info(f"[{idx+1}/{total_questions}] Processing question {question_id}")

        try:
            # Run agent
            result = run_single_question(agent, question, config)
            result['question_id'] = question_id
            result['global_index'] = global_idx

            # Save result to results/ subdirectory
            append_result(result, results_dir, args.worker_id)

            # Save individual logs
            save_question_logs(chats_dir, question_id, result['messages'],
                              system_prompt, tool_prompt)

            # Update checkpoint
            processed_ids.add(question_id)
            save_checkpoint(results_dir, args.worker_id, processed_ids)

            # Log progress
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (total_questions - idx - 1)

            logger.info(f"  → Success: {result['metadata']['success']}, "
                       f"Time: {result['metadata']['elapsed_time']:.1f}s, "
                       f"ETA: {remaining/60:.1f}min")
        except Exception as e:
            logger.error(f"[{idx+1}/{total_questions}] Fatal error on question {question_id}: {e}", exc_info=True)
            logger.info(f"  → Skipping question {question_id} and moving to next")
    
    total_time = time.time() - start_time
    logger.info(f"=== Worker {args.worker_id} Complete ===")
    logger.info(f"Processed: {len(processed_ids)} questions in {total_time/60:.1f} minutes")

    return {
        "experiment_dir": exp_dir,
        "worker_id": args.worker_id,
        "processed": len(processed_ids),
        "total": total_questions,
        "elapsed_seconds": total_time,
        "arch": args.arch,
        "sport": args.sport,
    }


if __name__ == "__main__":
    _args = _build_argparser().parse_args()
    _result = main_as_function(_args)
    print(json.dumps(_result, indent=2))

