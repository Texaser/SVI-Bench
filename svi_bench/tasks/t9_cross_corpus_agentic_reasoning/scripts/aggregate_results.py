#!/usr/bin/env python
"""
Aggregate results from multiple benchmark workers.
Works with experiments folder structure: experiments/{name}/results/
"""

import os
import json
import re
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")


def _experiments_dir() -> str:
    """Where experiment dirs live: ``$T9_ROOT/results/``. Deferred so import
    doesn't fail when ``T9_ROOT`` isn't set yet."""
    from svi_bench.tasks.t9_cross_corpus_agentic_reasoning._t9_root import require_t9_data_root
    return os.path.join(require_t9_data_root(), "results")

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file. Skips malformed lines with a warning rather than aborting
    — worker crashes can leave a partial last line."""
    results = []
    bad = 0
    with open(file_path, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad += 1
                print(f"  WARNING: skipped malformed line {lineno} in {file_path.name}: {e}")
    if bad:
        print(f"  WARNING: {bad} malformed line(s) skipped in {file_path.name}")
    return results


def aggregate_results(results_dir: str):
    """
    Aggregate results from all worker output files.
    
    Args:
        results_dir: Directory containing results_worker_*.jsonl files
        
    Returns:
        tuple: (Dict with aggregated results, judge_prompt_template)
    """
    results_path = Path(results_dir)
    
    # Load judge prompt
    judge_prompt_path = os.path.join(PROMPTS_DIR, "eval", "judge_prompt.txt")
    if os.path.exists(judge_prompt_path):
        with open(judge_prompt_path, 'r') as f:
            judge_prompt_template = f.read()
    else:
        print(f"Warning: Judge prompt not found at {judge_prompt_path}")
        judge_prompt_template = None

    # Find all worker result files
    result_files = sorted(results_path.glob("results_worker_*.jsonl"))
    
    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    
    print(f"Found {len(result_files)} worker result files")
    
    # Aggregate all results
    all_results = []
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'by_worker': defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0}),
        'total_time': 0,
        'errors': []
    }
    
    for result_file in result_files:
        worker_id = result_file.stem.split('_')[-1]
        print(f"  Processing {result_file.name}...")
        
        worker_results = load_jsonl(result_file)
        
        for result in worker_results:
            if 'answer' in result:
                match = re.search(r'<answer>(.*?)</answer>', result['answer'], re.DOTALL)
                if match:
                    result['answer_parsed'] = match.group(1).strip()
            
            # Filter raw_tool_response to keep only IDs (flattened list)
            for msg in result.get('messages', []):
                if 'extra' in msg and 'raw_tool_response' in msg['extra']:
                    raw_resp = msg['extra']['raw_tool_response']
                    if isinstance(raw_resp, list):
                        filtered_resp = []
                        for item in raw_resp:
                            if isinstance(item, dict):
                                if 'doc_id' in item:
                                    filtered_resp.append(item['doc_id'])
                                elif 'clip_id' in item:
                                    filtered_resp.append(item['clip_id'])
                        
                        # Only update if we extracted IDs, otherwise keep original (or empty if it was list of dicts but no IDs?)
                        # If raw_resp was a list of strings already, this logic skips. 
                        # Assuming raw_resp comes from tools as list of dicts.
                        if filtered_resp:
                            msg['extra']['raw_tool_response'] = filtered_resp

            all_results.append(result)
            stats['total'] += 1
            stats['by_worker'][worker_id]['total'] += 1
            
            if result.get('metadata', {}).get('success', False):
                stats['success'] += 1
                stats['by_worker'][worker_id]['success'] += 1
            else:
                stats['failed'] += 1
                stats['by_worker'][worker_id]['failed'] += 1
                error = result.get('metadata', {}).get('error')
                if error:
                    stats['errors'].append({
                        'question_id': result.get('question_id'),
                        'error': error
                    })
            
            elapsed = result.get('metadata', {}).get('elapsed_time', 0)
            stats['total_time'] += elapsed

    # Dedup on question.id: worker retries can produce multiple rows per question.
    # Prefer successful over failed; if both same status, keep the later (most recent) row.
    def _qid(r: Dict) -> str:
        q = r.get('question') or {}
        return str(q.get('id', r.get('question_id', '')))

    dedup: Dict[str, Dict] = {}
    duplicates_seen = 0
    for r in all_results:
        qid = _qid(r)
        if not qid:
            continue  # no-id rows handled separately below
        prev = dedup.get(qid)
        if prev is None:
            dedup[qid] = r
            continue
        duplicates_seen += 1
        prev_ok = bool(prev.get('metadata', {}).get('success'))
        this_ok = bool(r.get('metadata', {}).get('success'))
        if (this_ok and not prev_ok) or (this_ok == prev_ok):
            dedup[qid] = r

    no_id_rows = [r for r in all_results if not _qid(r)]
    all_results = list(dedup.values()) + no_id_rows

    if duplicates_seen:
        print(f"  NOTE: dedup removed {duplicates_seen} duplicate question_id row(s); "
              f"kept {len(dedup)} unique + {len(no_id_rows)} no-id row(s).")
        stats['duplicates_removed'] = duplicates_seen

    # Sort by global index
    all_results.sort(key=lambda x: x.get('global_index', 0))
    
    # Calculate averages
    if stats['total'] > 0:
        stats['avg_time_per_question'] = stats['total_time'] / stats['total']
        stats['success_rate'] = stats['success'] / stats['total'] * 100
    
    aggregated = {
        'results': all_results,
        'stats': dict(stats)
    }
    
    return aggregated, judge_prompt_template


def generate_batch_input(all_results: List[Dict], judge_prompt_template: str, output_path: str):
    """Generate OpenAI Batch API input file. Also writes a sidecar cache_keys.json
    mapping custom_id -> cache_key (hash of qid+gold+pred) for judge cache lookup."""
    if not judge_prompt_template:
        return

    cache_keys = {}

    with open(output_path, 'w') as f:
        for result in all_results:
            question_data = result.get('question', {})
            question_text = question_data.get('question', '')
            gt_answer = question_data.get('answer', '')
            pred_answer = result.get('answer_parsed', result.get('answer', ''))

            # Construct Prompt
            prompt = judge_prompt_template.format(
                gt_answer=gt_answer,
                pred_answer=pred_answer
            )

            # Construct Batch Request Body
            custom_id = str(question_data.get('id', result.get('question_id', 'unknown')))

            # Cache key stable across reruns with same agent answer
            cache_key = hashlib.sha256(
                f"{custom_id}|||{gt_answer}|||{pred_answer}".encode()
            ).hexdigest()
            cache_keys[custom_id] = cache_key

            request_body = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5.2",
                    "messages": [
                        {"role": "system", "content": "You are a strict grader. Return only a valid JSON object with keys 'verdict' and 'reason'."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_completion_tokens": 200
                }
            }

            f.write(json.dumps(request_body) + '\n')

    # Write sidecar mapping (OpenAI Batch API rejects per-request `metadata` fields)
    sidecar_path = os.path.join(os.path.dirname(output_path), "cache_keys.json")
    with open(sidecar_path, 'w') as sf:
        json.dump(cache_keys, sf, indent=2)

    print(f"Generated Batch Input: {output_path}")
    print(f"Generated Cache Keys:  {sidecar_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--experiment-name", default=None, 
                        help="Name of experiment in experiments/ folder")
    parser.add_argument("--results-dir", default=None, 
                        help="Direct path to results directory (overrides --experiment-name)")
    parser.add_argument("--output-file", default=None, 
                        help="Output file path (default: aggregated_results.json in results dir)")
    parser.add_argument("--stats-only", action="store_true", 
                        help="Only print statistics, don't save")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments")
    args = parser.parse_args()
    
    # List experiments if requested
    if args.list:
        experiments_dir = _experiments_dir()
        print(f"Experiments in {experiments_dir}:\n")
        if os.path.exists(experiments_dir):
            for exp in sorted(os.listdir(experiments_dir)):
                exp_path = os.path.join(experiments_dir, exp)
                if os.path.isdir(exp_path):
                    results_dir = os.path.join(exp_path, "results")
                    result_count = len(list(Path(results_dir).glob("results_worker_*.jsonl"))) if os.path.exists(results_dir) else 0
                    metadata_file = os.path.join(exp_path, "experiment_metadata.json")
                    created = "?"
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            meta = json.load(f)
                            created = meta.get('created_at', '?')[:10]
                    print(f"  {exp}  ({result_count} worker files, created: {created})")
        return
    
    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    elif args.experiment_name:
        results_dir = os.path.join(_experiments_dir(), args.experiment_name, "results")
    else:
        parser.error("Either --experiment-name or --results-dir is required (use --list to see experiments)")
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        return
    
    print(f"Aggregating results from: {results_dir}")
    
    aggregated, judge_template = aggregate_results(results_dir)
    
    # Print statistics
    stats = aggregated['stats']
    print("\n" + "=" * 50)
    print("AGGREGATION COMPLETE")
    print("=" * 50)
    print(f"Total questions: {stats['total']}")
    print(f"Successful: {stats['success']} ({stats.get('success_rate', 0):.1f}%)")
    print(f"Failed: {stats['failed']}")
    print(f"Total time: {stats['total_time']/60:.1f} minutes")
    print(f"Avg time per question: {stats.get('avg_time_per_question', 0):.1f}s")
    
    print("\nBy worker:")
    for worker_id, worker_stats in sorted(stats['by_worker'].items()):
        print(f"  Worker {worker_id}: {worker_stats['total']} total, "
              f"{worker_stats['success']} success, {worker_stats['failed']} failed")
    
    if stats['errors']:
        print(f"\nFirst 5 errors:")
        for err in stats['errors'][:5]:
            print(f"  - {err['question_id']}: {err['error'][:100]}...")
    
    # Save aggregated results
    if not args.stats_only:
        output_file = args.output_file or os.path.join(results_dir, "aggregated_results.json")
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        print(f"Saved aggregated results to: {output_file}")
        
        # Generate Batch Input
        if judge_template:
            batch_input_file = os.path.join(results_dir, "batch_eval_input.jsonl")
            generate_batch_input(aggregated['results'], judge_template, batch_input_file)


if __name__ == "__main__":
    main()
