#!/usr/bin/env python
"""
Compute QA accuracy from aggregated results + judge output.
Also handles OpenAI Batch API submission and retrieval for evaluation.
"""

import os
import json
import argparse
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI

# Dual import: works both when run as a script (`python scripts/analyze_results.py`)
# and when imported as a package module (`from ...scripts.analyze_results`).
try:
    from svi_bench.tasks.t9_cross_corpus_agentic_reasoning.scripts._judge_utils import parse_judge_verdict
except ImportError:
    from _judge_utils import parse_judge_verdict


# -----------------------------------------------------------------------------
# JUDGE CACHE
# -----------------------------------------------------------------------------
def _cache_path(results_dir: str) -> str:
    return os.path.join(results_dir, "judge_cache.json")


def load_judge_cache(results_dir: str) -> dict:
    p = _cache_path(results_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_judge_cache(results_dir: str, cache: dict):
    with open(_cache_path(results_dir), 'w') as f:
        json.dump(cache, f, indent=2)


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _experiments_dir() -> str:
    """Where experiment dirs live: ``$T9_ROOT/results/``. Deferred so import
    doesn't fail when ``T9_ROOT`` isn't set yet."""
    from svi_bench.tasks.t9_cross_corpus_agentic_reasoning._t9_root import require_t9_data_root
    return os.path.join(require_t9_data_root(), "results")

def load_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)

def load_jsonl(path: str) -> List[Dict]:
    results = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

class BatchManager:
    """Manages OpenAI Batch API interactions."""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def submit_batch(self, input_file_path: str) -> str:
        """Uploads input file and submits a batch job."""
        print(f"Uploading file: {input_file_path}")
        with open(input_file_path, "rb") as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        file_id = batch_input_file.id
        print(f"File uploaded. ID: {file_id}")
        
        print("Creating batch job...")
        batch_job = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        batch_id = batch_job.id
        print(f"Batch job created. ID: {batch_id}")
        print(f"Status: {batch_job.status}")
        return batch_id

    def retrieve_batch(self, batch_id: str) -> Any:
        """Retrieves the current status of a batch job."""
        return self.client.batches.retrieve(batch_id)

    def download_results(self, file_id: str, output_path: str):
        """Downloads the result file content."""
        print(f"Downloading results from file ID: {file_id}")
        content = self.client.files.content(file_id).content
        
        with open(output_path, 'wb') as f:
            f.write(content)
        print(f"Results saved to: {output_path}")

    def submit_batch_with_cache(self, input_file_path: str, cache: dict,
                                cache_key_by_custom_id: dict) -> tuple:
        """
        Filter out already-cached entries from input file before submitting.
        cache_key_by_custom_id is loaded from the cache_keys.json sidecar
        (since OpenAI Batch API rejects per-request `metadata` fields).

        Returns (batch_id, filtered_count).
        If no new entries to submit, returns (None, 0).
        """
        filtered_path = input_file_path + ".filtered"
        new_count = 0
        skipped = 0
        with open(input_file_path) as fin, open(filtered_path, 'w') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                req = json.loads(line)
                custom_id = req.get('custom_id', '')
                cache_key = cache_key_by_custom_id.get(custom_id, '')
                if cache_key and cache_key in cache:
                    skipped += 1
                    continue
                fout.write(line + '\n')
                new_count += 1

        print(f"Cache: {skipped} hit, {new_count} to submit")

        if new_count == 0:
            print("All entries in cache — nothing to submit.")
            try: os.remove(filtered_path)
            except: pass
            return None, 0

        batch_id = self.submit_batch(filtered_path)
        return batch_id, new_count

    def wait_for_batch(self, batch_id: str, poll_interval: int = 60) -> str:
        """Polls the batch job until it is completed or failed."""
        print(f"Waiting for batch {batch_id} to complete...")
        while True:
            batch = self.retrieve_batch(batch_id)
            status = batch.status
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")
            
            if status in ['completed', 'failed', 'cancelled', 'expired']:
                if status == 'completed' and batch.output_file_id:
                    return batch.output_file_id
                elif status == 'completed' and not batch.output_file_id:
                     print("Batch completed but no output file ID found.")
                     return None
                else:
                    print(f"Batch ended with status: {status}")
                    if batch.errors:
                        print(f"Errors: {batch.errors}")
                    return None
            
            time.sleep(poll_interval)


class Analyzer:
    def __init__(self, results_path: str, eval_output_path: str = None,
                 judge_cache: dict = None, cache_key_by_custom_id: dict = None):
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")

        self.results_file = results_path
        self.results_data = load_json(results_path)
        self.results = self.results_data.get('results', [])

        judge_cache = judge_cache or {}
        cache_key_by_custom_id = cache_key_by_custom_id or {}

        # eval_results: custom_id -> {'verdict': 'Right'|'Wrong', 'reason': str}
        self.eval_results = {}

        # 1) Hydrate from cache (cache is keyed by cache_key)
        for cid, ck in cache_key_by_custom_id.items():
            if ck in judge_cache:
                self.eval_results[cid] = judge_cache[ck]

        # 2) Overlay batch output (newer than cache)
        self.judge_parse_failures = []   # (custom_id, content_snippet) — verdict unparseable
        self.judge_response_errors = []  # (custom_id, error_str) — response had no content
        if eval_output_path and os.path.exists(eval_output_path):
            print(f"Loading evaluation results from {eval_output_path}")
            eval_lines = load_jsonl(eval_output_path)
            for item in eval_lines:
                custom_id = item.get("custom_id")
                try:
                    choice = item['response']['body']['choices'][0]['message']['content']
                except (KeyError, IndexError, TypeError) as e:
                    self.judge_response_errors.append((custom_id, str(e)))
                    continue
                parsed = parse_judge_verdict(choice)
                if parsed:
                    self.eval_results[custom_id] = parsed
                else:
                    self.judge_parse_failures.append((custom_id, (choice or '')[:200]))

            if self.judge_response_errors:
                print(f"WARNING: {len(self.judge_response_errors)} judge responses had no content; "
                      f"first few: {self.judge_response_errors[:3]}")
            if self.judge_parse_failures:
                print(f"WARNING: {len(self.judge_parse_failures)} judge outputs were unparseable; "
                      f"first few custom_ids: {[c for c, _ in self.judge_parse_failures[:5]]}")

    def compute_accuracy(self):
        if not self.eval_results and not self.judge_parse_failures and not self.judge_response_errors:
            return None

        correct = 0
        judged = 0
        unjudged = 0  # parse-failed, response-errored, or missing entirely

        for result in self.results:
            qid = str(result.get('question', {}).get('id'))
            if qid in self.eval_results:
                grade = self.eval_results[qid]
                verdict = grade.get('verdict') if isinstance(grade, dict) else grade
                if verdict == 'Right':
                    correct += 1
                judged += 1
            else:
                unjudged += 1

        total = judged + unjudged
        if total == 0:
            return 0.0
        if unjudged:
            print(f"WARNING: {unjudged}/{total} questions have no usable judge verdict "
                  f"(parse-failed, error, or missing). These count as Wrong in the denominator.")
        return correct / total * 100

    def _accuracy_counts(self):
        correct = 0
        judged = 0
        unjudged = 0
        for result in self.results:
            qid = str(result.get('question', {}).get('id'))
            if qid in self.eval_results:
                grade = self.eval_results[qid]
                verdict = grade.get('verdict') if isinstance(grade, dict) else grade
                if verdict == 'Right':
                    correct += 1
                judged += 1
            else:
                unjudged += 1
        return correct, judged, unjudged

    def print_metrics(self):
        accuracy = self.compute_accuracy()

        print("\n" + "=" * 50)
        print("EVALUATION RESULT")
        print("=" * 50)

        if accuracy is None:
            print("QA Accuracy: N/A (Run with --eval-output to compute)")
            return

        correct, judged, unjudged = self._accuracy_counts()
        total = judged + unjudged
        print(f"QA Accuracy: {accuracy:.2f}%  ({correct}/{total})")

        output_path = 'analysis_metrics.json'
        if self.results_file:
            output_dir = os.path.dirname(self.results_file)
            if os.path.exists(output_dir):
                output_path = os.path.join(output_dir, 'analysis_metrics.json')

        with open(output_path, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'correct': correct,
                'judged': judged,
                'total': total,
            }, f, indent=2)
        print(f"Saved metrics to {output_path}")


def score_aggregated(
    aggregated_results_path: str,
    output_dir: str,
    *,
    submit_if_missing: bool,
) -> Dict | None:
    """Score one run's aggregated output, optionally submitting the OpenAI
    Batch judge first when no ``batch_eval_output.jsonl`` exists.

    Args:
        aggregated_results_path: path to ``aggregated_results.json`` produced
            by ``aggregate_results.aggregate_results()``.
        output_dir: directory holding (or where to write) ``batch_eval_input.jsonl``,
            ``batch_eval_output.jsonl``, ``judge_cache.json``, ``cache_keys.json``.
        submit_if_missing: if True and no ``batch_eval_output.jsonl`` exists,
            submit a fresh batch and block until it finishes. If False and the
            output is missing, return None without spending.

    Returns the metrics dict ``{'accuracy', 'correct', 'judged', 'total'}``
    or None if scoring couldn't run (no output and no submit).
    """
    if not os.path.exists(aggregated_results_path):
        raise FileNotFoundError(f"aggregated_results.json not found: {aggregated_results_path}")

    submit_input_file = os.path.join(output_dir, "batch_eval_input.jsonl")
    eval_output_path = os.path.join(output_dir, "batch_eval_output.jsonl")
    sidecar = os.path.join(output_dir, "cache_keys.json")

    judge_cache = load_judge_cache(output_dir)
    cache_key_by_custom_id: Dict[str, str] = {}
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            cache_key_by_custom_id = json.load(f)

    if not os.path.exists(eval_output_path):
        if not submit_if_missing:
            return None

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to submit the judge batch.")
        if not os.path.exists(submit_input_file):
            raise FileNotFoundError(
                f"batch_eval_input.jsonl not found at {submit_input_file}. "
                f"Run aggregate_results.aggregate_results() first."
            )

        manager = BatchManager(api_key)
        batch_id, _ = manager.submit_batch_with_cache(
            submit_input_file, judge_cache, cache_key_by_custom_id
        )
        if batch_id is not None:
            output_file_id = manager.wait_for_batch(batch_id)
            if not output_file_id:
                raise RuntimeError("Batch job failed or returned no output.")
            manager.download_results(output_file_id, eval_output_path)

        # Merge fresh batch results into cache (best-effort; mirrors __main__ logic)
        if os.path.exists(eval_output_path):
            new_entries = 0
            for item in load_jsonl(eval_output_path):
                custom_id = item.get('custom_id')
                ck = cache_key_by_custom_id.get(custom_id)
                if not ck:
                    continue
                try:
                    content = item['response']['body']['choices'][0]['message']['content']
                except (KeyError, IndexError, TypeError):
                    continue
                parsed = parse_judge_verdict(content)
                if parsed:
                    judge_cache[ck] = parsed
                    new_entries += 1
            if new_entries:
                save_judge_cache(output_dir, judge_cache)

    eval_output = eval_output_path if os.path.exists(eval_output_path) else None
    analyzer = Analyzer(
        aggregated_results_path, eval_output,
        judge_cache=judge_cache,
        cache_key_by_custom_id=cache_key_by_custom_id,
    )
    accuracy = analyzer.compute_accuracy()
    if accuracy is None:
        return None
    correct, judged, unjudged = analyzer._accuracy_counts()
    return {
        'accuracy': accuracy,
        'correct': correct,
        'judged': judged,
        'total': judged + unjudged,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Analysis arguments
    parser.add_argument("--results-file", help="Path to aggregated_results.json")
    parser.add_argument("--eval-output", help="Path to OpenAI Batch API output file (JSONL)")
    parser.add_argument("--experiment-name", help="Name of experiment in experiments/ folder")

    # Batch API arguments
    parser.add_argument("--submit-batch", action="store_true", help="Submit batch job (requires --results-file or --experiment-name)")
    parser.add_argument("--batch-id", help="Existing batch ID to check or retrieve")
    parser.add_argument("--retrieve-batch", action="store_true", help="Retrieve results for the specified --batch-id")
    parser.add_argument("--wait", action="store_true", help="Wait for batch completion (polling)")
    parser.add_argument("--output-dir", default=None, help="Directory to save batch results")
    
    args = parser.parse_args()
    
    # Resolve paths based on experiment name
    results_file = args.results_file
    submit_input_file = None
    output_dir = args.output_dir

    if args.experiment_name:
        # Experiment dirs live at $T9_ROOT/results/<name>/ (matches what
        # submit_experiment.sh writes).
        exp_dir = os.path.join(_experiments_dir(), args.experiment_name)
        results_dir = os.path.join(exp_dir, "results")
        
        if not results_file:
            results_file = os.path.join(results_dir, "aggregated_results.json")
        
        if not output_dir:
            output_dir = results_dir
            
        submit_input_file = os.path.join(results_dir, "batch_eval_input.jsonl")
    
    # Fallback to current dir or results file dir
    if not output_dir:
        if results_file:
            output_dir = os.path.dirname(os.path.abspath(results_file))
        else:
            output_dir = os.getcwd()

    # Load existing judge cache (keyed by cache_key hash)
    judge_cache = load_judge_cache(output_dir)
    if judge_cache:
        print(f"Loaded judge cache: {len(judge_cache)} entries from {_cache_path(output_dir)}")

    # Load custom_id -> cache_key mapping from sidecar file written by aggregate_results.py
    cache_key_by_custom_id = {}
    if submit_input_file:
        sidecar = os.path.join(os.path.dirname(submit_input_file), "cache_keys.json")
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                cache_key_by_custom_id = json.load(f)
            print(f"Loaded {len(cache_key_by_custom_id)} cache keys from {sidecar}")

    # Handle Batch Operations
    if args.submit_batch or args.retrieve_batch or (args.batch_id and args.wait):
        if not OPENAI_API_KEY or "..." in OPENAI_API_KEY:
            # Fallback to env var if variable is not set
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY is not defined in script or environment.")
                exit(1)
        else:
            api_key = OPENAI_API_KEY

        manager = BatchManager(api_key)
        batch_id = args.batch_id

        # 1. Submit (with cache filter)
        if args.submit_batch:
            if not submit_input_file:
                if args.results_file:
                    submit_input_file = os.path.join(os.path.dirname(args.results_file), "batch_eval_input.jsonl")

            if not submit_input_file or not os.path.exists(submit_input_file):
                print(f"Error: Input file not found: {submit_input_file}")
                print("Please ensure aggregate_results.py has generated 'batch_eval_input.jsonl' or specify paths manually.")
                exit(1)

            batch_id, new_count = manager.submit_batch_with_cache(
                submit_input_file, judge_cache, cache_key_by_custom_id
            )

            if batch_id is None:
                # Everything cached — no batch submitted. Proceed to analysis with cache only.
                args.wait = False
            elif not args.wait:
                print(f"Batch submitted. ID: {batch_id}")
                print(f"Use --batch-id {batch_id} --retrieve-batch --wait to get results later.")

        # 2. Wait / Poll
        if args.wait and batch_id:
            output_file_id = manager.wait_for_batch(batch_id)
            if output_file_id:
                output_path = os.path.join(output_dir, "batch_eval_output.jsonl")
                manager.download_results(output_file_id, output_path)
                args.eval_output = output_path
            else:
                print("Batch failed or no output generated.")
                exit(1)

        # 3. Retrieve (if not just waiting)
        elif args.retrieve_batch and batch_id:
            batch = manager.retrieve_batch(batch_id)
            print(f"Batch Status: {batch.status}")
            if batch.status == 'completed' and batch.output_file_id:
                output_path = os.path.join(output_dir, "batch_eval_output.jsonl")
                manager.download_results(batch.output_file_id, output_path)
                args.eval_output = output_path

        # 4. Merge fresh batch results into cache
        if args.eval_output and os.path.exists(args.eval_output):
            new_entries = 0
            cache_response_errors = 0
            cache_parse_failures = 0
            for item in load_jsonl(args.eval_output):
                custom_id = item.get('custom_id')
                ck = cache_key_by_custom_id.get(custom_id)
                if not ck:
                    continue
                try:
                    content = item['response']['body']['choices'][0]['message']['content']
                except (KeyError, IndexError, TypeError):
                    cache_response_errors += 1
                    continue
                parsed = parse_judge_verdict(content)
                if parsed:
                    judge_cache[ck] = parsed
                    new_entries += 1
                else:
                    cache_parse_failures += 1
            if new_entries:
                save_judge_cache(output_dir, judge_cache)
                print(f"Saved {new_entries} new verdicts to cache ({len(judge_cache)} total).")
            if cache_response_errors or cache_parse_failures:
                print(f"NOTE: skipped {cache_response_errors} response-error and "
                      f"{cache_parse_failures} parse-failure rows during cache merge.")

    # Run Analysis if results provided
    if results_file:
        eval_output = args.eval_output
        if not eval_output:
            possible_eval = os.path.join(output_dir, "batch_eval_output.jsonl")
            if os.path.exists(possible_eval):
                print(f"Auto-detected batch output: {possible_eval}")
                eval_output = possible_eval

        analyzer = Analyzer(
            results_file, eval_output,
            judge_cache=judge_cache,
            cache_key_by_custom_id=cache_key_by_custom_id,
        )
        analyzer.print_metrics()
    elif not (args.submit_batch or args.retrieve_batch or args.batch_id):
        parser.print_help()
