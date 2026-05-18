#!/usr/bin/env python
"""
Orchestrate the evaluation pipeline:
1. Aggregate results from experiment.
2. Submit batch job to OpenAI, wait for results, and run analysis.
"""

import os
import argparse
import subprocess
import sys

# Base paths
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
AGGREGATE_SCRIPT = os.path.join(SCRIPTS_DIR, "aggregate_results.py")
ANALYZE_SCRIPT = os.path.join(SCRIPTS_DIR, "analyze_results.py")

def main():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline: Aggregate -> Analyze (Batch)")
    parser.add_argument("--experiment-name", required=True, help="Name of experiment in experiments/ folder")
    parser.add_argument("--skip-aggregate", action="store_true", help="Skip aggregation step")
    parser.add_argument("--reuse-eval", action="store_true",
                        help="Reuse existing batch_eval_output.jsonl even if its row count doesn't match batch_eval_input.jsonl")

    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    
    print(f"Starting evaluation pipeline for: {experiment_name}")
    print("="*60)

    # 1. Aggregate Results
    if not args.skip_aggregate:
        print("\n[Step 1] Aggregating Results...")
        cmd_agg = [
            "python", AGGREGATE_SCRIPT,
            "--experiment-name", experiment_name
        ]
        print(f"Running: {' '.join(cmd_agg)}")
        try:
            subprocess.run(cmd_agg, check=True)
            print("[Step 1] Aggregation complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error during aggregation: {e}")
            sys.exit(1)
    else:
        print("\n[Step 1] Skipping Aggregation.")

    # 2. Analyze Results (Submit Batch + Wait)
    print("\n[Step 2] Analysis (Batch Submission & Analysis)...")
    
    # Path to check for existing results
    base_dir = os.path.dirname(SCRIPTS_DIR)
    results_dir = os.path.join(base_dir, "experiments", experiment_name, "results")
    eval_output_file = os.path.join(results_dir, "batch_eval_output.jsonl")
    
    cmd_analyze = ["python", ANALYZE_SCRIPT]

    # Freshness check: count batch input/output rows. If they differ, the cached
    # output is stale (e.g., new workers were added since last submit).
    batch_input_file = os.path.join(results_dir, "batch_eval_input.jsonl")

    def _count_lines(p: str) -> int:
        if not os.path.exists(p):
            return -1
        with open(p, 'r') as f:
            return sum(1 for line in f if line.strip())

    n_in = _count_lines(batch_input_file)
    n_out = _count_lines(eval_output_file)

    if n_out >= 0 and n_in >= 0 and n_in == n_out:
        print(f"Found existing evaluation results at {eval_output_file} "
              f"({n_out} lines, matches batch input). Skipping batch submission.")
    elif n_out >= 0 and args.reuse_eval:
        print(f"WARNING: {eval_output_file} has {n_out} lines but batch input has {n_in}. "
              f"Reusing anyway (--reuse-eval).")
    elif n_out >= 0:
        print(f"WARNING: {eval_output_file} exists ({n_out} lines) but batch input has {n_in} lines. "
              f"Resubmitting batch. Pass --reuse-eval to skip submission.")
        cmd_analyze.extend(["--submit-batch", "--wait"])
    else:
        cmd_analyze.extend(["--submit-batch", "--wait"])
        
    cmd_analyze.append("--experiment-name")
    cmd_analyze.append(experiment_name)
    
    print(f"Running: {' '.join(cmd_analyze)}")
    try:
        subprocess.run(cmd_analyze, check=True)
        print("[Step 2] Analysis complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()
