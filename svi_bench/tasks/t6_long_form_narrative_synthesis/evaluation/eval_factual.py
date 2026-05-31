#!/usr/bin/env python3
"""
Factual Correctness Evaluation for Generated Game Reports

Uses a thinking LLM (via vLLM) to extract atomic facts from a generated report
and verify each against ground-truth resources (game report, stats, play-by-play log).

Usage:
  python eval_factual.py \
      --sport basketball \
      --data_dir data/basketball \
      --predictions results/basketball_qwen_aggregated.json \
      --output results/basketball_factual_eval.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ============================================================================
# Sport-specific prompt sections
# ============================================================================
SPORT_PROMPTS = {
    "basketball": {
        "sport_name": "basketball game",
        "stats_list": "points, rebounds, assists, steals, fouls, turnovers, shooting makes/misses, and explicitly stated quantifiable performance outcomes",
        "events_list": "scoring plays, lead changes, momentum plays, clutch shots, defensive events, blocks, steals, or any other on-court actions",
        "action_location": "on-court",
        "example_input": '-Input: "Player A scored 20 points and had 5 assists."',
        "example_facts": """- Team A won the game.
- The game result was 100-95.
- Player A scored 30 points.
- Player B grabbed 10 rebounds.
- Player C made a layup with 2:12 remaining.
- Player D shot 8-for-15 from the field.
- Team A led 58-52 at halftime.
- Team B made 3-of-8 three-pointers.""",
        "gt_resources": "(1) the official game report, (2) the period-by-period box score statistics, (3) play-by-play Game Log",
        "report_label": "Game Report To Be Evaluated",
        "gt_label": "Ground Truth Game Report",
        "stats_label": "Period-by-Period Stats",
        "log_label": "Play-by-Play Game Log",
    },
    "hockey": {
        "sport_name": "hockey game",
        "stats_list": "goals, assists, shots on goal, saves, penalties, faceoffs won/lost, hits, turnovers, power play goals, short-handed goals, shooting percentage",
        "events_list": "goals, assists, saves, hits, penalties, faceoffs, power plays, or any other on-ice actions",
        "action_location": "on-ice",
        "example_input": '-Input: "Player A made 5 shots and scored 1 goal."',
        "example_facts": """- The game result was 4-3.
- Player A scored 2 goals.
- Player B made 1 assist.
- Player D made 3 blocks.
- Team B attempted 3 shots on goal in first period.""",
        "gt_resources": "(1) the official game report, (2) the period-by-period box score statistics, (3) play-by-play Game Log",
        "report_label": "Game Report To Be Evaluated",
        "gt_label": "Ground Truth Game Report",
        "stats_label": "Period-by-Period Stats",
        "log_label": "Play-by-Play Game Log",
    },
    "soccer": {
        "sport_name": "soccer match",
        "stats_list": "goals, assists, shots, shots on target, saves, tackles, interceptions, fouls, yellow cards, red cards, corners, free kicks, penalties, offsides, possession percentage, pass completion",
        "events_list": "goals, assists, defensive plays, saves, tackles, fouls, card incidents (yellow/red), set pieces (corners, free kicks, penalties), offsides, substitutions, or any other on-field actions",
        "action_location": "on-field",
        "example_input": '-Input: "Player A scored 2 goals and had 1 assist."',
        "example_facts": """- Team A won the match.
- The match result was 3-1.
- Player C made a save with 15 minutes remaining.
- Player D received a yellow card.
- Team A led 2-0 at halftime.
- Team B attempted 3 shots on target in first half.""",
        "gt_resources": "(1) the official match report, (2) the half-by-half box score statistics, (3) the play-by-play match log",
        "report_label": "Match Report To Be Evaluated",
        "gt_label": "Ground Truth Match Report",
        "stats_label": "Half-by-Half Stats",
        "log_label": "Play-by-Play Match Log",
    },
}


def build_factual_prompt(sport: str) -> str:
    """Build the factual correctness evaluation prompt for a given sport."""
    s = SPORT_PROMPTS[sport]
    return f"""You will perform a two-step task on {s['sport_name']} text. Your job is (1) to extract atomic, observable in-game facts, and then (2) to verify each fact against the provided ground-truth resources.

STEP 1 — EXTRACT OBSERVABLE IN-GAME FACTS

Read the report below and decompose every factual claim into the smallest possible atomic statements.

Target types of facts:
1. **Statistics**: {s['stats_list']}.
2. **Key Events**: {s['events_list']}.
3. **Game Context**: scores at specific points, lead changes, game-deciding moments, and similar contextual details.

Rules:
- Each atomic statement must be independently verifiable.
- Do NOT extract opinions, subjective assessments, or predictions.
- Preserve exact numbers, names, and details from the original text.
- If a sentence contains multiple facts, split them into separate statements.
- If a fact is ambiguous or vague, still extract it and note the ambiguity.

Example:
{s['example_input']}
Atomic statements:
- Player A scored 20 points.
- Player A had 5 assists.

Now extract ALL atomic facts from the following report. List each fact on a new line, prefixed with "- ".

Facts:
{s['example_facts']}

STEP 2 — VERIFY EXTRACTED FACTS AGAINST GROUND TRUTH

For each extracted fact, classify it as:
- **Supported**: The fact is confirmed by the ground-truth resources.
- **Contradicted**: The fact directly conflicts with the ground-truth resources.
- **Inconclusive**: The fact cannot be confirmed or denied by the available ground-truth resources.

Use the following ground-truth resources: {s['gt_resources']}.

Verification rules:
- A fact is "Supported" ONLY if the ground-truth resources explicitly confirm it OR it can be directly computed/inferred from the data.
- A fact is "Contradicted" if the ground-truth resources provide a different value or directly negate the claim.
- A fact is "Inconclusive" if the ground-truth resources do not contain enough information to confirm or deny.
- When checking statistics, verify against the box score. When checking events, verify against the play-by-play log.

For each fact, provide:
-- [Fact text] — [Supported/Contradicted/Inconclusive] — [Brief justification]

After verifying all facts, provide a summary line:
Overall Score: X/Y
Where X = number of Supported facts and Y = total number of Supported facts + Contradicted facts. Do not include Inconclusive facts in the score.

FINAL TEMPLATE

{s['report_label']}:
{{report}}

{s['gt_label']}:
{{ground_truth}}

{s['stats_label']}:
{{stats}}

{s['log_label']}:
{{log}}"""


# ============================================================================
# Helpers
# ============================================================================
def read_file(path):
    """Read file contents or return empty string."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def resolve_placeholders(text, metadata):
    """Replace template placeholders in prompt with values from metadata."""
    replacements = {
        "{player}": metadata.get("selected_player"),
        "{title}": metadata.get("title"),
        "{attribute}": metadata.get("attribute"),
        "{timeframe}": metadata.get("timeframe"),
        "{event_description}": metadata.get("event_description"),
    }
    for placeholder, value in replacements.items():
        if value is not None:
            text = text.replace(placeholder, str(value))
    return text


def get_sample_dir(data_dir, q_type, sample_id):
    """Get component folder path: single_Q1 → single_game/Q1/<id>."""
    if q_type.startswith("multi_"):
        q_num = q_type.replace("multi_", "")
        return os.path.join(data_dir, "multi_game", q_num, str(sample_id))
    else:
        q_num = q_type.replace("single_", "")
        return os.path.join(data_dir, "single_game", q_num, str(sample_id))


def extract_score(response_text):
    """Extract X/Y from 'Overall Score: X/Y'."""
    m = re.search(r"Overall Score:\s*(\d+)\s*/\s*(\d+)", response_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Fallback: count Supported/Contradicted lines (exclude Inconclusive)
    supported = len(re.findall(r"—\s*\*{0,2}Supported", response_text, re.IGNORECASE))
    contradicted = len(re.findall(r"—\s*\*{0,2}Contradicted", response_text, re.IGNORECASE))
    total = supported + contradicted
    return (supported, total) if total > 0 else (None, None)


def format_chat(prompt):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def parse_thinking(output, tokenizer):
    """Separate thinking content from response using </think> token."""
    token_ids = list(output.outputs[0].token_ids)
    full_text = output.outputs[0].text
    try:
        idx = len(token_ids) - token_ids[::-1].index(151668)
        thinking = tokenizer.decode(token_ids[:idx], skip_special_tokens=True).strip()
        response = tokenizer.decode(token_ids[idx:], skip_special_tokens=True).strip()
    except ValueError:
        if "<think>" in full_text and "</think>" in full_text:
            start = full_text.find("<think>") + len("<think>")
            end = full_text.find("</think>")
            thinking = full_text[start:end].strip()
            response = full_text[end + len("</think>"):].strip()
        else:
            thinking, response = "", full_text.strip()
    return thinking, response


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Factual Correctness Evaluation")
    parser.add_argument("--sport", required=True, choices=["basketball", "hockey", "soccer"])
    parser.add_argument("--data_dir", required=True, help="Path to sport data dir (e.g., data/basketball)")
    parser.add_argument("--predictions", required=True, help="Aggregated predictions JSON {q_type: {id: text}}")
    parser.add_argument("--output", required=True, help="Output evaluation results JSON")
    parser.add_argument("--judge_model", default="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8")
    parser.add_argument("--tensor_parallel", type=int, default=4)
    parser.add_argument("--pipeline_parallel", type=int, default=2,
                        help="Pipeline parallel size (use 2 for A6000 8-GPU, 1 for H100 4-GPU)")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    prompt_template = build_factual_prompt(args.sport)

    # Load test list and predictions
    test_list_path = os.path.join(args.data_dir, "test_list.json")
    with open(test_list_path) as f:
        test_list = json.load(f)
    with open(args.predictions) as f:
        predictions = json.load(f)
    print(f"Loaded predictions: {list(predictions.keys())}")

    # Load vLLM model
    # A6000 (8 GPUs): --tensor_parallel 4 --pipeline_parallel 2
    # H100  (4 GPUs): --tensor_parallel 4 --pipeline_parallel 1
    print(f"Loading judge model: {args.judge_model}")
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel,
        pipeline_parallel_size=args.pipeline_parallel,
        gpu_memory_utilization=0.90,
        max_model_len=32768 * 5,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=32768, stop=["<|im_end|>"],
    )

    # Process all samples
    results = {"per_sample": {}, "summary": {}}
    all_scores = []

    for q_type in sorted(test_list.keys()):
        agg_key = q_type.replace("single_", "").replace("multi_", "multi_")
        if q_type.startswith("single_"):
            agg_key = q_type.replace("single_", "")
        else:
            agg_key = q_type  # multi_Q1 stays multi_Q1

        sample_ids = test_list[q_type]
        results["per_sample"][q_type] = []
        q_scores = []

        # Prepare batch
        batch_prompts = []
        batch_meta = []

        for sid in sample_ids:
            sid_str = str(sid)
            pred_key = agg_key if agg_key in predictions else q_type
            if pred_key not in predictions or sid_str not in predictions.get(pred_key, {}):
                continue

            report = predictions[pred_key][sid_str]
            sample_dir = get_sample_dir(args.data_dir, q_type, sid)
            gt = read_file(os.path.join(sample_dir, "ground_truth_report.txt"))
            stats = read_file(os.path.join(sample_dir, "stats.txt"))
            log = read_file(os.path.join(sample_dir, "log.txt"))
            if not gt:
                continue

            filled = prompt_template.format(report=report, ground_truth=gt, stats=stats, log=log)
            formatted = format_chat(filled)

            token_len = len(tokenizer.encode(formatted))
            if token_len > 32768 * 5:
                print(f"  Skipping {q_type}/{sid}: prompt too long ({token_len} tokens)")
                continue

            batch_prompts.append(formatted)
            batch_meta.append({"q_type": q_type, "sample_id": sid})

        # Run inference in batches
        print(f"\nProcessing {q_type}: {len(batch_prompts)} samples")
        for i in tqdm(range(0, len(batch_prompts), args.batch_size), desc=q_type):
            chunk_prompts = batch_prompts[i:i + args.batch_size]
            chunk_meta = batch_meta[i:i + args.batch_size]
            outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=False)

            for output, meta in zip(outputs, chunk_meta):
                thinking, response = parse_thinking(output, tokenizer)
                supported, total = extract_score(response)
                score = supported / total if total and total > 0 else None
                result = {
                    "sample_id": meta["sample_id"],
                    "supported": supported, "total": total, "score": score,
                    "response": response,
                }
                results["per_sample"][q_type].append(result)
                if score is not None:
                    q_scores.append(score)
                    all_scores.append(score)

        if q_scores:
            results["summary"][q_type] = {
                "avg_score": sum(q_scores) / len(q_scores),
                "num_samples": len(q_scores),
            }
            print(f"  {q_type}: avg={sum(q_scores)/len(q_scores):.4f} ({len(q_scores)} samples)")

    if all_scores:
        results["summary"]["overall"] = {
            "avg_score": sum(all_scores) / len(all_scores),
            "num_samples": len(all_scores),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    single_qtypes = sorted(k for k in results["summary"] if k.startswith("single_") or (not k.startswith("multi_") and k != "overall"))
    multi_qtypes = sorted(k for k in results["summary"] if k.startswith("multi_"))

    if single_qtypes:
        print("\nSingle-Game Performance:")
        for q in single_qtypes:
            s = results["summary"][q]
            samples = results["per_sample"].get(q, [])
            total_supported = sum(r.get("supported", 0) for r in samples if r.get("score") is not None)
            total_facts = sum(r.get("total", 0) for r in samples if r.get("score") is not None)
            print(f"  {q}: {s['avg_score']:.4f} ({s['num_samples']}/{len(test_list.get(q, test_list.get('single_' + q.replace('single_', ''), [])))} samples) | {total_supported}/{total_facts} facts supported")

    if multi_qtypes:
        print("\nMulti-Game Performance:")
        for q in multi_qtypes:
            s = results["summary"][q]
            samples = results["per_sample"].get(q, [])
            total_supported = sum(r.get("supported", 0) for r in samples if r.get("score") is not None)
            total_facts = sum(r.get("total", 0) for r in samples if r.get("score") is not None)
            print(f"  {q}: {s['avg_score']:.4f} ({s['num_samples']}/{len(test_list.get(q, []))} samples) | {total_supported}/{total_facts} facts supported")

    if "overall" in results["summary"]:
        o = results["summary"]["overall"]
        print(f"\nOverall Average Score: {o['avg_score']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
