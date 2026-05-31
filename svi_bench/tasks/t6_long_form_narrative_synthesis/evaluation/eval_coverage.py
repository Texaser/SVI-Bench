#!/usr/bin/env python3
"""
Coverage Evaluation for Generated Game Reports

Uses a thinking LLM (via vLLM) to check whether a generated report covers
each ground-truth atomic fact from a reference fact list.

Usage:
  python eval_coverage.py \
      --sport basketball \
      --data_dir data/basketball \
      --predictions results/basketball_qwen_aggregated.json \
      --output results/basketball_coverage_eval.json
"""

import argparse
import json
import os
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ============================================================================
# Sport-specific prompt sections
# ============================================================================
SPORT_PROMPTS = {
    "basketball": {
        "directionality": "made vs missed, won vs lost, led vs trailed",
        "numbers": "points, rebounds, assists, score, margin, time remaining",
        "timing": "a quarter, half, time remaining, or sequence",
        "specificity_examples": (
            '   - Example: "scored a lot" does NOT cover "scored 28 points"\n'
            '   - Example: "played well" does NOT cover "grabbed 10 rebounds and 5 assists"'
        ),
    },
    "hockey": {
        "directionality": "made vs missed, won vs lost, led vs trailed",
        "numbers": "goals, assists, saves, shots, score, margin, time remaining",
        "timing": "a period, overtime, time remaining, or sequence",
        "specificity_examples": (
            '   - Example: "scored a lot" does NOT cover "scored 3 goals"\n'
            '   - Example: "played well" does NOT cover "made 35 saves and 2 assists"'
        ),
    },
    "soccer": {
        "directionality": "scored vs missed, won vs lost, led vs trailed",
        "numbers": "goals, assists, saves, shots, shots on target, tackles, fouls, yellow cards, red cards, corners, possession percentage, pass completions, interceptions, score, margin, time remaining",
        "timing": "a half, extra time, stoppage time, time remaining, or sequence",
        "specificity_examples": (
            '   - Example: "scored multiple goals" does NOT cover "scored 3 goals"\n'
            '   - Example: "played well" does NOT cover "completed 42 passes and made 5 interceptions"'
        ),
    },
}


def build_coverage_prompt(sport: str) -> str:
    s = SPORT_PROMPTS[sport]
    return f"""You will evaluate the factual coverage of a GENERATED REPORT against a fixed list of GROUND-TRUTH ATOMIC FACTS.

For each ground-truth fact, determine whether the generated report contains the same meaning.

## DEFINITIONS
**Covered**: The generated report explicitly states the fact OR clearly implies it with equivalent meaning, without altering any key details.
**Not Covered**: The fact is missing, too vague to confirm, contradicted, or altered in any key way.

## EQUIVALENCE RULES (STRICT)
1. **Entities**: Correct player/team names are used. Spelling variants are acceptable; completely wrong names are NOT.
2. **Directionality/Polarity**: The meaning is preserved—{s['directionality']}, positive vs negative.
3. **Numbers**: If the fact includes any number ({s['numbers']}), the generated report must match the number EXACTLY. Rounding, approximation, or "about X" does NOT count.
4. **Timing/Period**: If the fact specifies {s['timing']}, the generated report must match this exactly.
5. **Specificity**: Generic statements do NOT cover specific facts.
{s['specificity_examples']}

## PROCEDURE
For each ground-truth fact:
1. Search the generated report for a matching statement.
2. Apply the equivalence rules above.
3. Classify as **Covered** or **Not Covered**.
4. Provide a brief justification.

Format each result as:
-- [Fact text] — [Covered/Not Covered] — [Brief justification]

After all facts, provide:
Overall Score: X/Y
Where X = number of Covered facts and Y = total number of ground-truth facts.

GROUND-TRUTH FACT LIST:
{{fact_list}}

GENERATED REPORT:
{{report}}"""


# ============================================================================
# Helpers
# ============================================================================
def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def get_sample_dir(data_dir, q_type, sample_id):
    if q_type.startswith("multi_"):
        q_num = q_type.replace("multi_", "")
        return os.path.join(data_dir, "multi_game", q_num, str(sample_id))
    else:
        q_num = q_type.replace("single_", "")
        return os.path.join(data_dir, "single_game", q_num, str(sample_id))


def extract_score(response_text):
    m = re.search(r"Overall Score:\s*(\d+)\s*/\s*(\d+)", response_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    covered = len(re.findall(r"—\s*Covered", response_text, re.IGNORECASE))
    not_covered = len(re.findall(r"—\s*Not Covered", response_text, re.IGNORECASE))
    total = covered + not_covered
    return (covered, total) if total > 0 else (None, None)


def format_chat(prompt):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def parse_thinking(output, tokenizer):
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
    parser = argparse.ArgumentParser(description="Coverage Evaluation")
    parser.add_argument("--sport", required=True, choices=["basketball", "hockey", "soccer"])
    parser.add_argument("--data_dir", required=True, help="Path to sport data dir")
    parser.add_argument("--predictions", required=True, help="Aggregated predictions JSON")
    parser.add_argument("--output", required=True, help="Output evaluation results JSON")
    parser.add_argument("--judge_model", default="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8")
    parser.add_argument("--tensor_parallel", type=int, default=4)
    parser.add_argument("--pipeline_parallel", type=int, default=2,
                        help="Pipeline parallel size (use 2 for A6000 8-GPU, 1 for H100 4-GPU)")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    prompt_template = build_coverage_prompt(args.sport)

    test_list_path = os.path.join(args.data_dir, "test_list.json")
    with open(test_list_path) as f:
        test_list = json.load(f)
    with open(args.predictions) as f:
        predictions = json.load(f)

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

    results = {"per_sample": {}, "summary": {}}
    all_covered, all_total_facts = 0, 0

    for q_type in sorted(test_list.keys()):
        agg_key = q_type.replace("single_", "") if q_type.startswith("single_") else q_type
        sample_ids = test_list[q_type]
        results["per_sample"][q_type] = []
        q_covered, q_total = 0, 0

        batch_prompts, batch_meta = [], []

        for sid in sample_ids:
            sid_str = str(sid)
            pred_key = agg_key if agg_key in predictions else q_type
            if pred_key not in predictions or sid_str not in predictions.get(pred_key, {}):
                continue

            report = predictions[pred_key][sid_str]
            sample_dir = get_sample_dir(args.data_dir, q_type, sid)
            fact_list = read_file(os.path.join(sample_dir, "coverage_facts.txt"))
            if not fact_list:
                continue

            filled = prompt_template.format(fact_list=fact_list, report=report)
            formatted = format_chat(filled)

            token_len = len(tokenizer.encode(formatted))
            if token_len > 32768 * 5:
                print(f"  Skipping {q_type}/{sid}: prompt too long ({token_len} tokens)")
                continue

            batch_prompts.append(formatted)
            batch_meta.append({"q_type": q_type, "sample_id": sid})

        print(f"\nProcessing {q_type}: {len(batch_prompts)} samples")
        for i in tqdm(range(0, len(batch_prompts), args.batch_size), desc=q_type):
            chunk_prompts = batch_prompts[i:i + args.batch_size]
            chunk_meta = batch_meta[i:i + args.batch_size]
            outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=False)

            for output, meta in zip(outputs, chunk_meta):
                thinking, response = parse_thinking(output, tokenizer)
                covered, total = extract_score(response)
                score = covered / total if total and total > 0 else None
                result = {
                    "sample_id": meta["sample_id"],
                    "covered": covered, "total": total, "score": score,
                    "response": response,
                }
                results["per_sample"][q_type].append(result)
                if covered is not None and total is not None:
                    q_covered += covered
                    q_total += total

        if q_total > 0:
            results["summary"][q_type] = {
                "total_covered": q_covered, "total_facts": q_total,
                "coverage_rate": q_covered / q_total,
                "num_samples": len([r for r in results["per_sample"][q_type] if r.get("score") is not None]),
            }
            all_covered += q_covered
            all_total_facts += q_total
            print(f"  {q_type}: {q_covered}/{q_total} = {q_covered/q_total:.4f}")

    if all_total_facts > 0:
        results["summary"]["overall"] = {
            "total_covered": all_covered, "total_facts": all_total_facts,
            "coverage_rate": all_covered / all_total_facts,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 40)
    single_qtypes = sorted(k for k in results["summary"] if not k.startswith("multi_") and k != "overall")
    multi_qtypes = sorted(k for k in results["summary"] if k.startswith("multi_"))

    if single_qtypes:
        print("\nSingle-Game Performance:")
        for q in single_qtypes:
            s = results["summary"][q]
            total_samples = len(test_list.get(q, test_list.get("single_" + q.replace("single_", ""), [])))
            print(f"  {q}: {s['coverage_rate']:.4f} ({s['total_covered']}/{s['total_facts']} facts from {s['num_samples']}/{total_samples} samples)")

    if multi_qtypes:
        print("\nMulti-Game Performance:")
        for q in multi_qtypes:
            s = results["summary"][q]
            total_samples = len(test_list.get(q, []))
            print(f"  {q}: {s['coverage_rate']:.4f} ({s['total_covered']}/{s['total_facts']} facts from {s['num_samples']}/{total_samples} samples)")

    if "overall" in results["summary"]:
        o = results["summary"]["overall"]
        print(f"\n{'=' * 40}")
        print(f"OVERALL COVERAGE SCORE: {o['coverage_rate']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
