#!/usr/bin/env python3
"""
Writing Style Evaluation for Generated Game Reports

Uses a thinking LLM (via vLLM) to evaluate writing style, formatting,
persona adherence, narrative coherence, and writing flow.
The prompt is sport-agnostic.

Usage:
  python eval_writing.py \
      --data_dir data/basketball \
      --predictions results/basketball_qwen_aggregated.json \
      --output results/basketball_writing_eval.json
"""

import argparse
import json
import os
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ============================================================================
# Writing Style Prompt (identical across all sports)
# ============================================================================
WRITING_STYLE_PROMPT = """You are an expert sports analyst and a meticulous long-form report evaluation assistant.
Your task is to evaluate how well a Report satisfies the REPORT_INSTRUCTION specifically in terms of writing style, formatting, persona adherence, narrative coherence, and writing flow.

Crucial Constraints:
- You DO NOT generate your own report. You only analyze and score the Pred report.
- You MUST follow a strict Chain-of-Thought (CoT) procedure for every evaluation category below:
  * task_input_analysis: Identify style, formatting, persona, tone, and perspective required by the REPORT_INSTRUCTION.
  * report_analysis: Analyze the Pred report's tone, structure, flow, formatting, and stylistic elements.
  * justification_cot: Explicitly compare the Pred report against the stylistic/formatting instructions. Provide step-by-step reasoning.
  * score: Assign a score from 1 to 5 based on the rubric.

EVALUATION CATEGORIES AND PROCEDURES

## STYLISTIC & PERSONA ADHERENCE

FOCUS:
- Does the report successfully adopt the requested Writing Style (e.g., Analytical, Journalistic, Dramatic)?
- Does it successfully adopt the requested Perspective/Audience (e.g., Scout, Fan, Neutral Analyst)?
- Does it adhere to formatting constraints (word count, paragraph form, bullets, etc.)?

PROCEDURE:
- task_input_analysis: Identify required style, persona, formatting.
- report_analysis: Analyze tone, vocabulary, narrative approach, formatting.
- justification_cot: Evaluate consistency with required style/persona; check formatting constraint adherence.

SCORING GUIDE:
- 5: Perfect stylistic alignment; consistent persona; flawless formatting.
- 4: Strong alignment; minor tone or formatting inconsistencies.
- 3: Mixed or inconsistent style/persona; noticeable formatting problems.
- 2: Minimal adherence; generic tone; major formatting issues.
- 1: Does not follow style/persona/format at all.

## NARRATIVE COHERENCE & QUALITY

FOCUS:
- Coherence: Is the report logically structured?
- Flow: Do ideas transition smoothly?
- Clarity & Fluency: Is writing professional, readable, and non-repetitive?
- Synthesis: Does the report integrate ideas into a cohesive narrative?

PROCEDURE:
- task_input_analysis: N/A
- report_analysis: Evaluate structure, transitions, clarity, grammar, repetition, and synthesis.
- justification_cot: Provide detailed reasoning on coherence, flow, and clarity.

SCORING GUIDE:
- 5: Highly coherent, smooth, and professional; strong synthesis.
- 4: Clear and mostly smooth; minor repetition or structural issues.
- 3: Generally readable but with noticeable disruptions in flow, clarity, or structure.
- 2: Poor flow or clarity; disjointed or difficult to follow.
- 1: Incoherent, broken language; severely unclear.

## FINAL HOLISTIC SCORE

The final_overall score MUST reflect overall stylistic and narrative quality.

ERROR SEVERITY:
- Minor Error: Small style/flow/formatting issues.
- Major Error: Broken flow, inconsistent persona, major formatting failures.
- Critical Error: Completely incoherent or disregards all stylistic and formatting rules.

SCORING GUIDE:
- 5 (Excellent): Highly reliable, coherent, consistent style/persona; only a few minor flaws.
- 4 (Good): Strong overall; may contain several minor errors or one major error.
- 3 (Acceptable): Some value, but multiple major issues or notable weaknesses.
- 2 (Poor): Style/flow significantly flawed; text not useful without heavy revision.
- 1 (Very Poor): Incoherent or completely ignores style/format requests.

OUTPUT FORMAT (STRICT JSON STRUCTURE)
{{
  "stylistic_persona_adherence": {{
    "task_input_analysis": "...",
    "report_analysis": "...",
    "justification_cot": "...",
    "score": X
  }},
  "narrative_coherence_quality": {{
    "task_input_analysis": "...",
    "report_analysis": "...",
    "justification_cot": "...",
    "score": X
  }},
  "final_overall": {{
    "justification_cot": "...",
    "score": X
  }}
}}

Do NOT include any extra keys.
Do NOT output explanations outside the JSON object.
Ensure the JSON is valid.

REPORT TO ANALYZE:
{report}

REPORT_INSTRUCTION:
{instruction}
"""


# ============================================================================
# Helpers
# ============================================================================
def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def resolve_placeholders(text, metadata):
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
    if q_type.startswith("multi_"):
        q_num = q_type.replace("multi_", "")
        return os.path.join(data_dir, "multi_game", q_num, str(sample_id))
    else:
        q_num = q_type.replace("single_", "")
        return os.path.join(data_dir, "single_game", q_num, str(sample_id))


def get_instruction(sample_dir):
    """Read prompt.txt and resolve placeholders from metadata.json."""
    instruction = read_file(os.path.join(sample_dir, "prompt.txt"))
    if not instruction:
        return None
    metadata_path = os.path.join(sample_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        instruction = resolve_placeholders(instruction, metadata)
    return instruction


def extract_scores(response_text):
    """Extract 3 scores from JSON response."""
    result = {
        "stylistic_persona_adherence_score": None,
        "narrative_coherence_quality_score": None,
        "final_overall_score": None,
        "parse_error": None,
    }
    # Try full JSON parse
    try:
        parsed = json.loads(response_text.strip())
        result["stylistic_persona_adherence_score"] = parsed.get("stylistic_persona_adherence", {}).get("score")
        result["narrative_coherence_quality_score"] = parsed.get("narrative_coherence_quality", {}).get("score")
        result["final_overall_score"] = parsed.get("final_overall", {}).get("score")
        return result
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from text
    m = re.search(r"\{[\s\S]*\}", response_text)
    if m:
        try:
            parsed = json.loads(m.group())
            result["stylistic_persona_adherence_score"] = parsed.get("stylistic_persona_adherence", {}).get("score")
            result["narrative_coherence_quality_score"] = parsed.get("narrative_coherence_quality", {}).get("score")
            result["final_overall_score"] = parsed.get("final_overall", {}).get("score")
            return result
        except json.JSONDecodeError as e:
            result["parse_error"] = str(e)
    else:
        result["parse_error"] = "No JSON found in response"
    # Regex fallback
    for key, pattern in [
        ("stylistic_persona_adherence_score", r'"stylistic_persona_adherence"[^}]*"score"\s*:\s*(\d+)'),
        ("narrative_coherence_quality_score", r'"narrative_coherence_quality"[^}]*"score"\s*:\s*(\d+)'),
        ("final_overall_score", r'"final_overall"[^}]*"score"\s*:\s*(\d+)'),
    ]:
        m = re.search(pattern, response_text)
        if m:
            result[key] = int(m.group(1))
    return result


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
    parser = argparse.ArgumentParser(description="Writing Style Evaluation")
    parser.add_argument("--data_dir", required=True, help="Path to sport data dir")
    parser.add_argument("--predictions", required=True, help="Aggregated predictions JSON")
    parser.add_argument("--output", required=True, help="Output evaluation results JSON")
    parser.add_argument("--judge_model", default="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8")
    parser.add_argument("--tensor_parallel", type=int, default=4)
    parser.add_argument("--pipeline_parallel", type=int, default=2,
                        help="Pipeline parallel size (use 2 for A6000 8-GPU, 1 for H100 4-GPU)")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

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
    all_stylistic, all_narrative, all_final = [], [], []

    for q_type in sorted(test_list.keys()):
        agg_key = q_type.replace("single_", "") if q_type.startswith("single_") else q_type
        sample_ids = test_list[q_type]
        results["per_sample"][q_type] = []
        q_stylistic, q_narrative, q_final = [], [], []

        batch_prompts, batch_meta = [], []

        for sid in sample_ids:
            sid_str = str(sid)
            pred_key = agg_key if agg_key in predictions else q_type
            if pred_key not in predictions or sid_str not in predictions.get(pred_key, {}):
                continue

            report = predictions[pred_key][sid_str]
            sample_dir = get_sample_dir(args.data_dir, q_type, sid)
            instruction = get_instruction(sample_dir)
            if not instruction:
                continue

            filled = WRITING_STYLE_PROMPT.format(report=report, instruction=instruction)
            formatted = format_chat(filled)

            token_len = len(tokenizer.encode(formatted))
            if token_len > 32768 * 5:
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
                scores = extract_scores(response)
                result = {"sample_id": meta["sample_id"], "response": response, **scores}
                results["per_sample"][q_type].append(result)
                if scores["stylistic_persona_adherence_score"] is not None:
                    q_stylistic.append(scores["stylistic_persona_adherence_score"])
                    all_stylistic.append(scores["stylistic_persona_adherence_score"])
                if scores["narrative_coherence_quality_score"] is not None:
                    q_narrative.append(scores["narrative_coherence_quality_score"])
                    all_narrative.append(scores["narrative_coherence_quality_score"])
                if scores["final_overall_score"] is not None:
                    q_final.append(scores["final_overall_score"])
                    all_final.append(scores["final_overall_score"])

        if q_final:
            results["summary"][q_type] = {
                "avg_stylistic": sum(q_stylistic) / len(q_stylistic),
                "avg_narrative": sum(q_narrative) / len(q_narrative),
                "avg_final": sum(q_final) / len(q_final),
                "num_samples": len(q_final),
            }
            print(f"  {q_type}: style={sum(q_stylistic)/len(q_stylistic):.2f}, "
                  f"narrative={sum(q_narrative)/len(q_narrative):.2f}, "
                  f"final={sum(q_final)/len(q_final):.2f}")

    if all_final:
        results["summary"]["overall"] = {
            "avg_stylistic": sum(all_stylistic) / len(all_stylistic),
            "avg_narrative": sum(all_narrative) / len(all_narrative),
            "avg_final": sum(all_final) / len(all_final),
            "num_samples": len(all_final),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # Print summary
    if "overall" in results["summary"]:
        o = results["summary"]["overall"]
        print(f"\n{'=' * 40}")
        print("OVERALL SCORES:")
        print(f"  Stylistic & Persona Adherence: {o['avg_stylistic']:.2f}")
        print(f"  Narrative Coherence & Quality: {o['avg_narrative']:.2f}")
        print(f"  Final Overall Score: {o['avg_final']:.2f}")
        print(f"  Samples Evaluated: {o['num_samples']}")
        print("=" * 40)


if __name__ == "__main__":
    main()
