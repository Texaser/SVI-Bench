from evaluation.models.model import Model
from evaluation.prompts import *

import argparse
import json
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import time
import os
import random
from tqdm import tqdm

OPENROUTER_API_KEY = None

def get_sport(league: str) -> str:
    if league in {"NBA", "NCAA", "EuroLeague"}:
        return "basketball"
    elif league in {"NHL"}:
        return "hockey"
    else:
        return "soccer"

def get_video_path(item: dict, video_root: str) -> str:
    """Resolve video path using the relative video_path field and a root directory."""
    rel_path = item.get("video_path", "")
    if not rel_path:
        return None
    path = os.path.join(video_root, rel_path) if video_root else rel_path
    if os.path.exists(path):
        return path
    return None

def parse_answers(text: str, max_answers: int = 5) -> list[str]:
    answers = [
        a.strip()
        for a in re.findall(
            r"Answer (?:\d+|[iv]+):\s*(.+?)(?=Answer (?:\d+|[iv]+):|$)",
            text,
            re.DOTALL,
        )
    ]

    return answers[:max_answers]

def llm_judge(question: str, answer: str, pred: str) -> int:
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    
    base = {"question": question, "answer": answer, "pred": pred}

    for i in range(5):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": JUDGE_PROMPT.format(question, answer, pred)}
                ],
                temperature=1.0,
                top_p=0.95,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "output",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "gt_analysis": {"type": "string"},
                                "pred_analysis": {"type": "string"},
                                "justification_cot": {"type": "string"},
                                "score": {"type": "integer"}
                            },
                            "required": ["gt_analysis", "pred_analysis", "justification_cot", "score"],
                            "additionalProperties": False
                        }
                    }
                },
                stream=False,
                extra_body={
                    "reasoning": {"enabled": True},
                    "provider": {"require_parameters": True}
                }
            )
            content = response.choices[0].message.content
            return base | json.loads(content)
        except Exception as e:
            print(f"LLM judge attempt {i+1} failed: {e}")
            time.sleep(60)
    
    return base

def score_answers(question: str, answer: str, answers: list[str]) -> int:
    if len(answers) == 0:
        return 0, []

    score = -1

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(llm_judge, question, answer, pred) for pred in answers]
        results = [f.result() for f in futures]

    for result in results:
        cur = result.get("score", -1)
        score = max(score, cur)

    return score, results

def process_qa(path: str, model: Model, video_root: str = "", max_answers=5):

    with open(path, 'r') as f:
        qa = json.load(f)

    trace = []

    for item in tqdm(qa, desc=f"{model.name()}, {path}"):
        try:
            response = model.process_question(EVAL_PROMPT.format(item["question"], max_answers), get_video_path(item, video_root))
            answers = parse_answers(response, max_answers=max_answers)
            score, output = score_answers(item["question"], item["answer"], answers)
            trace += output
        except Exception as e:
            print(f"Exception occured: {e}")
            answers = []
            score = -1

        if "responses" not in item:
            item["responses"] = dict()
        item["responses"][model.name()] = answers

        if "score" not in item:
            item["score"] = dict()
        item["score"][model.name()] = score
    
    return qa, trace

def evaluate(model: Model, video_root: str = "", max_answers=5, subset=False):

    out_dir = f"outputs/{model.name()}/k-{max_answers}"
    os.makedirs(out_dir, exist_ok=True)

    qa, trace = process_qa("dataset/qa.json" if not subset else "dataset/qa_subset.json", model, video_root=video_root, max_answers=max_answers)

    with open(f"{out_dir}/llm_judge_trace.json", 'w') as f:
        json.dump(trace, f, indent=4)
    with open(f"{out_dir}/results.json", 'w') as f:
        json.dump(qa, f, indent=4)

    scores = dict()

    for item in qa:
        sport = get_sport(item["league"])
        score = item.get("score", dict()).get(model.name(), -1)
        if sport not in scores:
            scores[sport] = []
        scores[sport].append(score)

    total_score = 0
    total_succeeded = 0

    for sport in scores:
        sport_score = 0
        sport_succeeded = 0
        for score in scores[sport]:
            if score > -1:
                total_score += score
                total_succeeded += 1
                sport_score += score
                sport_succeeded += 1

        with open(f"{out_dir}/results.txt", 'a') as f:
            f.write(f"{sport} score: {sport_score / sport_succeeded if sport_succeeded > 0 else -1}, {sport_succeeded}/{len(scores[sport])} questions succeeded\n")
    
    with open(f"{out_dir}/results.txt", 'a') as f:
        f.write(f"overall score: {total_score / total_succeeded if total_succeeded > 0 else -1}, {total_succeeded}/{len(qa)} questions succeeded\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T4 Strategic Reasoning QA Evaluation")
    parser.add_argument("--video_root", default="", help="Root directory to prepend to relative video paths")
    parser.add_argument("--model", default="qwen", choices=["qwen", "molmo", "gpt", "gemini"],
                        help="Model to evaluate")
    parser.add_argument("--model_key", default="", help="API key for GPT/Gemini models")
    parser.add_argument("--max_answers", type=int, default=5, help="Max answers per question")
    parser.add_argument("--subset", action="store_true", help="Use qa_subset.json instead of qa.json")
    args = parser.parse_args()

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Please export it, e.g.\n"
            "  export OPENROUTER_API_KEY=your_key_here"
        )

    if args.model == "molmo":
        from evaluation.models.molmo import Molmo
        evaluate(Molmo(), video_root=args.video_root, max_answers=args.max_answers, subset=args.subset)
    elif args.model == "qwen":
        from evaluation.models.qwen import Qwen
        evaluate(Qwen(), video_root=args.video_root, max_answers=args.max_answers, subset=args.subset)
    elif args.model == "gpt":
        from evaluation.models.gpt import GPT
        evaluate(GPT(key=args.model_key), video_root=args.video_root, max_answers=args.max_answers, subset=args.subset)
    elif args.model == "gemini":
        from evaluation.models.gemini import Gemini
        evaluate(Gemini(key=args.model_key), video_root=args.video_root, max_answers=args.max_answers, subset=args.subset)