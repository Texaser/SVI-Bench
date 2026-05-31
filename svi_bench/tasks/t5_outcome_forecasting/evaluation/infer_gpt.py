#!/usr/bin/env python3
"""
GPT Forecasting Inference Script (OpenAI Responses API)

Runs GPT models on sports forecasting MCQ data using the OpenAI Responses API.
Extracts frames from video on-the-fly using decord (no pre-extracted frames needed).
Extracts logprobs for Brier score calculation.

Requires: openai, decord, numpy, tqdm, Pillow

Usage:
  # Set your API key
  export OPENAI_API_KEY="sk-..."

  # Run inference
  python infer_gpt.py --test_json data/basketball_test.json --output results.json

  # With custom settings
  python infer_gpt.py --test_json data/hockey_test.json --output results.json \
      --model gpt-4o --frame_fps 0.5 --image_detail low
"""

import argparse
import io
import json
import math
import os
import re
import base64
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from openai import OpenAI
from decord import VideoReader, cpu
from PIL import Image


# =========================
# Video frame extraction
# =========================
def fps_sample_indices(num_total: int, video_fps: float, sample_fps: float) -> np.ndarray:
    """Sample frame indices at a target FPS rate."""
    if num_total <= 0:
        raise ValueError("Video has zero frames.")
    duration_seconds = num_total / video_fps
    num_samples = max(1, int(duration_seconds * sample_fps))
    if num_samples == 1:
        return np.array([0], dtype=np.int64)
    frame_interval = video_fps / sample_fps
    indices = np.arange(0, num_total, frame_interval, dtype=np.float64)
    indices = np.round(indices).astype(np.int64)
    indices = np.clip(indices, 0, num_total - 1)
    indices = np.unique(indices)
    return indices


def extract_frames_as_base64(video_path: str, sample_fps: float = 0.5,
                              jpeg_quality: int = 80) -> list:
    """Extract frames from video and return as base64-encoded JPEG strings."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    total = len(vr)
    video_fps = vr.get_avg_fps()
    idx = fps_sample_indices(total, video_fps, sample_fps)
    batch = vr.get_batch(idx).asnumpy()

    b64_frames = []
    for frame_arr in batch:
        img = Image.fromarray(frame_arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        b64_frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return b64_frames


# =========================
# MCQ helpers
# =========================
def infer_mcq_options_from_text(question_text: str) -> list:
    """Infer MCQ options (A-E) from question text."""
    text = question_text.upper()
    found = set(m.group(1) for m in re.finditer(r"\b([A-E])\s*:", text))
    if not found:
        raise ValueError("Could not infer MCQ options")
    order = ["A", "B", "C", "D", "E"]
    return [o for o in order if o in found]


# =========================
# Brier score
# =========================
def brier_score_multiclass(option_probs: dict, gt_letter: str, options: list) -> float:
    K = len(options)
    s = sum((float(option_probs.get(opt, 0.0)) - (1.0 if opt == gt_letter else 0.0)) ** 2
            for opt in options)
    return s / K


def brier_random_baseline(options: list) -> float:
    K = len(options)
    return (K - 1) / (K * K)


# =========================
# Logprob extraction (Responses API)
# =========================
def get_assistant_output_text_item(body: dict):
    """Find the assistant message's output_text item from response body."""
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            return None
    if not isinstance(body, dict):
        return None
    for item in body.get("output", []) or []:
        if item.get("type") == "message" and item.get("role") == "assistant":
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text":
                    return c
    return None


def extract_logprobs_from_response(response, options) -> dict:
    """Extract option logprobs from a Responses API response object."""
    try:
        body = response.model_dump() if hasattr(response, "model_dump") else response
        text_item = get_assistant_output_text_item(body)
        if not text_item:
            return {}
        lps = text_item.get("logprobs") or []
        for tokpos in lps:
            top = tokpos.get("top_logprobs") or []
            found = {}
            for alt in top:
                t = (alt.get("token") or "").strip()
                if t in options and t not in found:
                    found[t] = alt.get("logprob")
            if found:
                return found
        return {}
    except Exception:
        return {}


def get_prediction_text(response) -> str:
    """Extract prediction text from a Responses API response."""
    try:
        body = response.model_dump() if hasattr(response, "model_dump") else response
        text_item = get_assistant_output_text_item(body)
        return text_item.get("text", "") if text_item else ""
    except Exception:
        return ""


def compute_probs_from_logprobs(logprobs: dict, options: list) -> dict:
    """Convert logprobs to normalized probabilities."""
    if not logprobs:
        return {opt: 1.0 / len(options) for opt in options}
    option_logprobs = [logprobs.get(opt, -100.0) for opt in options]
    max_lp = max(option_logprobs)
    exp_logprobs = [math.exp(lp - max_lp) for lp in option_logprobs]
    total = sum(exp_logprobs)
    return {opt: e / total for opt, e in zip(options, exp_logprobs)}


# =========================
# Evaluation
# =========================
def get_question_type(entry_id: str) -> str:
    """Extract question type (e.g. 'Q1') from entry id (e.g. 'Q1_0_404818')."""
    m = re.match(r"(Q\d+)", str(entry_id))
    return m.group(1) if m else ""


def evaluate_results(results_list: list) -> tuple:
    """Compute accuracy and Brier scores. Auto-detects question types."""
    mcq_sources = sorted(set(get_question_type(r.get("id", "")) for r in results_list
                             if get_question_type(r.get("id", ""))))
    overall_items = []
    per_q = {q: [] for q in mcq_sources}

    for r in results_list:
        if r.get("error") is not None:
            continue
        pred_text = (r.get("prediction") or "").strip().upper()
        gt_text = (r.get("ground_truth") or "").strip().upper()
        ds = get_question_type(r.get("id", ""))
        if not pred_text or not gt_text or ds not in mcq_sources:
            continue
        pred_m = re.search(r"\b([A-E])\b", pred_text)
        gt_m = re.search(r"\b([A-E])\b", gt_text)
        if not (pred_m and gt_m):
            continue
        pred_letter, gt_letter = pred_m.group(1), gt_m.group(1)
        correct = pred_letter == gt_letter
        options = r.get("mcq_options", ["A", "B", "C", "D", "E"])
        probs = r.get("option_probabilities", {})
        item = {"correct": correct, "gt": gt_letter, "probs": probs, "options": options}
        overall_items.append(item)
        per_q[ds].append(item)

    def compute_metrics(items):
        if not items:
            return None
        total = len(items)
        correct = sum(1 for x in items if x["correct"])
        brier_scores, baselines = [], []
        for x in items:
            try:
                brier_scores.append(brier_score_multiclass(x["probs"], x["gt"], x["options"]))
                baselines.append(brier_random_baseline(x["options"]))
            except Exception:
                continue
        return {
            "count": total, "correct": correct,
            "accuracy": correct / total,
            "brier_score": float(np.mean(brier_scores)) if brier_scores else None,
            "random_baseline_brier": float(np.mean(baselines)) if baselines else None,
        }

    overall_metrics = {}
    if overall_items:
        overall_metrics["overall"] = compute_metrics(overall_items)
    per_question_metrics = {q: compute_metrics(per_q[q]) for q in mcq_sources}
    per_question_metrics = {k: v for k, v in per_question_metrics.items() if v}
    return overall_metrics, per_question_metrics


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="GPT Forecasting Inference (Responses API)")
    parser.add_argument("--test_json", required=True, help="Path to test JSON file")
    parser.add_argument("--output", required=True, help="Path to output results JSON")
    parser.add_argument("--model", default="gpt-5-2025-08-07", help="GPT model name")
    parser.add_argument("--frame_fps", type=float, default=0.5,
                        help="Frame sampling rate for video (FPS)")
    parser.add_argument("--image_detail", default="low", choices=["low", "high", "auto"],
                        help="Image detail level for API")
    parser.add_argument("--request_delay", type=float, default=0.5,
                        help="Delay between API requests (seconds)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries per request")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")
    client = OpenAI(api_key=api_key)

    # Load existing results for resumption
    existing_results = []
    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing_results = json.load(f)
        processed_ids = {str(r.get("id", "")) for r in existing_results}
        print(f"Found {len(existing_results)} existing results, will skip those entries.")

    # Load test data
    print(f"Loading test data from {args.test_json}")
    with open(args.test_json, "r") as f:
        test_data = json.load(f)
    print(f"Total entries: {len(test_data)}")

    # Filter out already processed
    entries_to_process = [e for e in test_data if str(e.get("id", "")) not in processed_ids]
    print(f"Entries to process: {len(entries_to_process)}")

    if not entries_to_process:
        print("No new entries to process.")
        return

    new_results = []
    for idx, entry in enumerate(tqdm(entries_to_process, desc="Processing")):
        entry_id = entry.get("id", f"entry_{idx}")
        video_path = entry.get("video")

        question, ground_truth = None, None
        for conv in entry.get("conversations", []):
            if conv.get("from") == "human":
                question = (conv.get("value", "") or "").replace("<video>", "").strip()
            elif conv.get("from") == "gpt":
                ground_truth = conv.get("value", "")

        if not question or not video_path:
            continue

        if not os.path.exists(video_path):
            new_results.append({"id": entry_id,
                                "video_path": video_path, "question": question,
                                "ground_truth": ground_truth, "prediction": None,
                                "error": "Video file not found"})
            continue

        try:
            mcq_options = infer_mcq_options_from_text(question)
        except Exception:
            mcq_options = ["A", "B", "C", "D", "E"]

        # Extract frames from video
        try:
            b64_frames = extract_frames_as_base64(video_path, sample_fps=args.frame_fps)
        except Exception as e:
            new_results.append({"id": entry_id,
                                "video_path": video_path, "question": question,
                                "ground_truth": ground_truth, "prediction": None,
                                "error": f"Frame extraction error: {e}"})
            continue

        # Build API request content
        content = []
        for b64 in b64_frames:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
                "detail": args.image_detail,
            })
        prompt_text = (f"Pick the best answer. Reply with exactly one letter: "
                       f"A, B, C, D, or E.\n\n{question}")
        content.append({"type": "input_text", "text": prompt_text})

        # Make API call with retries
        response = None
        last_error = None
        for attempt in range(args.max_retries):
            try:
                response = client.responses.create(
                    model=args.model,
                    instructions="You are an expert at analyzing sports game footage and answering multiple choice questions about game events and statistics.",
                    input=[{"role": "user", "content": content}],
                    max_output_tokens=16,
                    top_logprobs=20,
                    temperature=0,
                    include=["message.output_text.logprobs"],
                    reasoning={"effort": "none"},
                )
                break
            except Exception as e:
                last_error = e
                if attempt < args.max_retries - 1:
                    delay = 2.0 * (2 ** attempt)
                    print(f"\n  Retry {attempt + 1} for {entry_id} after {delay}s: {e}")
                    time.sleep(delay)

        if response is None:
            new_results.append({"id": entry_id,
                                "video_path": video_path, "question": question,
                                "ground_truth": ground_truth, "prediction": None,
                                "mcq_options": mcq_options, "option_logprobs": None,
                                "option_probabilities": None,
                                "error": str(last_error)})
            time.sleep(args.request_delay)
            continue

        pred_text = get_prediction_text(response)
        logprobs = extract_logprobs_from_response(response, tuple(mcq_options))
        option_probs = compute_probs_from_logprobs(logprobs, mcq_options)

        new_results.append({
            "id": entry_id,
            "video_path": video_path, "question": question,
            "ground_truth": ground_truth, "prediction": pred_text,
            "mcq_options": mcq_options, "option_logprobs": logprobs,
            "option_probabilities": option_probs, "error": None,
        })
        time.sleep(args.request_delay)

        # Periodic save
        if (idx + 1) % 50 == 0:
            all_results = existing_results + new_results
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n  Saved intermediate results ({idx + 1}/{len(entries_to_process)})")

    # Final save
    all_results = existing_results + new_results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} total results to {args.output}")
    print(f"  New: {len(new_results)}, Existing: {len(existing_results)}")

    # Evaluate
    overall_metrics, per_question_metrics = evaluate_results(all_results)
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    for q in sorted(per_question_metrics.keys()):
        m = per_question_metrics[q]
        brier_str = f", Brier={m['brier_score']:.4f}" if m.get("brier_score") else ""
        print(f"  {q}: Count={m['count']}, Acc={m['accuracy']:.4f}{brier_str}")
    if "overall" in overall_metrics:
        om = overall_metrics["overall"]
        brier_str = f", Brier={om['brier_score']:.4f}" if om.get("brier_score") else ""
        print(f"\n  Overall: Count={om['count']}, Acc={om['accuracy']:.4f}{brier_str}")

    metrics_path = args.output.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"overall_metrics": overall_metrics,
                   "per_question_metrics": per_question_metrics}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
