#!/usr/bin/env python3
"""
Gemini Forecasting Inference Script (Video Upload)

Runs Gemini models on sports forecasting MCQ data by uploading videos
directly via the Gemini Files API.

Requires: google-genai, tqdm

Usage:
  # Set your API key
  export GEMINI_API_KEY="AIza..."

  # Run inference
  python infer_gemini.py --test_json data/basketball_test.json --output results.json

  # With custom settings
  python infer_gemini.py --test_json data/soccer_test.json --output results.json \
      --model gemini-2.5-flash-preview --request_delay 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import mimetypes
import random
from pathlib import Path
from typing import Any
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except Exception as e:
    raise ImportError(
        "google-genai package is required. Install with: pip install google-genai"
    ) from e


# =========================
# Utility functions
# =========================
def ensure_client(api_key: str) -> genai.Client:
    """Create Gemini client."""
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def _state_to_upper_str(state: Any) -> str:
    """Normalize SDK state values."""
    if state is None:
        return ""
    if hasattr(state, "name"):
        return str(state.name).upper()
    return str(state).upper()


def _wait_until_active(client: genai.Client, file_obj: Any,
                       timeout_s: int = 300, interval_s: int = 5) -> Any:
    """Poll Files API until the uploaded video becomes ACTIVE."""
    start_ts = time.time()
    name = getattr(file_obj, "name", None)
    if not name:
        raise ValueError("Uploaded file object has no 'name' field.")
    last_state = None
    while True:
        refreshed = client.files.get(name=name)
        state = getattr(refreshed, "state", None)
        if state != last_state:
            print(f"  File state: {state}")
            last_state = state
        state_str = _state_to_upper_str(state)
        if state_str == "ACTIVE":
            return refreshed
        if state_str == "FAILED":
            raise RuntimeError(f"File processing FAILED: {getattr(refreshed, 'error', None)}")
        if time.time() - start_ts > timeout_s:
            raise TimeoutError("Waiting for file to become ACTIVE timed out.")
        time.sleep(interval_s)


def _guess_mime_type(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "video/mp4"


# =========================
# Gemini video query
# =========================
def query_gemini_with_video(client: genai.Client, video_path: str, question: str,
                            model_id: str, max_retries: int = 3,
                            delete_uploaded: bool = True) -> dict:
    """Query Gemini with video via Files API, with retry logic."""
    if not os.path.isfile(video_path):
        return {"prediction": None, "error": f"Video not found: {video_path}"}
    if os.path.getsize(video_path) < 1024:
        return {"prediction": None, "error": "File too small or corrupt"}

    for attempt in range(max_retries):
        uploaded = None
        try:
            uploaded = client.files.upload(file=video_path)
            uploaded = _wait_until_active(client, uploaded)

            file_uri = getattr(uploaded, "uri", None)
            uploaded_mime = getattr(uploaded, "mime_type", None) or _guess_mime_type(video_path)

            prompt_text = (f"Pick the best answer. Reply with the option letter only "
                           f"(A, B, C, D, or E). Do not include any explanation.\n\n{question}")

            content = types.Content(parts=[
                types.Part(file_data=types.FileData(file_uri=file_uri, mime_type=uploaded_mime)),
                types.Part(text=prompt_text),
            ])

            response = client.models.generate_content(
                model=model_id, contents=content,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    media_resolution="MEDIA_RESOLUTION_LOW",
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False, thinking_level="LOW",
                    ),
                ),
            )
            prediction = getattr(response, "text", None) or ""
            return {"prediction": prediction.strip(), "error": None}

        except Exception as e:
            error_str = str(e)
            if any(s in error_str for s in ["429", "503", "RESOURCE_EXHAUSTED"]):
                if attempt < max_retries - 1:
                    retry_delay = 2 * (2 ** attempt)
                    print(f"  Rate limit hit, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
            return {"prediction": None, "error": error_str}

        finally:
            if delete_uploaded and uploaded is not None:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass

    return {"prediction": None, "error": "Max retries exceeded"}


# =========================
# Evaluation
# =========================
def get_question_type(entry_id: str) -> str:
    """Extract question type (e.g. 'Q1') from entry id (e.g. 'Q1_0_404818')."""
    m = re.match(r"(Q\d+)", str(entry_id))
    return m.group(1) if m else ""


def evaluate_results(results_list: list) -> tuple:
    """Compute accuracy. Auto-detects question types."""
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
        # Take the LAST letter match (handles models that output reasoning before answer)
        pred_matches = re.findall(r"\b([A-E])\b", pred_text)
        gt_matches = re.findall(r"\b([A-E])\b", gt_text)
        if not pred_matches or not gt_matches:
            continue
        correct = pred_matches[-1] == gt_matches[-1]
        overall_items.append({"correct": correct})
        per_q[ds].append({"correct": correct})

    def compute_metrics(items):
        if not items:
            return None
        total = len(items)
        correct = sum(1 for x in items if x["correct"])
        return {"count": total, "correct": correct, "accuracy": correct / total}

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
    parser = argparse.ArgumentParser(description="Gemini Forecasting Inference (Video Upload)")
    parser.add_argument("--test_json", required=True, help="Path to test JSON file")
    parser.add_argument("--output", required=True, help="Path to output results JSON")
    parser.add_argument("--model", default="gemini-2.5-flash-preview",
                        help="Gemini model name")
    parser.add_argument("--request_delay", type=float, default=2.0,
                        help="Delay between requests (seconds)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries per request")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required.")
    client = ensure_client(api_key)

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

    print(f"\nModel: {args.model}")
    print(f"Request delay: {args.request_delay}s")

    new_results = []
    for idx, entry in enumerate(tqdm(entries_to_process, desc="Processing")):
        entry_id = entry.get("id", f"entry_{idx}")

        question, ground_truth = None, None
        for conv in entry.get("conversations", []):
            if conv.get("from") == "human":
                question = (conv.get("value", "") or "").replace("<video>", "").strip()
            elif conv.get("from") == "gpt":
                ground_truth = conv.get("value", "")

        if not question:
            new_results.append({"id": entry_id,
                                "video_path": entry.get("video"), "question": question,
                                "ground_truth": ground_truth, "prediction": None,
                                "error": "Missing question"})
            continue

        video_path = entry.get("video", "")
        if not os.path.exists(video_path):
            new_results.append({"id": entry_id,
                                "video_path": video_path, "question": question,
                                "ground_truth": ground_truth, "prediction": None,
                                "error": f"Video not found: {video_path}"})
            continue

        result = query_gemini_with_video(
            client, video_path, question,
            model_id=args.model, max_retries=args.max_retries,
        )

        new_results.append({
            "id": entry_id,
            "video_path": video_path, "question": question,
            "ground_truth": ground_truth,
            "prediction": result.get("prediction"),
            "error": result.get("error"),
        })

        if idx < len(entries_to_process) - 1:
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

    error_count = sum(1 for r in new_results if r.get("error") is not None)
    if error_count > 0:
        print(f"  Errors: {error_count}")

    # Evaluate
    overall_metrics, per_question_metrics = evaluate_results(all_results)
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    for q in sorted(per_question_metrics.keys()):
        m = per_question_metrics[q]
        print(f"  {q}: Count={m['count']}, Correct={m['correct']}, Acc={m['accuracy']:.4f}")
    if "overall" in overall_metrics:
        om = overall_metrics["overall"]
        print(f"\n  Overall: Count={om['count']}, Correct={om['correct']}, Acc={om['accuracy']:.4f}")

    metrics_path = args.output.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"overall_metrics": overall_metrics,
                   "per_question_metrics": per_question_metrics,
                   "total_entries": len(all_results)}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
