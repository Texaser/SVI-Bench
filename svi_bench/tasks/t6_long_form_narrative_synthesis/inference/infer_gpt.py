#!/usr/bin/env python3
"""
GPT Inference for Report Generation (OpenAI Responses API)

Generates game reports using GPT models with pre-extracted video frames.
Handles both single-game and multi-game entries across all sports.

Usage:
  export OPENAI_API_KEY="sk-..."

  python infer_gpt.py \
      --test_list data/basketball/test_list.json \
      --data_dir data/basketball \
      --frames_dir /path/to/full_game_video_frames \
      --output results/basketball_gpt.json
"""

import argparse
import base64
import json
import os
import re
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI


# ============================================================================
# Helpers
# ============================================================================
def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def resolve_placeholders(text, metadata):
    """Replace template placeholders with values from metadata.json."""
    for placeholder, key in [("{player}", "selected_player"), ("{title}", "title"),
                              ("{attribute}", "attribute"), ("{timeframe}", "timeframe"),
                              ("{event_description}", "event_description")]:
        value = metadata.get(key)
        if value is not None:
            text = text.replace(placeholder, str(value))
    return text


def qtype_to_dirs(q_type):
    """Map q_type to (game_type_dir, q_num). e.g., single_Q1 -> (single_game, Q1)."""
    if q_type.startswith("multi_"):
        return "multi_game", q_type.replace("multi_", "")
    return "single_game", q_type.replace("single_", "")


def get_sample_dir(data_dir, q_type, sample_id):
    game_dir, q_num = qtype_to_dirs(q_type)
    return os.path.join(data_dir, game_dir, q_num, str(sample_id))


def get_prompt(sample_dir):
    """Read prompt.txt and resolve placeholders."""
    prompt = read_file(os.path.join(sample_dir, "prompt.txt"))
    if not prompt:
        return None
    meta_path = os.path.join(sample_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            prompt = resolve_placeholders(prompt, json.load(f))
    return prompt


def get_game_ids(sample_dir, q_type):
    """Get game ID(s) from metadata.json."""
    meta_path = os.path.join(sample_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path) as f:
        meta = json.load(f)
    if q_type.startswith("multi_"):
        return meta.get("game_id_list", [])
    gid = meta.get("game_id")
    return [str(gid)] if gid else []


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_frame_paths(frames_dir, game_id, max_frames):
    """Get sorted frame paths for a single game, up to max_frames."""
    game_dir = os.path.join(frames_dir, str(game_id))
    if not os.path.isdir(game_dir):
        return []
    frames = []
    for f in os.listdir(game_dir):
        m = re.match(r"frame_(\d+)\.jpg$", f, re.IGNORECASE)
        if m:
            frames.append((int(m.group(1)), os.path.join(game_dir, f)))
    frames.sort()
    return [p for _, p in frames[:max_frames]]


def get_frame_paths_multi(frames_dir, game_ids, max_frames):
    """Uniformly distribute max_frames across multiple games."""
    if not game_ids:
        return []
    per_game = max_frames // len(game_ids)
    remainder = max_frames % len(game_ids)
    all_paths = []
    for i, gid in enumerate(game_ids):
        n = per_game + (1 if i < remainder else 0)
        game_dir = os.path.join(frames_dir, str(gid))
        if not os.path.isdir(game_dir):
            continue
        frames = []
        for f in os.listdir(game_dir):
            m = re.match(r"frame_(\d+)\.jpg$", f, re.IGNORECASE)
            if m:
                frames.append((int(m.group(1)), os.path.join(game_dir, f)))
        frames.sort()
        if len(frames) <= n:
            all_paths.extend(p for _, p in frames)
        else:
            indices = np.linspace(0, len(frames) - 1, n, dtype=int)
            all_paths.extend(frames[idx][1] for idx in indices)
    return all_paths


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="GPT Report Generation Inference")
    parser.add_argument("--test_list", required=True, help="Path to test_list.json")
    parser.add_argument("--data_dir", required=True, help="Path to sport data dir")
    parser.add_argument("--output", required=True, help="Output results JSON")
    parser.add_argument("--frames_dir", required=True, help="Base dir for pre-extracted frames")
    parser.add_argument("--model", default="gpt-4o", help="GPT model name")
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--image_detail", default="low", choices=["low", "high", "auto"])
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--request_delay", type=float, default=1.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")
    client = OpenAI(api_key=api_key)

    with open(args.test_list) as f:
        test_list = json.load(f)

    # Load existing results for resumption
    aggregated = {}
    results_list = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        aggregated = existing.get("predictions", {})
        results_list = existing.get("results", [])
        print(f"Loaded {len(results_list)} existing results")

    processed = 0
    for q_type in sorted(test_list.keys()):
        agg_key = q_type.replace("single_", "") if q_type.startswith("single_") else q_type
        sample_ids = test_list[q_type]
        is_multi = q_type.startswith("multi_")

        if agg_key not in aggregated:
            aggregated[agg_key] = {}

        for sid in tqdm(sample_ids, desc=q_type):
            sid_str = str(sid)
            if sid_str in aggregated.get(agg_key, {}):
                continue

            sample_dir = get_sample_dir(args.data_dir, q_type, sid)
            prompt = get_prompt(sample_dir)
            if not prompt:
                continue

            game_ids = get_game_ids(sample_dir, q_type)
            if not game_ids:
                continue

            # Get frames
            if is_multi:
                frame_paths = get_frame_paths_multi(args.frames_dir, game_ids, args.max_frames)
            else:
                frame_paths = get_frame_paths(args.frames_dir, game_ids[0], args.max_frames)

            if not frame_paths:
                results_list.append({"id": f"{q_type}_{sid}", "q_type": q_type,
                                     "sample_id": sid, "prediction": None,
                                     "error": "No frames found"})
                continue

            # Build API content
            content = []
            for fp in frame_paths:
                b64 = encode_image_b64(fp)
                content.append({"type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64}",
                                "detail": args.image_detail})
            content.append({"type": "input_text", "text": prompt})

            # API call with retries
            response = None
            last_error = None
            for attempt in range(args.max_retries):
                try:
                    response = client.responses.create(
                        model=args.model,
                        instructions="You are a professional sports journalist who writes analytical game reports.",
                        input=[{"role": "user", "content": content}],
                        max_output_tokens=4096,
                        temperature=0.7,
                    )
                    break
                except Exception as e:
                    last_error = e
                    delay = 2.0 * (2 ** attempt)
                    print(f"\n  Retry {attempt+1} for {q_type}/{sid} after {delay}s: {e}")
                    time.sleep(delay)

            if response is None:
                results_list.append({"id": f"{q_type}_{sid}", "q_type": q_type,
                                     "sample_id": sid, "prediction": None,
                                     "error": str(last_error)})
                time.sleep(args.request_delay)
                continue

            # Extract prediction text
            pred_text = ""
            try:
                body = response.model_dump() if hasattr(response, "model_dump") else response
                for item in body.get("output", []):
                    if item.get("type") == "message" and item.get("role") == "assistant":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                pred_text = c.get("text", "")
            except Exception:
                pred_text = str(response)

            aggregated.setdefault(agg_key, {})[sid_str] = pred_text
            results_list.append({"id": f"{q_type}_{sid}", "q_type": q_type,
                                 "sample_id": sid, "prediction": pred_text, "error": None})
            processed += 1
            time.sleep(args.request_delay)

            # Periodic save
            if processed % 10 == 0:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump({"predictions": aggregated, "results": results_list}, f, indent=2)

    # Final save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"predictions": aggregated, "results": results_list}, f, indent=2)
    print(f"\nDone. {processed} new entries. Total: {len(results_list)}. Saved to {args.output}")


if __name__ == "__main__":
    main()
