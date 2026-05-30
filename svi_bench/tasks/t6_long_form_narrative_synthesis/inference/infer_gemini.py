#!/usr/bin/env python3
"""
Gemini Report Generation Inference Script (Video Upload)

Generates game reports using Gemini by compressing and uploading full game videos.
Handles both single-game and multi-game entries across all sports.

Usage:
  export GEMINI_API_KEY="AIza..."

  python infer_gemini.py \
      --test_list data/basketball/test_list.json \
      --data_dir data/basketball \
      --video_dir /path/to/full_game_videos \
      --video_cache_dir /path/to/compressed_cache \
      --output results/basketball_gemini.json
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError("Install google-genai: pip install google-genai") from e


# ============================================================================
# Helpers
# ============================================================================
def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def resolve_placeholders(text, metadata):
    for placeholder, key in [("{player}", "selected_player"), ("{title}", "title"),
                              ("{attribute}", "attribute"), ("{timeframe}", "timeframe"),
                              ("{event_description}", "event_description")]:
        value = metadata.get(key)
        if value is not None:
            text = text.replace(placeholder, str(value))
    return text


def get_sample_dir(data_dir, q_type, sample_id):
    if q_type.startswith("multi_"):
        q_num = q_type.replace("multi_", "")
        return os.path.join(data_dir, "multi_game", q_num, str(sample_id))
    q_num = q_type.replace("single_", "")
    return os.path.join(data_dir, "single_game", q_num, str(sample_id))


def get_prompt(sample_dir):
    prompt = read_file(os.path.join(sample_dir, "prompt.txt"))
    if not prompt:
        return None
    meta_path = os.path.join(sample_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            prompt = resolve_placeholders(prompt, json.load(f))
    return prompt


def get_video_paths_from_meta(sample_dir, q_type):
    """Get relative video paths from metadata.json."""
    meta_path = os.path.join(sample_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path) as f:
        meta = json.load(f)
    if q_type.startswith("multi_"):
        return meta.get("video_paths", [])
    vp = meta.get("video_path", "")
    return [vp] if vp else []


# ============================================================================
# Video compression
# ============================================================================
def get_video_duration(path):
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return None


def compress_video(src, dst, target_duration_sec):
    """Compress video to fit within target duration using ffmpeg."""
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src_duration = get_video_duration(src)
    if not src_duration:
        return None

    if src_duration <= target_duration_sec:
        speed = 1.0
    else:
        speed = src_duration / target_duration_sec

    pts_factor = 1.0 / speed
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"setpts={pts_factor}*PTS,scale=-2:min'(ih,400)'",
        "-r", "30", "-c:v", "libx264", "-crf", "32", "-preset", "fast",
        "-an", "-threads", "4", str(dst),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=600, check=True)
        return dst
    except Exception as e:
        print(f"  ffmpeg error: {e}")
        return None


def concat_videos(video_paths, output_path):
    """Concatenate multiple videos using ffmpeg concat demuxer."""
    if os.path.exists(output_path):
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for vp in video_paths:
            f.write(f"file '{vp}'\n")
        list_path = f.name
    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
               "-c", "copy", str(output_path)]
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)
        return output_path
    except Exception as e:
        print(f"  concat error: {e}")
        return None
    finally:
        os.unlink(list_path)


def prepare_video(video_paths, cache_dir, total_duration):
    """Compress and optionally concatenate game videos."""
    if len(video_paths) == 1:
        src = video_paths[0]
        if not os.path.exists(src):
            return None
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(cache_dir, f"{name}_compressed.mp4")
        return compress_video(src, dst, total_duration)

    # Multi-game: split duration evenly
    per_game = total_duration / len(video_paths)
    compressed = []
    for src in video_paths:
        if not os.path.exists(src):
            continue
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(cache_dir, f"{name}_compressed_{int(per_game)}s.mp4")
        result = compress_video(src, dst, per_game)
        if result:
            compressed.append(result)

    if not compressed:
        return None
    if len(compressed) == 1:
        return compressed[0]

    concat_name = "_".join(game_ids) + "_concat.mp4"
    concat_path = os.path.join(cache_dir, concat_name)
    return concat_videos(compressed, concat_path)


# ============================================================================
# Gemini API helpers
# ============================================================================
def _state_str(state):
    if state is None:
        return ""
    return str(state.name).upper() if hasattr(state, "name") else str(state).upper()


def wait_until_active(client, file_obj, timeout=300):
    start = time.time()
    name = getattr(file_obj, "name", None)
    while True:
        refreshed = client.files.get(name=name)
        state = _state_str(getattr(refreshed, "state", None))
        if state == "ACTIVE":
            return refreshed
        if state == "FAILED":
            raise RuntimeError(f"File processing failed: {getattr(refreshed, 'error', None)}")
        if time.time() - start > timeout:
            raise TimeoutError("File upload timed out")
        time.sleep(5)


def query_gemini(client, video_path, prompt, model_id, max_retries=3):
    """Upload video to Gemini and generate report."""
    if not os.path.isfile(video_path):
        return None, f"Video not found: {video_path}"

    for attempt in range(max_retries):
        uploaded = None
        try:
            uploaded = client.files.upload(file=video_path)
            uploaded = wait_until_active(client, uploaded)

            file_uri = getattr(uploaded, "uri", None)
            mime = getattr(uploaded, "mime_type", None) or "video/mp4"

            content = types.Content(parts=[
                types.Part(file_data=types.FileData(file_uri=file_uri, mime_type=mime)),
                types.Part(text=prompt),
            ])

            response = client.models.generate_content(
                model=model_id, contents=content,
                config=types.GenerateContentConfig(
                    temperature=0.7, max_output_tokens=8192,
                    media_resolution="MEDIA_RESOLUTION_LOW",
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False, thinking_level="LOW"),
                ),
            )
            text = getattr(response, "text", None) or ""
            return text.strip(), None

        except Exception as e:
            err = str(e)
            if any(s in err for s in ["429", "503", "RESOURCE_EXHAUSTED"]):
                if attempt < max_retries - 1:
                    time.sleep(2 * (2 ** attempt))
                    continue
            return None, err

        finally:
            if uploaded:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass

    return None, "Max retries exceeded"


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Gemini Report Generation Inference")
    parser.add_argument("--test_list", required=True, help="Path to test_list.json")
    parser.add_argument("--data_dir", required=True, help="Path to sport data dir")
    parser.add_argument("--video_dir", required=True, help="Base dir for full game videos")
    parser.add_argument("--video_cache_dir", required=True, help="Dir for compressed video cache")
    parser.add_argument("--output", required=True, help="Output results JSON")
    parser.add_argument("--model", default="gemini-2.5-flash-preview")
    parser.add_argument("--total_duration", type=int, default=3600,
                        help="Max total video duration in seconds (default 3600)")
    parser.add_argument("--request_delay", type=float, default=2.0)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY env var required.")
    client = genai.Client(api_key=api_key)

    with open(args.test_list) as f:
        test_list = json.load(f)

    # Load existing for resumption
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
        if agg_key not in aggregated:
            aggregated[agg_key] = {}

        for sid in tqdm(test_list[q_type], desc=q_type):
            sid_str = str(sid)
            if sid_str in aggregated.get(agg_key, {}):
                continue

            sample_dir = get_sample_dir(args.data_dir, q_type, sid)
            prompt = get_prompt(sample_dir)
            if not prompt:
                continue

            rel_paths = get_video_paths_from_meta(sample_dir, q_type)
            if not rel_paths:
                continue
            full_paths = [os.path.join(args.video_dir, rp) for rp in rel_paths]

            video_path = prepare_video(full_paths, args.video_cache_dir,
                                       args.total_duration)
            if not video_path:
                results_list.append({"q_type": q_type, "sample_id": sid,
                                     "prediction": None, "error": "Video preparation failed"})
                continue

            pred, error = query_gemini(client, video_path, prompt, args.model, args.max_retries)

            if pred:
                aggregated.setdefault(agg_key, {})[sid_str] = pred
            results_list.append({"q_type": q_type, "sample_id": sid,
                                 "prediction": pred, "error": error})
            processed += 1
            time.sleep(args.request_delay)

            # Periodic save
            if processed % 10 == 0:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump({"predictions": aggregated, "results": results_list}, f, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"predictions": aggregated, "results": results_list}, f, indent=2)
    print(f"\nDone. {processed} new entries. Total: {len(results_list)}. Saved to {args.output}")


if __name__ == "__main__":
    main()
