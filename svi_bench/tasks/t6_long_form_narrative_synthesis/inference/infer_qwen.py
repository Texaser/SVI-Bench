#!/usr/bin/env python3
"""
Qwen3-VL Report Generation Inference Script

Generates game reports using Qwen3-VL with optional LoRA adapter.
Handles both single-game and multi-game entries across all sports.
Supports multi-GPU distributed inference via torchrun.

Usage:
  python infer_qwen.py \
      --test_list data/basketball/test_list.json \
      --data_dir data/basketball \
      --video_dir /path/to/full_game_videos \
      --output results/basketball_qwen.json

  # Multi-GPU
  torchrun --nproc_per_node=4 infer_qwen.py \
      --test_list data/hockey/test_list.json \
      --data_dir data/hockey \
      --video_dir /path/to/hockey_videos \
      --output results/hockey_qwen.json --adapter /path/to/lora
"""

import argparse
import json
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from decord import VideoReader, cpu
from PIL import Image


# ============================================================================
# Distributed setup
# ============================================================================
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                timeout=torch.distributed.default_pg_timeout)
        return rank, world_size, local_rank
    return None, 1, 0


# ============================================================================
# Video loading
# ============================================================================
def sample_indices_at_fps(total_frames, video_fps, target_fps):
    """Sample frame indices at target FPS."""
    if total_frames <= 0:
        return np.array([0], dtype=np.int64)
    interval = video_fps / target_fps
    indices = np.arange(0, total_frames, interval, dtype=np.float64)
    indices = np.round(indices).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1)
    return np.unique(indices)


def load_video_frames(video_path, target_fps=1.0):
    """Load frames from a video file at target FPS using decord."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    total = len(vr)
    video_fps = vr.get_avg_fps()
    idx = sample_indices_at_fps(total, video_fps, target_fps)
    batch = vr.get_batch(idx).asnumpy()
    frames = [Image.fromarray(f) for f in batch]
    return frames, total, video_fps


# ============================================================================
# Data helpers
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


def get_video_paths(sample_dir, q_type, video_dir):
    """Get video file paths from metadata.json."""
    meta_path = os.path.join(sample_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path) as f:
        meta = json.load(f)
    if q_type.startswith("multi_"):
        game_ids = meta.get("game_id_list", [])
    else:
        gid = meta.get("game_id")
        game_ids = [str(gid)] if gid else []
    return [os.path.join(video_dir, f"{gid}_full.mp4") for gid in game_ids]


# ============================================================================
# Model input construction
# ============================================================================
def build_prompt(processor, messages):
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_inputs_single(processor, video_path, prompt_text, device, target_fps):
    frames, total, video_fps = load_video_frames(video_path, target_fps)
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": prompt_text}
    ]}]
    text = build_prompt(processor, messages)
    inputs = processor(
        text=text, videos=[frames],
        video_metadata=[{"total_num_frames": len(frames), "fps": target_fps,
                         "duration": float(total / video_fps)}],
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs.to(device)


def make_inputs_multi(processor, video_paths, prompt_text, device, target_fps):
    all_frames = []
    video_metadatas = []
    content = []
    for vp in video_paths:
        if not os.path.exists(vp):
            continue
        frames, total, video_fps = load_video_frames(vp, target_fps)
        all_frames.append(frames)
        video_metadatas.append({"total_num_frames": len(frames), "fps": target_fps,
                                "duration": float(total / video_fps)})
        content.append({"type": "video"})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    text = build_prompt(processor, messages)
    inputs = processor(
        text=text, videos=all_frames, video_metadata=video_metadatas,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs.to(device)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Report Generation Inference")
    parser.add_argument("--test_list", required=True, help="Path to test_list.json")
    parser.add_argument("--data_dir", required=True, help="Path to sport data dir")
    parser.add_argument("--video_dir", required=True, help="Base dir for full game videos")
    parser.add_argument("--output", required=True, help="Output results JSON")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter", default="", help="LoRA adapter path (optional)")
    parser.add_argument("--sample_fps", type=float, default=1.0)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load model
    if rank is None or rank == 0:
        print(f"Loading model: {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto",
        device_map={"": local_rank} if torch.cuda.is_available() else {"": "cpu"},
    )
    if args.adapter and os.path.exists(args.adapter):
        if rank is None or rank == 0:
            print(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torchcodec.*")
        warnings.filterwarnings("ignore", message=".*torchvision.*video.*")
        processor = AutoProcessor.from_pretrained(args.model)

    # Load test list and build work items
    with open(args.test_list) as f:
        test_list = json.load(f)

    work_items = []
    for q_type in sorted(test_list.keys()):
        for sid in test_list[q_type]:
            work_items.append((q_type, sid))

    # Distribute across ranks
    if rank is not None:
        splits = np.array_split(np.arange(len(work_items)), world_size)
        work_items = [work_items[i] for i in splits[rank].tolist()]
        if rank == 0:
            print(f"Processing {len(work_items)} items on rank 0 (world_size={world_size})")

    # Inference
    results = []
    aggregated = {}
    progress = tqdm(work_items, desc=f"Inference (GPU {rank})",
                    disable=(rank is not None and rank != 0))

    for q_type, sid in progress:
        sample_dir = get_sample_dir(args.data_dir, q_type, sid)
        prompt_text = get_prompt(sample_dir)
        if not prompt_text:
            continue

        video_paths = get_video_paths(sample_dir, q_type, args.video_dir)
        is_multi = q_type.startswith("multi_")
        agg_key = q_type.replace("single_", "") if q_type.startswith("single_") else q_type

        try:
            if is_multi and len(video_paths) > 1:
                inputs = make_inputs_multi(processor, video_paths, prompt_text, device, args.sample_fps)
            elif video_paths:
                inputs = make_inputs_single(processor, video_paths[0], prompt_text, device, args.sample_fps)
            else:
                continue

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=True,
                                         temperature=0.7, top_p=0.9)
            in_len = inputs["input_ids"].shape[1]
            prediction = processor.batch_decode(
                outputs[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            aggregated.setdefault(agg_key, {})[str(sid)] = prediction
            results.append({"q_type": q_type, "sample_id": sid, "prediction": prediction, "error": None})

        except Exception as e:
            if rank is None or rank == 0:
                print(f"\n[SKIP] {q_type}/{sid}: {type(e).__name__}: {e}")
            results.append({"q_type": q_type, "sample_id": sid, "prediction": None, "error": str(e)})

    # Save / merge
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if rank is not None:
        per_rank = args.output.replace(".json", f"_rank_{rank}.json")
        with open(per_rank, "w") as f:
            json.dump({"predictions": aggregated, "results": results}, f, indent=2)

        marker = per_rank.replace(".json", "_done.txt")
        with open(marker, "w") as f:
            f.write(f"Rank {rank} done\n")
            f.flush()
            os.fsync(f.fileno())

        if rank != 0:
            if dist.is_initialized():
                dist.destroy_process_group()
            raise SystemExit(0)

        # Rank 0: wait and merge
        start = time.time()
        completed = set()
        while len(completed) < world_size:
            if time.time() - start > 14400:
                raise TimeoutError("Timeout waiting for all ranks.")
            for r in range(world_size):
                m = args.output.replace(".json", f"_rank_{r}_done.txt")
                if r not in completed and os.path.exists(m):
                    completed.add(r)
            if len(completed) < world_size:
                time.sleep(5)

        merged_agg = {}
        merged_results = []
        for r in range(world_size):
            pf = args.output.replace(".json", f"_rank_{r}.json")
            with open(pf) as f:
                rd = json.load(f)
            for k, v in rd.get("predictions", {}).items():
                merged_agg.setdefault(k, {}).update(v)
            merged_results.extend(rd.get("results", []))
            os.remove(pf)
            os.remove(pf.replace(".json", "_done.txt"))

        aggregated = merged_agg
        results = merged_results
        if dist.is_initialized():
            dist.destroy_process_group()

    with open(args.output, "w") as f:
        json.dump({"predictions": aggregated, "results": results}, f, indent=2)
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
