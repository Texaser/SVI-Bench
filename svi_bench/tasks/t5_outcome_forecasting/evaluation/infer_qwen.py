#!/usr/bin/env python3
"""
Qwen3-VL Forecasting Inference Script

Runs Qwen3-VL (with optional LoRA adapter) on sports forecasting MCQ data.
Supports multi-GPU distributed inference via torchrun.

Usage:
  # Single GPU
  python infer_qwen.py --test_json data/basketball_test.json --output results.json

  # With LoRA adapter
  python infer_qwen.py --test_json data/hockey_test.json --output results.json \
      --adapter /path/to/lora/checkpoint

  # Multi-GPU (4 GPUs)
  torchrun --nproc_per_node=4 infer_qwen.py \
      --test_json data/soccer_test.json --output results.json
"""

import argparse
import json
import os
import re
import time
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from decord import VideoReader, cpu
from PIL import Image


# =========================
# Distributed setup
# =========================
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed run requested but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            timeout=torch.distributed.default_pg_timeout,
        )
        return rank, world_size, local_rank
    return None, 1, 0


# =========================
# Video loading with decord
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


def load_video_frames(video_path: str, sample_fps: float = 1.0):
    """Load and sample frames from video at a specific FPS rate using decord."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    total = len(vr)
    video_fps = vr.get_avg_fps()
    idx = fps_sample_indices(total, video_fps, sample_fps)
    batch = vr.get_batch(idx).asnumpy()
    frames = [Image.fromarray(frame) for frame in batch]
    return frames, total, video_fps


# =========================
# Prompt + input construction
# =========================
def build_prompt(processor, messages):
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_inputs(processor, video_path, question, device, sample_fps: float = 1.0):
    """Load video frames with decord and create model inputs."""
    frames, total_frames, video_fps = load_video_frames(video_path, sample_fps=sample_fps)
    question = question + "\nPlease answer with only the letter (A, B, C, D, or E) corresponding to your choice, without any additional text or explanation."
    messages = [
        {"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": question},
        ]}
    ]
    prompt = build_prompt(processor, messages)
    inputs = processor(
        text=prompt,
        videos=[frames],
        video_metadata=[{"total_num_frames": len(frames), "fps": sample_fps,
                         "duration": float(total_frames / video_fps)}],
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs.to(device)


# =========================
# MCQ helpers
# =========================
def infer_mcq_options_from_text(question_text: str):
    """Extract MCQ options (A-E) from question text."""
    text = question_text.upper()
    found = set(m.group(1) for m in re.finditer(r"\b([A-E])\s*:", text))
    if not found:
        raise ValueError("Could not infer MCQ options from text.")
    order = ["A", "B", "C", "D", "E"]
    max_opt = max(found)
    return order[: order.index(max_opt) + 1]


def get_option_logits_and_probs(processor, outputs, options):
    """Extract logits and softmax probabilities for MCQ option tokens."""
    if not (hasattr(outputs, "scores") and outputs.scores and len(outputs.scores) > 0):
        raise ValueError("No generation scores available.")
    tokenizer = processor.tokenizer
    opt_token_ids = {}
    for opt in options:
        ids = tokenizer.encode(" " + opt, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Tokenizer produced no ids for option '{opt}'.")
        opt_token_ids[opt] = ids[-1]
    step0_logits = outputs.scores[0][0]
    opt_logits = {opt: float(step0_logits[opt_token_ids[opt]].detach().cpu()) for opt in options}
    logits_tensor = torch.tensor([opt_logits[o] for o in options], dtype=torch.float32)
    probs = F.softmax(logits_tensor, dim=0)
    opt_probs = {o: float(p.item()) for o, p in zip(options, probs)}
    return opt_logits, opt_probs


# =========================
# Brier score
# =========================
def brier_score_multiclass(option_probs, gt_letter, options):
    K = len(options)
    s = sum((float(option_probs.get(opt, 0.0)) - (1.0 if opt == gt_letter else 0.0)) ** 2
            for opt in options)
    return s / K


def brier_random_baseline(options):
    K = len(options)
    return (K - 1) / (K * K)


# =========================
# Evaluation
# =========================
def get_question_type(entry_id: str) -> str:
    """Extract question type (e.g. 'Q1') from entry id (e.g. 'Q1_0_404818')."""
    m = re.match(r"(Q\d+)", str(entry_id))
    return m.group(1) if m else ""


def evaluate_results(results_list):
    """Compute accuracy and Brier scores. Auto-detects question types from data."""
    mcq_sources = sorted(set(get_question_type(r.get("id", "")) for r in results_list
                             if get_question_type(r.get("id", ""))))
    overall_items = []
    per_q = {q: [] for q in mcq_sources}

    for r in results_list:
        if r.get("error") is not None or r.get("prediction") is None:
            continue
        ds = get_question_type(r.get("id", ""))
        if ds not in mcq_sources:
            continue
        pred_m = re.search(r"\b([A-E])\b", (r.get("prediction") or "").strip().upper())
        gt_m = re.search(r"\b([A-E])\b", (r.get("ground_truth") or "").strip().upper())
        if not (pred_m and gt_m):
            continue
        pred_letter, gt_letter = pred_m.group(1), gt_m.group(1)
        correct = pred_letter == gt_letter
        options = r.get("mcq_options")
        probs = r.get("option_probabilities")
        if not options or probs is None:
            continue
        item = {"correct": correct, "gt": gt_letter, "probs": probs, "options": options}
        overall_items.append(item)
        per_q[ds].append(item)

    def compute_metrics(items):
        if not items:
            return None
        total = len(items)
        correct = sum(1 for x in items if x["correct"])
        brier_scores = [brier_score_multiclass(x["probs"], x["gt"], x["options"]) for x in items]
        baselines = [brier_random_baseline(x["options"]) for x in items]
        l2_scores = [1 - (1 - x["probs"].get(x["gt"], 0.0)) ** 2 for x in items]
        return {
            "count": total, "correct": correct,
            "accuracy": correct / total,
            "brier_score": float(np.mean(brier_scores)),
            "l2_brier_score": float(np.mean(l2_scores)),
            "random_baseline_brier": float(np.mean(baselines)),
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
    parser = argparse.ArgumentParser(description="Qwen3-VL Forecasting Inference")
    parser.add_argument("--test_json", required=True, help="Path to test JSON file")
    parser.add_argument("--output", required=True, help="Path to output results JSON")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="Base model name/path")
    parser.add_argument("--adapter", default="", help="Path to LoRA adapter checkpoint (optional)")
    parser.add_argument("--sample_fps", type=float, default=0.2, help="Frame sampling rate (FPS)")
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
            print(f"Loading LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    elif args.adapter:
        if rank is None or rank == 0:
            print(f"Warning: Adapter path not found: {args.adapter}. Using base model.")
    model.eval()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torchcodec.*")
        warnings.filterwarnings("ignore", message=".*torchvision.*video.*")
        processor = AutoProcessor.from_pretrained(args.model)

    # Load data
    if rank is None or rank == 0:
        print(f"Loading test data from {args.test_json}")
    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    # Split across ranks
    if rank is not None:
        splits = np.array_split(np.arange(len(test_data)), world_size)
        test_data = [test_data[i] for i in splits[rank].tolist()]
        if rank == 0:
            print(f"Processing {len(test_data)} entries on rank 0 (world_size={world_size})")

    # Inference loop
    results = []
    progress = tqdm(test_data, desc=f"Inference (GPU {rank})",
                    disable=(rank is not None and rank != 0))

    for idx, entry in enumerate(progress):
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
            print(f"[SKIP] Video not found: {video_path}")
            continue

        mcq_options = infer_mcq_options_from_text(question)

        try:
            inputs = make_inputs(processor, video_path, question, device, sample_fps=args.sample_fps)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, max_new_tokens=128,
                    return_dict_in_generate=True, output_scores=True, do_sample=False,
                )
            in_len = inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(
                outputs.sequences[:, in_len:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            prediction = decoded[0] if decoded else ""
            option_logits, option_probs = get_option_logits_and_probs(processor, outputs, mcq_options)

            results.append({
                "id": entry_id, "video_path": video_path,
                "question": question, "ground_truth": ground_truth,
                "prediction": prediction, "mcq_options": mcq_options,
                "option_probabilities": option_probs,
                "option_logits": option_logits, "error": None,
            })
        except Exception as e:
            if rank is None or rank == 0:
                print(f"\n[SKIP] {entry_id}: {type(e).__name__}: {e}")
            continue

    # Save / merge results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if rank is not None:
        per_gpu_output = args.output.replace(".json", f"_rank_{rank}.json")
        with open(per_gpu_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Rank {rank}] Saved {len(results)} results to {per_gpu_output}")

        completion_marker = per_gpu_output.replace(".json", "_completed.txt")
        with open(completion_marker, "w") as f:
            f.write(f"Rank {rank} completed at {time.time()}\n")
            f.flush()
            os.fsync(f.fileno())

        if rank != 0:
            if dist.is_initialized():
                dist.destroy_process_group()
            raise SystemExit(0)

        # Rank 0: wait and merge
        print("[Rank 0] Waiting for all ranks...")
        start_wait = time.time()
        completed = set()
        while len(completed) < world_size:
            if time.time() - start_wait > 7200:
                raise TimeoutError("Timeout waiting for all ranks.")
            for r in range(world_size):
                if r not in completed:
                    marker = args.output.replace(".json", f"_rank_{r}_completed.txt")
                    if os.path.exists(marker):
                        completed.add(r)
            if len(completed) < world_size:
                time.sleep(5)

        all_results = []
        for r in range(world_size):
            per_file = args.output.replace(".json", f"_rank_{r}.json")
            with open(per_file, "r") as f:
                all_results.extend(json.load(f))
            os.remove(per_file)
            marker = args.output.replace(".json", f"_rank_{r}_completed.txt")
            if os.path.exists(marker):
                os.remove(marker)

        all_results.sort(key=lambda x: x.get("id", ""))
        results = all_results
        if dist.is_initialized():
            dist.destroy_process_group()

    # Save final results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {args.output}")

    # Evaluate
    overall_metrics, per_question_metrics = evaluate_results(results)
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    for q in sorted(per_question_metrics.keys()):
        m = per_question_metrics[q]
        print(f"  {q}: Count={m['count']}, Acc={m['accuracy']:.4f}, "
              f"Brier={m['brier_score']:.4f}, L2Brier={m['l2_brier_score']:.4f}")
    if "overall" in overall_metrics:
        om = overall_metrics["overall"]
        print(f"\n  Overall: Count={om['count']}, Acc={om['accuracy']:.4f}, "
              f"Brier={om['brier_score']:.4f}, L2Brier={om['l2_brier_score']:.4f}")

    metrics_path = args.output.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"overall_metrics": overall_metrics,
                   "per_question_metrics": per_question_metrics}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
