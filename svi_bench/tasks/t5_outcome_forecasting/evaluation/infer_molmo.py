#!/usr/bin/env python3
"""
Molmo2 Forecasting Inference Script

Runs Molmo2-8B on sports forecasting MCQ data.
Supports multi-GPU distributed inference via torchrun.
Requires the `molmo_utils` package (pip install molmo_utils).

Usage:
  # Single GPU
  python infer_molmo.py --test_json data/basketball_test.json --output results.json

  # Multi-GPU (4 GPUs)
  torchrun --nproc_per_node=4 infer_molmo.py \
      --test_json data/hockey_test.json --output results.json
"""

import argparse
import json
import os
import re
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from molmo_utils import process_vision_info


# =========================
# Distributed setup
# =========================
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(
                backend="nccl", init_method="env://",
                timeout=torch.distributed.default_pg_timeout,
            )
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    return None, 1, 0


# =========================
# MCQ helpers
# =========================
def infer_mcq_options_from_text(question_text: str):
    """Extract MCQ options (A-E) from question text."""
    text = question_text.upper()
    found = set(m.group(1) for m in re.finditer(r"\b([A-E])\s*:", text))
    if not found:
        return None
    order = ["A", "B", "C", "D", "E"]
    max_opt = max(found)
    return order[: order.index(max_opt) + 1]


def get_option_logits_and_probs(processor, outputs, options):
    """Extract logits and softmax probabilities for MCQ option tokens."""
    if not (hasattr(outputs, "scores") and outputs.scores and len(outputs.scores) > 0):
        return None, None
    tokenizer = processor.tokenizer
    opt_token_ids = {}
    for opt in options:
        ids = tokenizer.encode(" " + opt, add_special_tokens=False)
        if not ids:
            return None, None
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
        if r.get("error") or r.get("prediction") is None:
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
        if not options or not probs:
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
    parser = argparse.ArgumentParser(description="Molmo2 Forecasting Inference")
    parser.add_argument("--test_json", required=True, help="Path to test JSON file")
    parser.add_argument("--output", required=True, help="Path to output results JSON")
    parser.add_argument("--model", default="allenai/Molmo2-8B", help="Model name/path")
    parser.add_argument("--sample_fps", type=float, default=0.2, help="Frame sampling rate (FPS)")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Load model
    if rank is None or rank == 0:
        print(f"Loading model: {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto",
        device_map={"": device},
    )
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto", device_map="auto",
    )

    # Load data
    if rank is None or rank == 0:
        print(f"Loading test data from {args.test_json}")
    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    # Split across ranks
    if rank is not None:
        entries_per_gpu = len(test_data) // world_size
        start_idx = rank * entries_per_gpu
        end_idx = start_idx + entries_per_gpu if rank < world_size - 1 else len(test_data)
        test_data = test_data[start_idx:end_idx]
        if rank == 0:
            print(f"Processing {len(test_data)} entries on rank 0 (world_size={world_size})")
    else:
        print(f"Total entries: {len(test_data)}")

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
            print(f"Warning: Video not found: {video_path}")
            results.append({"id": entry_id, "video_path": video_path,
                            "question": question, "ground_truth": ground_truth,
                            "prediction": None, "error": "Video file not found"})
            continue

        mcq_options = infer_mcq_options_from_text(question)
        is_mcq = mcq_options is not None

        try:
            question_with_prompt = question + "\nPlease answer with only the letter (A, B, C, D, or E) corresponding to your choice, without any additional text or explanation."
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path,
                     "frame_sampling_mode": "uniform_last_frame",
                     "num_frames": 999, "max_fps": args.sample_fps},
                    {"type": "text", "text": question_with_prompt},
                ],
            }]

            _, videos, video_kwargs = process_vision_info(messages)
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                videos=videos, video_metadata=video_metadatas,
                text=text, padding=True, return_tensors="pt", **video_kwargs,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                if is_mcq:
                    outputs = model.generate(
                        **inputs, max_new_tokens=2048,
                        return_dict_in_generate=True, output_scores=True, do_sample=False,
                    )
                    generated_ids = outputs.sequences
                else:
                    outputs = None
                    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

            generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
            prediction = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            option_logits, option_probs = None, None
            if is_mcq and outputs is not None:
                option_logits, option_probs = get_option_logits_and_probs(processor, outputs, mcq_options)

            result_entry = {
                "id": entry_id, "video_path": video_path,
                "question": question, "ground_truth": ground_truth,
                "prediction": prediction, "error": None,
            }
            if is_mcq:
                result_entry["mcq_options"] = mcq_options
                if option_probs is not None:
                    result_entry["option_probabilities"] = option_probs
                if option_logits is not None:
                    result_entry["option_logits"] = option_logits
            results.append(result_entry)

        except torch.cuda.OutOfMemoryError as e:
            print(f"[Rank {rank}] CUDA OOM for {entry_id}, skipping...")
            results.append({"id": entry_id, "video_path": video_path,
                            "question": question, "ground_truth": ground_truth,
                            "prediction": None, "error": f"CUDA OOM: {e}"})
        except Exception as e:
            print(f"[Rank {rank}] Error for {entry_id}: {e}")
            results.append({"id": entry_id, "video_path": video_path,
                            "question": question, "ground_truth": ground_truth,
                            "prediction": None, "error": str(e)})

    # Save / merge results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if rank is not None:
        per_gpu_output = args.output.replace(".json", f"_rank_{rank}.json")
        with open(per_gpu_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Rank {rank}] Saved {len(results)} results")

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
