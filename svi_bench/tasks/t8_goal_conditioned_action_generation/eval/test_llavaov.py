"""T8 goal-accuracy worker: run a fine-tuned LLaVA-Qwen QA model over one or
more Q*.json files, sharded across visible GPUs.

The model is loaded once per GPU and reused across all input Q*.json files,
so the 8-question-type pass costs one model load per GPU, not eight.

Per-Q*.json results land at:
    <results_dir>/<Q*-stem>/<eval_type>_eval_f<frames>_outputs.json
    <results_dir>/<Q*-stem>/<eval_type>_eval_f<frames>_results.json
"""
import argparse
import copy
import json
import os
import time
import warnings
from collections import defaultdict
from operator import attrgetter

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")
mp.set_start_method("spawn", force=True)


def load_model(device, model_name, model_base, model_path):
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device_map={"": device},
    )
    return tokenizer, model, image_processor


def load_video(video_path, max_frames_num):
    vr = VideoReader(video_path if isinstance(video_path, str) else video_path[0], ctx=cpu(0))
    frame_idx = np.linspace(0, len(vr) - 1, max_frames_num, dtype=int).tolist()
    return vr.get_batch(frame_idx).asnumpy()


def split_list(lst, num_splits):
    chunk_size = len(lst) // num_splits
    remainder = len(lst) % num_splits
    return [
        lst[i * chunk_size + min(i, remainder): (i + 1) * chunk_size + min(i + 1, remainder)]
        for i in range(num_splits)
    ]


def process_chunk(gpu_id, chunk, logs, args):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    tokenizer, model, image_processor = load_model(
        device, args.model_name, args.model_base, args.model_path,
    )

    t0 = time.time()
    for sample in tqdm(chunk, desc=f"GPU {gpu_id}"):
        try:
            video_frames = load_video(sample["video"], args.eval_frames)
            frames = (
                image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
                .half()
                .to(device)
            )

            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            question = sample["conversations"][0]["value"].replace("<image>", DEFAULT_IMAGE_TOKEN)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            input_ids = (
                tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(device)
            )
            image_sizes = [frame.size for frame in video_frames]

            cont = model.generate(
                input_ids,
                images=[frames],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                modalities=["video"],
            )
            pred = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

            logs.append({
                "_qa_type":     sample["_qa_type"],
                "video":        sample["video"],
                "question_type": sample.get("question_type", ""),
                "ground_truth": sample["conversations"][1]["value"],
                "prediction":   pred,
            })
        except Exception as e:
            print(f"GPU {gpu_id} error on {sample.get('video')}: {e}")

    print(f"GPU {gpu_id} done in {time.time() - t0:.1f}s")


def write_per_file_results(logs, args):
    """Group logs by source Q*.json file and write per-file outputs/results."""
    by_qa_type = defaultdict(list)
    for entry in logs:
        by_qa_type[entry["_qa_type"]].append(entry)

    for qa_type, entries in by_qa_type.items():
        out_dir = os.path.join(args.results_dir, qa_type)
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, f"{args.eval_type}_eval_f{args.eval_frames}_outputs.json"), "w") as f:
            stripped = [{k: v for k, v in e.items() if k != "_qa_type"} for e in entries]
            json.dump(stripped, f, indent=4)

        if args.infer_only:
            continue

        source_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for e in entries:
            is_correct = int(e["ground_truth"].strip() == e["prediction"].strip())
            source_stats[e["question_type"]]["correct"] += is_correct
            source_stats[e["question_type"]]["total"] += 1

        result = {}
        total_correct = total_total = 0
        for source, stats in source_stats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] else 0
            result[source] = {
                "accuracy": round(acc, 4),
                "correct":  stats["correct"],
                "total":    stats["total"],
            }
            total_correct += stats["correct"]
            total_total += stats["total"]
        result["overall"] = {
            "accuracy": round(total_correct / total_total, 4) if total_total else 0,
            "correct":  total_correct,
            "total":    total_total,
        }

        with open(os.path.join(out_dir, f"{args.eval_type}_eval_f{args.eval_frames}_results.json"), "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLaVA-Qwen QA over one or more Q*.json files")
    parser.add_argument("--model_name", default="llava_qwen")
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--results_dir", required=True,
                        help="Parent dir; per-Q*.json subdirs get created here")
    parser.add_argument("--eval_frames", type=int, default=16)
    parser.add_argument("--test_json_paths", nargs="+", required=True,
                        help="One or more Q*.json files; model loads once per GPU and runs them all")
    parser.add_argument("--eval_type", default="qa")
    parser.add_argument("--infer_only", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit each Q*.json to first N samples (0 = all)")
    args = parser.parse_args()

    all_samples = []
    for jf in args.test_json_paths:
        qa_type = os.path.splitext(os.path.basename(jf))[0]
        with open(jf, "rb") as f:
            data = json.load(f)
        if args.max_samples > 0:
            data = data[: args.max_samples]
        for sample in data:
            sample["_qa_type"] = qa_type
        all_samples.extend(data)
    print(f"Loaded {len(all_samples)} samples across {len(args.test_json_paths)} files")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    chunks = split_list(all_samples, num_gpus)
    manager = mp.Manager()
    logs = manager.list()

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(target=process_chunk, args=(gpu_id, chunk, logs, args))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    write_per_file_results(list(logs), args)
    print("done")
