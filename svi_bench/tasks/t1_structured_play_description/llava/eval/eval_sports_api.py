import argparse
import torch
import cv2
import base64
import numpy as np
from PIL import Image
import warnings
from decord import VideoReader, cpu
import re
from tqdm import tqdm
import json
import pandas as pd
import time
import datetime
import os
from collections import defaultdict
from openai import OpenAI
import random
from prompts import get_prompt_caption_generation

import io
from google import genai
from google.genai import types

def resize_with_aspect_ratio(image, short_size=224):
    """Resize image while maintaining aspect ratio, keeping the shortest side = 224."""
    h, w, _ = image.shape
    if h < w:
        new_h, new_w = short_size, int((w / h) * short_size)
    else:
        new_h, new_w = int((h / w) * short_size), short_size
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_video(video_path, max_frames_num):
    """Extract frames at 1 FPS. If total sampled frames exceed max_frames_num,
    uniformly downsample to max_frames_num."""
    
    video_path = video_path if isinstance(video_path, str) else video_path[0]
    vr = VideoReader(video_path, ctx=cpu(0))
    
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    # Number of frames corresponding to 1 second
    step = int(round(fps))
    step = max(step, 1)  # safety
    
    # Sample at 1 FPS
    frame_idx = np.arange(0, total_frames, step)
    
    # If too many frames, uniformly downsample to max_frames_num
    if len(frame_idx) > max_frames_num:
        frame_idx = np.linspace(
            0, len(frame_idx) - 1, max_frames_num, dtype=int
        )
        frame_idx = np.array(frame_idx)
    
    frames = vr.get_batch(frame_idx.tolist()).asnumpy()
    return frames


def frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def frames_to_parts(frames):
    parts = []

    for frame in frames:
        # Convert numpy frame (H, W, 3) -> JPEG bytes
        img = Image.fromarray(frame.astype("uint8"))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

        parts.append(
            types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/jpeg"
            )
        )

    return parts


def call_vlm(client, video_path, question, max_frames=16, model_name="gpt-5.2-2025-12-11"):
    if "gemini" in model_name:
        video_bytes = open(video_path, 'rb').read()
        response = client.models.generate_content(
            model=model_name,
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ),
                    types.Part(text=question)
                ]
            )
        )
        try:
            PROMPT_TOKEN = response.usage_metadata.prompt_token_count
            OUTPUT_TOKEN = response.usage_metadata.candidates_token_count
            if model_name=='gemini-3.1-pro-preview':
                if PROMPT_TOKEN<=200000:
                    cost = (2 * PROMPT_TOKEN + 12 * OUTPUT_TOKEN)/1000000
                else:
                    cost = (4 * PROMPT_TOKEN + 18 * OUTPUT_TOKEN)/1000000
            elif model_name=='gemini-3-flash-preview':
                cost = (0.5 * PROMPT_TOKEN + 3 * OUTPUT_TOKEN)/1000000
        except:
            cost = 0
        return response.text, cost

    frames = load_video(video_path, max_frames)
    content = []
    content.append({
        "type": "input_text",
        "text": question
    })
    for f in frames:
        img_b64 = frame_to_base64(f)

        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{frame_to_base64(f)}",
        })
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    if model_name=="gpt-5.2-2025-12-11":
        PRICE_INPUT = 1.75 / 1_000_000       
        PRICE_CACHED_INPUT = 0.175 / 1_000_000 
        PRICE_OUTPUT = 14.00 / 1_000_000       
    elif model_name=="gpt-5-mini-2025-08-07":
        PRICE_INPUT = 0.25 / 1_000_000        
        PRICE_CACHED_INPUT = 0.025 / 1_000_000 
        PRICE_OUTPUT = 2 / 1_000_000      
    usage = response.usage
    input_tokens = usage.input_tokens
    cached_input_tokens = usage.input_tokens_details.cached_tokens or 0
    output_tokens = usage.output_tokens
    normal_input_tokens = input_tokens - cached_input_tokens
    cost = (
        normal_input_tokens * PRICE_INPUT
        + cached_input_tokens * PRICE_CACHED_INPUT
        + output_tokens * PRICE_OUTPUT
    )
    return response.output_text, cost

def calc_results(logs):
    source_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_cost = 0
    for sample in logs:
        is_correct = int(sample['ground_truth'].strip() == sample['prediction'].strip())
        source_stats[sample['question_type']]['correct'] += is_correct
        source_stats[sample['question_type']]['total'] += 1
        total_cost += sample['cost']

    # Calculate per-source accuracy and overall totals
    result_json = {}
    total_correct = 0
    total_total = 0

    for source, stats in source_stats.items():
        correct = stats['correct']
        total = stats['total']
        acc = correct / total if total > 0 else 0
        result_json[source] = {
            'accuracy': round(acc, 4),
            'correct': correct,
            'total': total
        }
        total_correct += correct
        total_total += total

    # Add overall accuracy
    result_json['overall'] = {
        'accuracy': round(total_correct / total_total, 4) if total_total > 0 else 0,
        'correct': total_correct,
        'total': total_total,
        'cost': total_cost,
    }
    return result_json


def run_eval(chunk, args):
    global_start_time = time.time()
    batch_start_time = time.time() 
    total_samples = len(chunk)
    logs = []

    os.makedirs(parse_args.results_dir, exist_ok=True)

    output_path = f"{parse_args.results_dir}/{parse_args.eval_type}_eval_f{parse_args.eval_frames}"
    print("Saving to:", output_path)

    existing_results = {}
    if os.path.exists(f"{output_path}_outputs.json"):
        with open(f"{output_path}_outputs.json", 'r') as f:
            existing_results = json.load(f)
            existing_results = {x['video']:x for x in existing_results}
        print("Found existing", f"{output_path}_outputs.json", len(existing_results))

    if "gpt" in args.model_name:
        client = OpenAI()
    elif 'gemini' in args.model_name:
        client = genai.Client()
    else:
        raise NotImplementedError

    if args.eval_type == 'caption':
        prompt_caption = get_prompt_caption_generation(args.sport)
    
    for idx, sample in enumerate(tqdm(chunk)):
        try:
            if sample['video'] in existing_results:
                logs.append(existing_results[sample['video']])
                continue

            if args.eval_type == 'qa':
                question = sample['question'] + "\n"
                # ins = "Answer by providing only the single letter that corresponds to the correct option."
                ins = "Answer by providing only the single letter that corresponds to the correct option (A, B, C, D, E). Don't output any explanation, punctuation or additional text. Your output should be exactly one letter."
                question += ins + "\n"
                for op in sample['options']:
                    question += op + "\n"
                question = question.strip()
            elif args.eval_type == 'caption':
                question = prompt_caption

            prediction, cost = call_vlm(client, sample['video'], question, max_frames=args.eval_frames, model_name=args.model_name)
            if args.eval_type == 'qa':
                logs.append({
                    'id': sample['id'],
                    'video': sample['video'],
                    'question_type': sample['question_type'],
                    'question': sample['question'],
                    'options': sample['options'],
                    'ground_truth': sample['answer'],
                    'prediction': prediction,
                    'cost': cost,
                })
            elif args.eval_type == 'caption':
                logs.append({
                    'data_source': sample['data_source'],
                    'video': sample['video'],
                    'ground_truth': sample['caption'],
                    'prediction': prediction,
                    'cost': cost,
                })

            if idx%10==0:
                with open(f"{output_path}_outputs.json", 'w', encoding='utf-8') as f:
                    json.dump(logs, f, ensure_ascii=False, indent=4)
                
                if not parse_args.infer_only:
                    result_json = calc_results(logs)
                    print(result_json)
                    with open(f"{output_path}_results.json", 'w') as f:
                        json.dump(result_json, f, indent=4)
        except Exception as e:
            print(f"Error : {e}")
    
    with open(f"{output_path}_outputs.json", 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
    
    if not parse_args.infer_only:
        result_json = calc_results(logs)
        print(result_json)
        with open(f"{output_path}_results.json", 'w') as f:
            json.dump(result_json, f, indent=4)

    total_time = time.time() - global_start_time
    print(f"Overall time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference")
    parser.add_argument("--model_name", type=str, default="gemini-3.1-pro-preview")  #gemini-3.1-pro-preview
    parser.add_argument("--sport", type=str, default="basketball")
    parser.add_argument("--results_dir", type=str, default="results/sports_pool1_final/gpt-5.2")
    parser.add_argument("--eval_frames", type=int, default=16)
    parser.add_argument("--test_json_path", type=str, default='/mnt/bum/mmiemon/LLaVA-NeXT/DATAS/sports_pool1_final/basketball_caption_val_1k.json')
    parser.add_argument("--eval_type", type=str, default='caption')
    parser.add_argument("--infer_only", action="store_true")
    parse_args = parser.parse_args()

    with open(parse_args.test_json_path, "rb") as file:
        test_data = json.load(file)
    # if parse_args.eval_type=='qa':
    #     test_data = test_data[::10]

    print(f"Loaded {len(test_data)} samples from : {parse_args.test_json_path}")

    run_eval(test_data, parse_args)