from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import re
from tqdm import tqdm
import json
import random
import torch.multiprocessing as mp
import pandas as pd
import time
import datetime
import os
from prompts import get_prompt_caption_generation

from collections import defaultdict

warnings.filterwarnings("ignore")
mp.set_start_method('spawn', force=True)


def load_model(device, model_name, model_base, model_path):
    tokenizer, model, image_processor, max_length = load_pretrained_model(
                                                        model_path = model_path, 
                                                        model_base = model_base,
                                                        model_name = model_name, 
                                                        device_map={"": device},
                                                    )
    return tokenizer, model.to(device), image_processor

def resize_with_aspect_ratio(image, short_size=224):
    """Resize image while maintaining aspect ratio, keeping the shortest side = 224."""
    h, w, _ = image.shape
    if h < w:
        new_h, new_w = short_size, int((w / h) * short_size)
    else:
        new_h, new_w = int((h / w) * short_size), short_size
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def load_video(video_path, max_frames_num):
    """Function to extract frames from video"""
    vr = VideoReader(video_path if isinstance(video_path, str) else video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    return frames

def load_images_from_folder(folder_path):
    """Load images from a folder, sort them, and return a stacked NumPy array."""
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    image_frames = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert("RGB")

        image_array = np.array(image) 
        image_frames.append(image_array)
    image_frames = np.stack(image_frames, axis=0)  
    return image_frames  

def split_list(lst, num_splits):
    """Split a list into evenly sized chunks"""
    chunk_size = len(lst) // num_splits
    remainder = len(lst) % num_splits
    return [lst[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(num_splits)]

def process_chunk(gpu_id, chunk, results, logs, args):
    """Process a chunk of data on a specific GPU."""
    device = f"cuda:{gpu_id}"
    tokenizer, model, image_processor = load_model(device, args.model_name, args.model_base, args.model_path)

    total_matches, total_number = 0, 0

    batch_size = 100
    global_start_time = time.time()  
    batch_start_time = time.time() 
    total_samples = len(chunk)

    if args.eval_type == 'caption':
        prompt_caption = get_prompt_caption_generation(args.sport)
    
    for sample in tqdm(chunk, desc=f"GPU {gpu_id}"):
        try:
            video_frames = load_video(sample['video'], args.eval_frames)
            frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(device)
            
            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            question = "<image>\n"

            if args.eval_type == 'qa':
                question += sample['question'] + "\n"
                ins = "Answer by providing only the single letter that corresponds to the correct option."
                question += ins + "\n"
                for op in sample['options']:
                    question += op + "\n"
                question = question.strip()
            elif args.eval_type == 'caption':
                if args.model_path=="lmms-lab/LLaVA-Video-7B-Qwen2":  #zero-shot model
                    question += prompt_caption
                else:
                    question += "Write a detailed caption of the sports video clip, describing the key actions, events, and the final outcome."
            else:
                raise NotImplementedError

            question = question.replace('<image>', DEFAULT_IMAGE_TOKEN)

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            
            
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
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
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            prediction = text_outputs[0].strip()

            if args.eval_type == 'qa':
                if prediction == sample['answer']:
                    total_matches += 1
                total_number += 1

                logs.append({
                    'id': sample['id'],
                    'video': sample['video'],
                    'question_type': sample['question_type'],
                    'question': sample['question'],
                    'options': sample['options'],
                    'ground_truth': sample['answer'],
                    'prediction': prediction
                })
            else:
                logs.append({
                    'data_source': sample['data_source'],
                    'video': sample['video'],
                    'ground_truth': sample['caption'],
                    'prediction': prediction
                })
                
        
        except Exception as e:
            print(f"Error on GPU {gpu_id}: {e}")
            continue

    total_time = time.time() - global_start_time
    print(f"GPU: {gpu_id}, Overall time: {total_time:.2f} seconds.")
    
    results[gpu_id] = total_matches, total_number

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--sport", type=str, default="basketball")
    parser.add_argument("--results_dir", type=str, default="results/sports/test")
    parser.add_argument("--train_frames", type=int, default=16)
    parser.add_argument("--eval_frames", type=int, default=16)
    parser.add_argument("--test_json_path", type=str, default='/mnt/bum/mmiemon/LLaVA-NeXT/DATAS/sports_pool1_final/basketball_caption_val_5k.json')
    parser.add_argument("--eval_type", type=str, default='caption')
    parser.add_argument("--infer_only", action="store_true")
    parse_args = parser.parse_args()

    parse_args.infer_only = True

    with open(parse_args.test_json_path, "rb") as file:
        test_data = json.load(file)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")
    
    test_chunks = split_list(test_data, num_gpus)
    manager = mp.Manager()
    results = manager.dict()
    logs = manager.list() 

    processes = []
    
    for gpu_id, chunk in enumerate(test_chunks):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p = mp.Process(target=process_chunk, args=(0, chunk, results, logs, parse_args))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

    os.makedirs(parse_args.results_dir, exist_ok=True)

    # Save the logs to a CSV file with the desired columns
    with open(f"{parse_args.results_dir}/{parse_args.eval_type}_eval_f{parse_args.eval_frames}_outputs.json", 'w', encoding='utf-8') as f:
        json.dump(list(logs), f, ensure_ascii=False, indent=4)

    # df = pd.DataFrame(list(logs), columns=["data_source", "ground_truth", "prediction"])
    # df.to_csv(f"{parse_args.results_dir}/eval_f{parse_args.eval_frames}_outputs100.csv", index=False)

    if not parse_args.infer_only:
        total_matches = sum(res[0] for res in results.values())
        total_number = sum(res[1] for res in results.values())
        accuracy = total_matches / total_number if total_number > 0 else 0
        print(f'Average accuracy: {accuracy}')

        source_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        # Populate stats
        # for video, data_source, ground_truth, prediction in logs:
        for sample in logs:
            is_correct = int(sample['ground_truth'].strip() == sample['prediction'].strip())
            source_stats[sample['question_type']]['correct'] += is_correct
            source_stats[sample['question_type']]['total'] += 1

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
            'total': total_total
        }
        print(result_json)

        # Save to JSON file
        with open(f"{parse_args.results_dir}/{parse_args.eval_type}_eval_f{parse_args.eval_frames}_results.json", 'w') as f:
            json.dump(result_json, f, indent=4)

    print('finish testing llavaov')