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

from collections import defaultdict

warnings.filterwarnings("ignore")
mp.set_start_method('spawn', force=True)

# Load the OneVision model parameters
# pretrained = "/mnt/meg/yulupan/LLaVA-NeXT/llava-onevision-qwen2-7b-ov"
# pretrained = "/mnt/meg/yulupan/LLaVA-NeXT/output/test/BASKET_QA_tune_all_llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_10%_lora" # Testing
# pretrained = "output/finetune/BASKET_QA_tune_all_llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_10%"
pretrained = 'output/finetune/BASKET_QA_tune_all_llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_50%_full_update'

llava_model_args = {"multimodal": True}

def load_model(device, model_name, model_base, model_path):
    """Initialize model on a specific GPU."""
    # model_name = "llava_qwen"
    # tokenizer, model, image_processor, max_length = load_pretrained_model(
    #     pretrained, None, model_name, device_map={"": device}, attn_implementation="flash_attention_2", **llava_model_args,
    # )
    # With LoRA
    # model_name = "llava_qwen_lora"
    # tokenizer, model, image_processor, max_length = load_pretrained_model(
    #     pretrained, "/mnt/meg/yulupan/LLaVA-NeXT/llava-onevision-qwen2-7b-ov", model_name, device_map={"": device}, attn_implementation="flash_attention_2", **llava_model_args,
    # )
    # model.eval()
    # print(model)
    # exit()
    tokenizer, model, image_processor, max_length = load_pretrained_model(
                                                        # model_path = "lmms-lab/LLaVA-Video-7B-Qwen2", 
                                                        # model_base = None, 
                                                        # model_name = "llava_qwen",
                                                        model_path = model_path, 
                                                        model_base = model_base,
                                                        model_name = model_name, 
                                                        device_map={"": device}, #change_me
                                                        #device_map = "auto",
                                                    )
    # return tokenizer, model.to(device), image_processor
    return tokenizer, model, image_processor

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

    # frames = np.array([resize_with_aspect_ratio(frame) for frame in frames])
    # print(frames.shape)

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
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}" #change_me
    #device = "cuda"
    tokenizer, model, image_processor = load_model(device, args.model_name, args.model_base, args.model_path)

    total_matches, total_number = 0, 0

    batch_size = 100
    global_start_time = time.time()  # Overall start time for the loop
    batch_start_time = time.time()   # Start time for the current batch
    total_samples = len(chunk)
    
    for sample in tqdm(chunk, desc=f"GPU {gpu_id}"):
    #for idx, sample in enumerate(chunk):
        try:
            video_frames = load_video(sample['video'], args.eval_frames)
            # video_frames = load_images_from_folder(sample['video'])
            frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(device)
            
            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            question = sample['conversations'][0]['value'].replace('<image>', DEFAULT_IMAGE_TOKEN)
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
            
            if text_outputs[0] == sample['conversations'][1]['value']:
                total_matches += 1
            total_number += 1

            if 'question_type' in sample:
                logs.append({
                    'video': sample['video'],
                    'question_type': sample['question_type'],
                    'ground_truth': sample['conversations'][1]['value'],
                    'prediction': text_outputs[0]
                })
            else:
                logs.append({
                    'video': sample['video'],
                    'ground_truth': sample['conversations'][1]['value'],
                    'prediction': text_outputs[0]
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
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-72B-Qwen2")
    parser.add_argument("--results_dir", type=str, default="results/sports/test")
    parser.add_argument("--train_frames", type=int, default=16)
    parser.add_argument("--eval_frames", type=int, default=16)
    parser.add_argument("--test_json_path", type=str, default='DATAS/eval/Sports/test_20.json')
    parser.add_argument("--eval_type", type=str, default='qa')
    parser.add_argument("--infer_only", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit evaluation to first N samples (0 = all)")
    parse_args = parser.parse_args()

    # test_json_path = '/mnt/opr/yulupan/basketball_QA_dataset/more_sports/soccer/QA_generation/QA_split/total_test.json'
    # test_json_path = 'DATAS/eval/Sports/test_20.json'
    # test_json_path = 'DATAS/eval/Sports/caption_test.json'
    # test_json_path = '/mnt/opr/yulupan/basketball_QA_dataset/utils/Q5_player_name_same_team/test.json'
    # test_json_path = '/mnt/opr/yulupan/basketball_QA_dataset/utils/Q5_player_number_two_team_similar_update/test.json'

    with open(parse_args.test_json_path, "rb") as file:
        test_data = json.load(file)

    if parse_args.max_samples > 0:
        test_data = test_data[:parse_args.max_samples]

    # random.shuffle(test_data)
    num_gpus = torch.cuda.device_count() #change_me
    #num_gpus = 1
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")
    
    test_chunks = split_list(test_data, num_gpus)
    manager = mp.Manager()
    results = manager.dict()
    logs = manager.list() 

    processes = []
    
    for gpu_id, chunk in enumerate(test_chunks):
        p = mp.Process(target=process_chunk, args=(gpu_id, chunk, results, logs, parse_args))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

    os.makedirs(parse_args.results_dir, exist_ok=True)

    # Save the logs to a CSV file with the desired columns
    with open(f"{parse_args.results_dir}/{parse_args.eval_type}_eval_f{parse_args.eval_frames}_outputs.json", 'w') as f:
        json.dump(list(logs), f, indent=4)
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

# sbatch --nodelist="mirage.ib" --job-name eval_llavaov --gpus 8 --wrap="python test_llavaov.py"
# sbatch --job-name eval_llavaov --gpus 8 --wrap="python test_llavaov.py"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_llavaov.py