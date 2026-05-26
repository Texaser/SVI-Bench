import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import argparse
import os

from prompts import get_prompt

# Initialize OpenAI client
client = OpenAI()

import json
import time

def call_vlm(system_prompt, user_prompt, model_name="gpt-5.2-2025-12-11", max_retries=3):

    for attempt in range(max_retries):

        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )

        # ----- Pricing -----
        if model_name == "gpt-5.2-2025-12-11":
            PRICE_INPUT = 1.75 / 1_000_000
            PRICE_CACHED_INPUT = 0.175 / 1_000_000
            PRICE_OUTPUT = 14.00 / 1_000_000
        elif model_name == "gpt-5-mini-2025-08-07":
            PRICE_INPUT = 0.25 / 1_000_000
            PRICE_CACHED_INPUT = 0.025 / 1_000_000
            PRICE_OUTPUT = 2 / 1_000_000
        else:
            raise ValueError(f"Unknown model: {model_name}")

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

        # ----- JSON parsing with retry -----
        try:
            result = json.loads(response.output_text)
            return result, cost

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                raise ValueError(
                    f"Failed to parse JSON after {max_retries} attempts.\n"
                    f"Last output:\n{response.output_text}"
                ) from e

            # Optional: small delay before retry
            time.sleep(0.5)

    raise RuntimeError("Unexpected failure in call_vlm")

def save_results(results, score_sums, total_cost, save_json_path):
    with open(f"{save_json_path}_scores_all_new.json", 'w') as f:
        json.dump(results, f, indent=4)

    # Compute averages
    n = len(results)
    avg_scores = {k: v / n for k, v in score_sums.items()}

    save_json = {}
    save_json["total_samples"] = n
    save_json["avg_scores"] = avg_scores
    save_json["total_cost"] = total_cost

    with open(f"{save_json_path}_scores_avg_new.json", 'w') as f:
        json.dump(save_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference")
    parser.add_argument("--model_name", type=str, default="gpt-5.2-2025-12-11")
    parser.add_argument("--sport", type=str, default="basketball")
    parser.add_argument("--test_json_path", type=str, default='/mnt/bum/mmiemon/LLaVA-NeXT/results/sports_pool1_final/sports_100k_f16_full_ft/basketball/val/caption_eval_f16_outputs_1k.json')

    parse_args = parser.parse_args()

    system_prompt, user_prompt = get_prompt(parse_args.sport)

    with open(parse_args.test_json_path, 'r') as f:
        data = json.load(f)
    
    print("Total samples", len(data))

    score_sums = {
        "action_accuracy": 0,
        "identity_accuracy": 0,
        "causality_outcome": 0,
        "spatial_understanding": 0,
        "temporal_understanding": 0,
        "contextual_details": 0,
        "final_holistic_score": 0
    }

    save_json_path = f"{parse_args.test_json_path[:-5]}_{parse_args.model_name}"
    print("Saving to:", save_json_path)

    results = []
    total_cost = 0.0
    for idx, row in enumerate(tqdm(data)):

        ground_truth = row["ground_truth"]
        prediction = row["prediction"]
        formatted_prompt = user_prompt.format(ground_truth, prediction)
        print(formatted_prompt)
        result, cost = call_vlm(system_prompt, formatted_prompt, parse_args.model_name)
        result['cost'] = cost
        result['data_source'] = row["data_source"]
        result['video'] = row["video"]
        result['ground_truth'] = row['ground_truth']
        result['prediction'] = row['prediction']

        results.append(result)       
        total_cost += cost
        for k in score_sums:
            score_sums[k] += result[k]['score']
        
        if idx>0 and idx%10==0:
            save_results(results, score_sums, total_cost, save_json_path)
    
    save_results(results, score_sums, total_cost, save_json_path)

    
