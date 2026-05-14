#!/usr/bin/env python3
"""
Convert training JSON files to Swift JSONL format for Qwen3-VL fine-tuning.

The input JSON files use the SVI-Bench conversation format:
  {"id": "...", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}], "video": "..."}

The output JSONL uses the ms-swift messages format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "videos": ["..."]}

Usage:
  # Single sport
  python convert_train_to_jsonl.py --input data/basketball_train.json --output data/train.jsonl

  # All sports combined
  python convert_train_to_jsonl.py \
      --input data/basketball_train.json data/hockey_train.json data/soccer_train.json \
      --output data/train.jsonl
"""

import argparse
import json


def convert_entry(entry: dict) -> dict:
    """Convert a single entry from conversation format to Swift messages format."""
    messages = []
    for conv in entry.get("conversations", []):
        if conv.get("from") == "human":
            messages.append({"role": "user", "content": conv.get("value", "")})
        elif conv.get("from") == "gpt":
            messages.append({"role": "assistant", "content": conv.get("value", "")})

    video_path = entry.get("video", "")
    result = {"messages": messages}
    if video_path:
        result["videos"] = [video_path]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert training JSON to Swift JSONL format")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input JSON file(s)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path")
    args = parser.parse_args()

    total = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for input_path in args.input:
            print(f"Processing {input_path}...")
            with open(input_path, "r", encoding="utf-8") as in_f:
                data = json.load(in_f)
            count = 0
            for entry in data:
                converted = convert_entry(entry)
                if converted["messages"]:
                    out_f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
            print(f"  Wrote {count} entries")
            total += count

    print(f"\nTotal: {total} entries written to {args.output}")


if __name__ == "__main__":
    main()
