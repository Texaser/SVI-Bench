#!/usr/bin/env python3
"""
Split validation set into N chunks for parallel processing.
"""

import argparse
from pathlib import Path


def split_file(input_file, output_dir, num_splits):
    """Split input file into num_splits chunks."""
    # Read all lines
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    chunk_size = (total_lines + num_splits - 1) // num_splits  # Ceiling division
    
    print(f"Total samples: {total_lines}")
    print(f"Number of splits: {num_splits}")
    print(f"Samples per split: ~{chunk_size}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    input_basename = Path(input_file).stem
    
    # Split into chunks
    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_lines)
        chunk = lines[start_idx:end_idx]
        
        output_file = output_dir / f"{input_basename}_split_{i}.txt"
        with open(output_file, 'w') as f:
            for line in chunk:
                f.write(line + '\n')
        
        print(f"Split {i}: {len(chunk)} samples -> {output_file}")
    
    print(f"\n✓ Created {num_splits} split files in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Split validation set for parallel processing')
    parser.add_argument('--input', type=str, required=True, help='Input file to split')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for split files')
    parser.add_argument('--num-splits', type=int, default=8, help='Number of splits (default: 8)')
    
    args = parser.parse_args()
    
    split_file(args.input, args.output_dir, args.num_splits)


if __name__ == '__main__':
    main()
