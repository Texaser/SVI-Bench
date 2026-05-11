#!/usr/bin/env python3
"""Task2 final validation: per-video prompts from polished captions,
first/last frame bbox only. Saves output per clip name for evaluation."""

import torch
import numpy as np
import os
import sys
import json
import glob
import random
from pathlib import Path
from PIL import Image, ImageDraw
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.unified_dataset import PolishedCaptionsLookup, LoadBBoxFromPlayerSpecs


def visualize_bbox(bbox_data, width=832, height=480):
    """Visualize sparse bbox data (only first/last frames have boxes)."""
    frames = []
    num_frames = bbox_data.shape[0]
    num_coords = bbox_data.shape[1]
    num_players = num_coords // 4

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 255), (255, 128, 128),
    ]

    for frame_idx in range(num_frames):
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        for player_idx in range(num_players):
            start_idx = player_idx * 4
            x1_norm, y1_norm, x2_norm, y2_norm = bbox_data[frame_idx, start_idx:start_idx+4]

            if x1_norm == 0 and y1_norm == 0 and x2_norm == 0 and y2_norm == 0:
                continue
            if x2_norm <= x1_norm or y2_norm <= y1_norm:
                continue

            x1 = max(0, min(int(x1_norm * width), width - 1))
            y1 = max(0, min(int(y1_norm * height), height - 1))
            x2 = max(0, min(int(x2_norm * width), width))
            y2 = max(0, min(int(y2_norm * height), height))

            color = colors[player_idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 5, y1 + 5), f"P{player_idx+1}", fill=color)

        frames.append(img)

    return frames


def main():
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        print("Usage: python ...task2-final.py <checkpoint_path>")
        sys.exit(1)

    # Validation parameters
    VALIDATION_BBOX_FOLDER = os.environ.get(
        'VALIDATION_BBOX_FOLDER',
        '/mnt/bum/hanyi/repo/sports_detection/segment-anything-2-real-time/basketball_set/test_task2_final.txt'
    )
    VALIDATION_VIDEO_BASE = os.environ.get(
        'VALIDATION_VIDEO_BASE',
        '/mnt/bum/hanyi/data/basketball_fps_15_task2'
    )
    VALIDATION_BACKGROUND_VIDEO_BASE = os.environ.get(
        'VALIDATION_BACKGROUND_VIDEO_BASE',
        '/mnt/bum/hanyi/data/basketball_inpainting_video_task2'
    )
    NUM_VALIDATION_SAMPLES = int(os.environ.get('NUM_VALIDATION_SAMPLES', '5000'))
    VALIDATION_NUM_FRAMES = int(os.environ.get('VALIDATION_NUM_FRAMES', '81'))
    VALIDATION_TIME_DIVISION_FACTOR = int(os.environ.get('VALIDATION_TIME_DIVISION_FACTOR', '1'))

    POLISHED_CAPTIONS = os.environ.get(
        'POLISHED_CAPTIONS',
        '/mnt/bum/hanyi/repo/sports_detection/segment-anything-2-real-time/polished_captions_final.json'
    ).split(',')

    print(f"Loading LoRA checkpoint: {checkpoint_path}")
    print(f"Validation bbox folder: {VALIDATION_BBOX_FOLDER}")
    print(f"Validation video base: {VALIDATION_VIDEO_BASE}")
    print(f"Validation background video base: {VALIDATION_BACKGROUND_VIDEO_BASE}")
    print(f"Number of validation samples: {NUM_VALIDATION_SAMPLES}")
    print(f"Polished captions: {POLISHED_CAPTIONS}")

    BBOX_CHANNELS = int(os.environ.get('BBOX_CHANNELS', '16'))
    BACKGROUND_VIDEO_CHANNELS = int(os.environ.get('BACKGROUND_VIDEO_CHANNELS', '8'))

    use_overlay_env = os.environ.get('USE_OVERLAY_METHOD', '1')
    use_overlay_method = str(use_overlay_env).strip().lower() in ('1', 'true', 'yes', 'y')
    bbox_color_mode = os.environ.get('BBOX_COLOR_MODE', 'color')
    print(f"BBox color mode: {bbox_color_mode}, overlay: {use_overlay_method}")

    # Load polished captions lookup
    captions_lookup = PolishedCaptionsLookup(POLISHED_CAPTIONS)
    bbox_from_specs = LoadBBoxFromPlayerSpecs(VALIDATION_NUM_FRAMES)

    # Load pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
        ],
    )
    try:
        setattr(pipe, 'use_overlay_method', use_overlay_method)
    except Exception:
        pass

    # Load LoRA checkpoint
    if os.path.exists(checkpoint_path):
        try:
            pipe.load_lora(pipe.dit, checkpoint_path, alpha=1.0)
            print(f"Loaded LoRA checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading LoRA checkpoint: {e}")
            print("Continuing with base model...")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    pipe.enable_vram_management()

    output_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    validation_output_dir = os.path.join(output_dir, "validation", checkpoint_name)
    os.makedirs(validation_output_dir, exist_ok=True)

    # Load validation bbox paths
    if not os.path.exists(VALIDATION_BBOX_FOLDER):
        print(f"Validation bbox folder not found: {VALIDATION_BBOX_FOLDER}")
        sys.exit(1)

    bbox_files = []
    if os.path.isfile(VALIDATION_BBOX_FOLDER):
        with open(VALIDATION_BBOX_FOLDER, 'r') as f:
            for line in f:
                p = line.strip()
                if p and (p.endswith('.npz') or p.endswith('.txt')):
                    if os.path.exists(p):
                        bbox_files.append(p)
    else:
        bbox_files = glob.glob(os.path.join(VALIDATION_BBOX_FOLDER, "**/*.txt"), recursive=True)
        bbox_files += glob.glob(os.path.join(VALIDATION_BBOX_FOLDER, "**/*.npz"), recursive=True)

    if not bbox_files:
        print(f"No bbox files found in {VALIDATION_BBOX_FOLDER}")
        sys.exit(1)

    print(f"Found {len(bbox_files)} bbox files in validation set")

    # Filter to only those with polished captions (player_specifications)
    bbox_files_with_specs = []
    for bp in bbox_files:
        prompt, specs = captions_lookup.get_entry(bp)
        if specs is not None:
            bbox_files_with_specs.append(bp)
    print(f"  {len(bbox_files_with_specs)} have player_specifications in polished captions")

    if not bbox_files_with_specs:
        print("No validation samples with player_specifications found!")
        sys.exit(1)

    # Use all samples (up to NUM_VALIDATION_SAMPLES), deterministic order
    num_samples = min(NUM_VALIDATION_SAMPLES, len(bbox_files_with_specs))
    if num_samples < len(bbox_files_with_specs):
        random.seed(42)
        sampled_bbox_files = random.sample(bbox_files_with_specs, num_samples)
    else:
        sampled_bbox_files = bbox_files_with_specs

    print(f"Generating {len(sampled_bbox_files)} validation videos...")

    for idx, bbox_path in enumerate(sampled_bbox_files):
        try:
            # Get per-video prompt and player specifications
            prompt, player_specs = captions_lookup.get_entry(bbox_path)

            # Build sparse bbox from player_specifications (first/last frame only)
            bbox_data = bbox_from_specs(player_specs, num_frames=VALIDATION_NUM_FRAMES)

            # Derive video path from bbox path
            bbox_path_normalized = os.path.normpath(bbox_path)
            parts = bbox_path_normalized.split(os.sep)
            mixsort_idx = None
            for i, part in enumerate(parts):
                if 'mixsort' in part or 'basketball_mixsort' in part:
                    mixsort_idx = i
                    break

            if mixsort_idx is not None:
                relative_parts = parts[mixsort_idx + 1:]
                relative_path = os.path.join(*relative_parts) if relative_parts else ""
            else:
                relative_path = os.path.basename(bbox_path)

            video_relative = os.path.splitext(relative_path)[0] + '.mp4'
            video_path = os.path.join(VALIDATION_VIDEO_BASE, video_relative)
            background_video_path = os.path.join(VALIDATION_BACKGROUND_VIDEO_BASE, video_relative)

            # Clip name for output directory
            bbox_filename_base = Path(bbox_path).stem

            # Skip if already generated
            sample_output_dir = os.path.join(validation_output_dir, bbox_filename_base)
            generated_path = os.path.join(sample_output_dir, "generated.mp4")
            if os.path.exists(generated_path):
                if (idx + 1) % 100 == 0:
                    print(f"[{idx+1}/{len(sampled_bbox_files)}] Skipping (exists): {bbox_filename_base}")
                continue

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"\n[{idx+1}/{len(sampled_bbox_files)}] Processing: {bbox_filename_base}")
                print(f"  Prompt: {prompt[:100]}...")
                print(f"  Players: {len(player_specs)}, BBox shape: {bbox_data.shape}")

            # Load video
            if not os.path.exists(video_path):
                if (idx + 1) % 100 == 0:
                    print(f"  Video not found: {video_path}")
                continue

            # Read original video dimensions before center-crop-resize (for bbox alignment)
            import imageio as _iio
            try:
                _reader = _iio.get_reader(video_path)
                _first = _reader.get_data(0)
                orig_video_height, orig_video_width = _first.shape[:2]
                _reader.close()
            except Exception as _e:
                print(f"  Could not read original video size, falling back to 832x480: {_e}")
                orig_video_width, orig_video_height = 832, 480

            video = VideoData(video_path, height=480, width=832)
            total_video_frames = len(video)
            start_idx = 0
            step = VALIDATION_TIME_DIVISION_FACTOR

            ground_truth_frames = []
            for i in range(VALIDATION_NUM_FRAMES):
                frame_idx = start_idx + i * step
                if frame_idx < total_video_frames:
                    ground_truth_frames.append(video[frame_idx])
                elif ground_truth_frames:
                    ground_truth_frames.append(ground_truth_frames[-1])
                else:
                    break

            if not ground_truth_frames:
                continue

            first_frame = ground_truth_frames[0]
            if not hasattr(first_frame, 'mode') or first_frame.mode != 'RGB':
                first_frame = first_frame.convert('RGB')

            # Load background video
            background_video_frames = None
            if os.path.exists(background_video_path):
                try:
                    bg_video = VideoData(background_video_path, height=480, width=832)
                    total_bg_frames = len(bg_video)
                    background_video_frames = []
                    for i in range(VALIDATION_NUM_FRAMES):
                        frame_idx = start_idx + i * step
                        if frame_idx < total_bg_frames:
                            frame = bg_video[frame_idx]
                            if not hasattr(frame, 'mode') or frame.mode != 'RGB':
                                frame = frame.convert('RGB')
                            background_video_frames.append(frame)
                        elif background_video_frames:
                            background_video_frames.append(background_video_frames[-1])
                        else:
                            background_video_frames.append(first_frame)
                    while len(background_video_frames) < VALIDATION_NUM_FRAMES:
                        background_video_frames.append(background_video_frames[-1] if background_video_frames else first_frame)
                except Exception as e:
                    print(f"  Error loading background video: {e}")
                    background_video_frames = None

            # Generate video
            try:
                bbox_tensor = torch.from_numpy(bbox_data).float()
                if hasattr(pipe, 'device') and pipe.device is not None:
                    bbox_tensor = bbox_tensor.to(pipe.device)

                pipe_kwargs = {
                    "prompt": prompt,
                    "input_image": first_frame,
                    "bbox": bbox_tensor,
                    "bbox_channels": BBOX_CHANNELS,
                    "bbox_color_mode": bbox_color_mode,
                    "bbox_first_last_only": True,
                    "use_overlay_method": use_overlay_method,
                    "orig_video_width": orig_video_width,  # For bbox center-crop alignment
                    "orig_video_height": orig_video_height,
                    "height": 480,
                    "width": 832,
                    "num_frames": VALIDATION_NUM_FRAMES,
                    "cfg_scale": 2.5,
                    "num_inference_steps": 50,
                    "sigma_shift": 6.0,
                    "tiled": False,
                    "reference_image": first_frame,
                }

                if background_video_frames is not None:
                    pipe_kwargs["background_video"] = background_video_frames
                    pipe_kwargs["background_video_channels"] = BACKGROUND_VIDEO_CHANNELS

                result = pipe(**pipe_kwargs)

                # Save per clip name
                os.makedirs(sample_output_dir, exist_ok=True)

                save_video(result, os.path.join(sample_output_dir, "generated.mp4"), fps=15, quality=5)
                save_video(ground_truth_frames, os.path.join(sample_output_dir, "groundtruth.mp4"), fps=15, quality=5)

                bbox_frames = visualize_bbox(bbox_data, width=832, height=480)
                save_video(bbox_frames, os.path.join(sample_output_dir, "bbox.mp4"), fps=15, quality=5)

                if background_video_frames is not None:
                    save_video(background_video_frames, os.path.join(sample_output_dir, "background.mp4"), fps=15, quality=5)

                if (idx + 1) % 50 == 0 or idx == 0:
                    print(f"  Saved to: {sample_output_dir}")

            except Exception as e:
                print(f"  Error generating video: {e}")
                import traceback
                traceback.print_exc()
                continue

        except Exception as e:
            print(f"  Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nValidation completed! Results saved to: {validation_output_dir}")


if __name__ == "__main__":
    main()
