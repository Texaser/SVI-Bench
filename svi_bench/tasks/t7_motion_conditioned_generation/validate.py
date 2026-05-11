#!/usr/bin/env python3

import torch
import numpy as np
import os
import sys
import glob
import random
from PIL import Image, ImageDraw
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

def visualize_bbox(bbox_data, width=832, height=480):
    """
    Visualize bbox data as frames with bounding boxes drawn.
    
    Args:
        bbox_data: numpy array of shape (num_frames, num_players * 4)
                   where each player has [x1, y1, x2, y2] in normalized coordinates (0-1 range)
        width: frame width
        height: frame height
    
    Returns:
        List of PIL Images with bounding boxes drawn
    """
    frames = []
    num_frames = bbox_data.shape[0]
    num_coords = bbox_data.shape[1]
    num_players = num_coords // 4
    
    # Color palette for different players
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light Blue
        (255, 128, 128),# Pink
    ]
    
    for frame_idx in range(num_frames):
        # Create blank canvas
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw each player's bbox
        for player_idx in range(num_players):
            start_idx = player_idx * 4
            x1_norm, y1_norm, x2_norm, y2_norm = bbox_data[frame_idx, start_idx:start_idx+4]
            
            # Skip if bbox is invalid (all zeros or invalid coordinates)
            if x1_norm == 0 and y1_norm == 0 and x2_norm == 0 and y2_norm == 0:
                continue
            if x2_norm <= x1_norm or y2_norm <= y1_norm:
                continue
            if (x2_norm - x1_norm) < 0.01 or (y2_norm - y1_norm) < 0.01:
                continue
            
            # Convert normalized coordinates [x1, y1, x2, y2] to pixel coordinates
            # Bbox format is [x1, y1, x2, y2] where x1,y1 is top-left and x2,y2 is bottom-right
            x1 = int(x1_norm * width)
            y1 = int(y1_norm * height)
            x2 = int(x2_norm * width)
            y2 = int(y2_norm * height)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Draw rectangle
            color = colors[player_idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw player number
            draw.text((x1 + 5, y1 + 5), f"P{player_idx+1}", fill=color)
        
        frames.append(img)
    
    return frames

def main():
    # Get checkpoint path from command line
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        print("Usage: python Wan2.1-Fun-V1.1-1.3B-Control-bbox-background-validation.py <checkpoint_path>")
        sys.exit(1)

    # Validation parameters from environment variables
    VALIDATION_BBOX_FOLDER = os.environ.get(
        'VALIDATION_BBOX_FOLDER',
        '/mnt/bum/hanyi/repo/sports_detection/segment-anything-2-real-time/basketball_set/val_5.txt'  # Use dedicated validation set
    )
    VALIDATION_VIDEO_BASE = os.environ.get(
        'VALIDATION_VIDEO_BASE', 
        '/mnt/bum/hanyi/data/basketball_fps_15'  # Video files location
    )
    VALIDATION_BACKGROUND_VIDEO_BASE = os.environ.get(
        'VALIDATION_BACKGROUND_VIDEO_BASE',
        '/mnt/bum/hanyi/data/basketball_inpainting_video'  # Background video files location
    )
    NUM_VALIDATION_SAMPLES = int(os.environ.get('NUM_VALIDATION_SAMPLES', '3'))
    VALIDATION_NUM_FRAMES = int(os.environ.get('VALIDATION_NUM_FRAMES', '81'))
    VALIDATION_TIME_DIVISION_FACTOR = int(os.environ.get('VALIDATION_TIME_DIVISION_FACTOR', '1'))

    print(f"Loading LoRA checkpoint: {checkpoint_path}")
    print(f"Validation bbox folder: {VALIDATION_BBOX_FOLDER}")
    print(f"Validation video base: {VALIDATION_VIDEO_BASE}")
    print(f"Validation background video base: {VALIDATION_BACKGROUND_VIDEO_BASE}")
    print(f"Number of validation samples: {NUM_VALIDATION_SAMPLES}")
    
    # Channel configuration (Basketball mode: 16 bbox channels, background video in y)
    BBOX_CHANNELS = int(os.environ.get('BBOX_CHANNELS', '16'))  # Default to 16 to match training script
    BACKGROUND_VIDEO_CHANNELS = int(os.environ.get('BACKGROUND_VIDEO_CHANNELS', '8'))  # Not used (background is in y)
    print(f"BBox channels: {BBOX_CHANNELS}, Background video channels: {BACKGROUND_VIDEO_CHANNELS} (note: background video is in y, not control_latents)")
    
    # Overlay/separate toggle from environment (default overlay for 1.3B Control)
    use_overlay_env = os.environ.get('USE_OVERLAY_METHOD', '1')
    use_overlay_method = str(use_overlay_env).strip().lower() in ('1', 'true', 'yes', 'y')
    print(f"Use overlay method: {use_overlay_method} (env USE_OVERLAY_METHOD={use_overlay_env})")
    
    # BBox color mode from environment (default color)
    bbox_color_mode = os.environ.get('BBOX_COLOR_MODE', 'color')
    print(f"BBox color mode: {bbox_color_mode}")

    # Load pipeline with Wan2.1-Fun-V1.1-1.3B-Control model configs
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
    # Propagate overlay/separate toggle to pipeline attribute for units to read
    try:
        setattr(pipe, 'use_overlay_method', use_overlay_method)
    except Exception:
        pass

    # Load LoRA checkpoint (using same method as official script)
    if os.path.exists(checkpoint_path):
        try:
            # Load LoRA weights using the pipeline's load_lora method (like official script)
            pipe.load_lora(pipe.dit, checkpoint_path, alpha=1.0)
            print(f"✓ Loaded LoRA checkpoint: {checkpoint_path}")
            
            # Ensure bbox embedding layers are properly set up (similar to training)
            if hasattr(pipe.dit, 'bbox_embedding') and pipe.dit.bbox_embedding is not None:
                print("✓ bbox_embedding layer found and loaded")
            if hasattr(pipe.dit, 'bbox_spatial_embedding') and pipe.dit.bbox_spatial_embedding is not None:
                print("✓ bbox_spatial_embedding layer found and loaded")
                
        except Exception as e:
            print(f"⚠ Error loading LoRA checkpoint: {e}")
            print("Continuing with base model...")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    pipe.enable_vram_management()

    # Get output directory and checkpoint name for saving validation results
    output_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    validation_output_dir = os.path.join(output_dir, "validation", checkpoint_name)
    os.makedirs(validation_output_dir, exist_ok=True)

    # Find validation samples from bbox folder or txt file (support both .npz and .txt)
    if os.path.exists(VALIDATION_BBOX_FOLDER):
        bbox_files = []
        
        # Check if VALIDATION_BBOX_FOLDER is a file (txt list) or directory
        if os.path.isfile(VALIDATION_BBOX_FOLDER):
            # Read bbox paths from txt file (one path per line)
            print(f"Reading bbox paths from file: {VALIDATION_BBOX_FOLDER}")
            with open(VALIDATION_BBOX_FOLDER, 'r') as f:
                for line in f:
                    bbox_path = line.strip()
                    if bbox_path and (bbox_path.endswith('.npz') or bbox_path.endswith('.txt')):
                        if os.path.exists(bbox_path):
                            bbox_files.append(bbox_path)
                        else:
                            print(f"  ⚠ Bbox file not found: {bbox_path}")
        else:
            # Scan directory for bbox files
            bbox_files_npz = glob.glob(os.path.join(VALIDATION_BBOX_FOLDER, "**/*.npz"), recursive=True)
            bbox_files_txt = glob.glob(os.path.join(VALIDATION_BBOX_FOLDER, "**/*.txt"), recursive=True)
            bbox_files = bbox_files_npz + bbox_files_txt
        
        if len(bbox_files) == 0:
            print(f"⚠ No bbox files (.npz or .txt) found in {VALIDATION_BBOX_FOLDER}")
            sys.exit(1)
        
        print(f"✓ Found {len(bbox_files)} bbox files in validation set")
        
        # Randomly sample validation files
        num_samples = min(NUM_VALIDATION_SAMPLES, len(bbox_files))
        sampled_bbox_files = random.sample(bbox_files, num_samples)
        
        print(f"Generating {num_samples} validation videos...")
        
        for idx, bbox_path in enumerate(sampled_bbox_files):
            try:
                # Derive video path from bbox path
                # If bbox_path is a full path (from txt file), extract relative path
                # by finding the part after "basketball_mixsort_all_*" directory
                bbox_path_normalized = os.path.normpath(bbox_path)
                parts = bbox_path_normalized.split(os.sep)
                
                # Find the index of "basketball_mixsort_all_*" directory
                mixsort_idx = None
                for i, part in enumerate(parts):
                    if 'basketball_mixsort_all' in part:
                        mixsort_idx = i
                        break
                
                if mixsort_idx is not None:
                    # Extract relative path after mixsort directory
                    relative_parts = parts[mixsort_idx + 1:]
                    relative_path = os.path.join(*relative_parts) if relative_parts else ""
                else:
                    # Fallback: if VALIDATION_BBOX_FOLDER is a directory, use relpath
                    if os.path.isdir(VALIDATION_BBOX_FOLDER):
                        relative_path = os.path.relpath(bbox_path, VALIDATION_BBOX_FOLDER)
                    else:
                        # Last resort: use basename
                        relative_path = os.path.basename(bbox_path)
                
                # Convert bbox path to video path
                base_name = os.path.basename(relative_path)
                dir_name = os.path.dirname(relative_path)
                
                # Keep any trailing suffix like _0/_1; only replace extension .npz/.txt -> .mp4
                name_wo_ext, _ = os.path.splitext(base_name)
                video_name = name_wo_ext + '.mp4'
                video_relative = os.path.join(dir_name, video_name) if dir_name else video_name
                video_path = os.path.join(VALIDATION_VIDEO_BASE, video_relative)
                
                # Background video has the same relative path as main video
                background_video_path = os.path.join(VALIDATION_BACKGROUND_VIDEO_BASE, video_relative)
                
                print(f"\n[{idx+1}/{num_samples}] Processing:")
                print(f"  BBox: {bbox_path}")
                print(f"  Video: {video_path}")
                print(f"  Background Video: {background_video_path}")
                
                # Load bbox data (support both .npz and .txt formats)
                if bbox_path.endswith('.txt'):
                    # Load from txt format: frame_id,object_id,x1,y1,x2,y2,confidence,...
                    bbox_dict = {}  # {(frame_id, object_id): [x1, y1, x2, y2]}
                    max_frame_id = -1
                    unique_object_ids = set()
                    
                    with open(bbox_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(',')
                            if len(parts) < 6:
                                continue
                            try:
                                frame_id = int(parts[0])
                                object_id = int(parts[1])
                                x1 = float(parts[2])
                                y1 = float(parts[3])
                                x2 = float(parts[4])
                                y2 = float(parts[5])
                                
                                max_frame_id = max(max_frame_id, frame_id)
                                unique_object_ids.add(object_id)
                                
                                bbox_dict[(frame_id, object_id)] = [x1, y1, x2, y2]
                            except (ValueError, IndexError):
                                continue
                    
                    if max_frame_id < 0 or len(unique_object_ids) == 0:
                        print(f"  ❌ No valid bbox data found in {bbox_path}")
                        continue
                    
                    # Create mapping from object_id to array index
                    sorted_object_ids = sorted(unique_object_ids)
                    object_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(sorted_object_ids)}
                    num_objects = len(unique_object_ids)
                    
                    # Create array: (num_frames, num_objects * 4)
                    num_frames = max_frame_id + 1
                    bbox_data = np.zeros((num_frames, num_objects * 4), dtype=np.float32)
                    
                    # Fill in the data using the mapping
                    for (frame_id, object_id), coords in bbox_dict.items():
                        if frame_id < num_frames and object_id in object_id_to_idx:
                            array_idx = object_id_to_idx[object_id]
                            start_idx = array_idx * 4
                            bbox_data[frame_id, start_idx:start_idx+4] = coords
                    
                    print(f"  ✓ Loaded bbox from txt format: {num_frames} frames, {num_objects} objects")
                else:
                    # Load from npz format
                    bbox_npz = np.load(bbox_path)
                    bbox_keys = list(bbox_npz.keys())
                    print(f"  BBox keys available: {bbox_keys}")
                    
                    # Get bbox data
                    if 'arr_0' in bbox_keys:
                        bbox_data = bbox_npz['arr_0']
                    elif 'bboxes' in bbox_keys:
                        bbox_data = bbox_npz['bboxes']
                    else:
                        bbox_data = bbox_npz[bbox_keys[0]]
                
                print(f"  BBox shape: {bbox_data.shape}")
                
                # Align bbox frames with training sampling (num_frames, start_idx, step)
                desired_num_frames = VALIDATION_NUM_FRAMES
                if bbox_data.shape[0] != desired_num_frames:
                    start_idx = 0
                    step = VALIDATION_TIME_DIVISION_FACTOR
                    total_frames_bbox = bbox_data.shape[0]
                    sampled = []
                    for i in range(desired_num_frames):
                        frame_id = start_idx + i * step
                        if frame_id < total_frames_bbox:
                            sampled.append(bbox_data[frame_id])
                        else:
                            pad = sampled[-1] if len(sampled) > 0 else np.zeros(bbox_data.shape[1], dtype=bbox_data.dtype)
                            sampled.append(pad)
                    bbox_data = np.stack(sampled, axis=0)
                    print(f"  ✓ Resampled bbox to {bbox_data.shape}")
                
                # Load video frames (first frame + ground truth for comparison)
                if os.path.exists(video_path):
                    try:
                        # Read original video dimensions before center-crop-resize (for bbox alignment)
                        import imageio as _iio
                        _reader = _iio.get_reader(video_path)
                        _first = _reader.get_data(0)
                        orig_video_height, orig_video_width = _first.shape[:2]
                        _reader.close()
                        print(f"  Original video size: {orig_video_width}x{orig_video_height}")

                        video = VideoData(video_path, height=480, width=832)
                        total_video_frames = len(video)
                        
                        # bbox_data.shape[0] tells us how many frames of bbox data we have
                        # This should match the number of frames used during training
                        bbox_num_frames = bbox_data.shape[0]
                        
                        # During training, LoadVideo and LoadBBox use shared_sampler with:
                        # - start_idx: starting frame (could be 0 or random)
                        # - step: time_division_factor
                        # - frame_id = start_idx + i * step
                        start_idx = 0
                        step = VALIDATION_TIME_DIVISION_FACTOR
                        
                        # Extract ground truth frames matching bbox coverage
                        ground_truth_frames = []
                        for i in range(bbox_num_frames):
                            frame_idx = start_idx + i * step
                            if frame_idx < total_video_frames:
                                ground_truth_frames.append(video[frame_idx])
                            else:
                                # Pad with last frame if video is shorter
                                if ground_truth_frames:
                                    ground_truth_frames.append(ground_truth_frames[-1])
                                break
                        
                        # Use first frame from ground truth
                        first_frame = ground_truth_frames[0] if ground_truth_frames else video[0]
                        
                        if not hasattr(first_frame, 'mode') or first_frame.mode != 'RGB':
                            first_frame = first_frame.convert('RGB')
                        
                        print(f"  ✓ Video total frames: {total_video_frames}")
                        print(f"  ✓ BBox frames: {bbox_num_frames}")
                        print(f"  ✓ Loaded first frame: {first_frame.size}, mode: {first_frame.mode}")
                        print(f"  ✓ Loaded {len(ground_truth_frames)} ground truth frames")
                    except Exception as e:
                        print(f"  ❌ Error loading video: {e}")
                        continue
                else:
                    print(f"  ⚠ Video not found: {video_path}")
                    continue
                
                # Load background video frames
                background_video_frames = None
                if os.path.exists(background_video_path):
                    try:
                        background_video = VideoData(background_video_path, height=480, width=832)
                        total_bg_frames = len(background_video)
                        
                        # Extract background video frames matching bbox coverage
                        background_video_frames = []
                        for i in range(bbox_num_frames):
                            frame_idx = start_idx + i * step
                            if frame_idx < total_bg_frames:
                                frame = background_video[frame_idx]
                                if not hasattr(frame, 'mode') or frame.mode != 'RGB':
                                    frame = frame.convert('RGB')
                                background_video_frames.append(frame)
                            else:
                                # Pad with last frame if background video is shorter
                                if background_video_frames:
                                    background_video_frames.append(background_video_frames[-1])
                                else:
                                    # If background video is empty, pad with first frame from main video
                                    background_video_frames.append(first_frame)
                                break
                        
                        # Ensure we have enough frames
                        while len(background_video_frames) < bbox_num_frames:
                            background_video_frames.append(background_video_frames[-1] if background_video_frames else first_frame)
                        
                        print(f"  ✓ Background video total frames: {total_bg_frames}")
                        print(f"  ✓ Loaded {len(background_video_frames)} background video frames")
                    except Exception as e:
                        print(f"  ⚠ Error loading background video: {e}")
                        print(f"  Continuing without background video...")
                        background_video_frames = None
                else:
                    print(f"  ⚠ Background video not found: {background_video_path}")
                    print(f"  Continuing without background video...")
                    background_video_frames = None
                
                # Generate video with bbox and background video conditioning (same parameters as training)
                try:
                    # Convert bbox to tensor and move to correct device
                    bbox_tensor = torch.from_numpy(bbox_data).float()
                    if hasattr(pipe, 'device') and pipe.device is not None:
                        bbox_tensor = bbox_tensor.to(pipe.device)
                    
                    # Prepare pipeline arguments
                    pipe_kwargs = {
                        "prompt": "a realistic basketball game video",
                        "input_image": first_frame,
                        "bbox": bbox_tensor,
                        "bbox_channels": BBOX_CHANNELS,
                        "bbox_color_mode": bbox_color_mode,  # Use color-coded rendering mode ("noise" or "color")
                        "use_overlay_method": use_overlay_method,
                        "orig_video_width": orig_video_width,  # For bbox center-crop alignment
                        "orig_video_height": orig_video_height,
                        "height": 480,
                        "width": 832,
                        "num_frames": VALIDATION_NUM_FRAMES,
                        "cfg_scale": 2.5,  # Very low CFG to stay closer to first frame's distribution
                        "num_inference_steps": 50,
                        "sigma_shift": 6.0,  # Very high sigma_shift for stronger temporal consistency
                        "tiled": False,
                        "reference_image": first_frame,  # Reference image helps maintain appearance consistency
                    }
                    
                    # Add background video if available
                    if background_video_frames is not None:
                        pipe_kwargs["background_video"] = background_video_frames
                        pipe_kwargs["background_video_channels"] = BACKGROUND_VIDEO_CHANNELS
                        print(f"  ✓ Using background video with {len(background_video_frames)} frames")
                    
                    result = pipe(**pipe_kwargs)
                    
                    # Save generated video
                    output_filename = f"validation_sample_{idx+1:03d}_generated.mp4"
                    output_path = os.path.join(validation_output_dir, output_filename)
                    save_video(result, output_path, fps=15, quality=5)
                    print(f"  ✓ Generated validation video: {output_path}")
                    
                    # Save ground truth video for comparison
                    gt_filename = f"validation_sample_{idx+1:03d}_groundtruth.mp4"
                    gt_path = os.path.join(validation_output_dir, gt_filename)
                    save_video(ground_truth_frames, gt_path, fps=15, quality=5)
                    print(f"  ✓ Saved ground truth video: {gt_path}")
                    
                    # Visualize and save bbox video
                    bbox_frames = visualize_bbox(bbox_data, width=832, height=480)
                    bbox_filename = f"validation_sample_{idx+1:03d}_bbox.mp4"
                    bbox_path = os.path.join(validation_output_dir, bbox_filename)
                    save_video(bbox_frames, bbox_path, fps=15, quality=5)
                    print(f"  ✓ Saved bbox visualization: {bbox_path}")
                    
                    # Save background video for comparison (if available)
                    if background_video_frames is not None:
                        bg_filename = f"validation_sample_{idx+1:03d}_background.mp4"
                        bg_path = os.path.join(validation_output_dir, bg_filename)
                        save_video(background_video_frames, bg_path, fps=15, quality=5)
                        print(f"  ✓ Saved background video: {bg_path}")
                    
                except Exception as e:
                    print(f"  ❌ Error generating video: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
            except Exception as e:
                print(f"  ❌ Error processing sample: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n✓ Validation completed! Results saved to: {validation_output_dir}")
        
    else:
        print(f"❌ Validation bbox folder not found: {VALIDATION_BBOX_FOLDER}")
        sys.exit(1)

if __name__ == "__main__":
    main()
